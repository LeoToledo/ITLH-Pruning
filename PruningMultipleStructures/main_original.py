# main.py

import argparse
import sys
import json
import gc
import numpy as np
from keras.activations import *
from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.mixed_precision import set_global_policy
import tensorflow as tf

import rebuild_filters as rf
import rebuild_layers as rl
from pruning_criteria import criteria_filter as cf
from pruning_criteria import criteria_layer as cl

sys.path.insert(0, '../utils')
import custom_functions as func
from PruningMultipleStructures.layer_rotation_original import LayerRotationTracker

# IMPORTANT: We only import the data_augmentation function from custom_functions
from custom_functions import data_augmentation

# -------------------------------------------------------------------------
# Memory management setup
# -------------------------------------------------------------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            # Enable memory growth
            tf.config.experimental.set_memory_growth(gpu, True)
            # Optional: Set memory limit (in MB)
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
            )
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# Optional: Use mixed precision
set_global_policy('mixed_float16')

# -------------------------------------------------------------------------
# Global variables
# -------------------------------------------------------------------------
ARCHITECTURE_NAME = 'ResNet56'
CRITERION_FILTER = 'random'
CRITERION_LAYER = 'random'
P_FILTER = 0.08
P_LAYER = 2
MAX_EPOCHS = 500

# Keep your global LR and batch size here
LEARNING_RATE = 0.001
MOMENTUM = 0.9
BATCH_SIZE = 512
ROTATION_START_EPOCH = 600
ROTATION_ANGLE_THRESHOLD = 1
ROTATION_STABLE_EPOCHS = 200

# -------------------------------------------------------------------------
# New Fine-Tuning Setup & Single-Epoch Functions
# -------------------------------------------------------------------------
def setup_fine_tuning_cnn(model, learning_rate):
    """
    Set up the model and callbacks for fine tuning
    (Compile the model, create LR schedule and callbacks, return both)
    """
    import custom_callbacks
    import keras

    # Learning rate schedule
    schedule = [(100, learning_rate / 10), (150, learning_rate / 100)]
    lr_scheduler = custom_callbacks.LearningRateScheduler(init_lr=learning_rate, schedule=schedule)
    callbacks = [lr_scheduler]

    # Compile model with SGD
    sgd = keras.optimizers.SGD(learning_rate=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model, callbacks


def fine_tuning_epoch(model, x_train, y_train, batch_size, callbacks):
    """
    Trains the model for exactly one epoch using data augmentation (tripling the data).
    Returns the final training accuracy of that epoch.
    """
    from tensorflow.python.data import Dataset

    # Data augmentation: triple your training data for the single epoch
    y_tmp = np.concatenate((y_train, y_train, y_train))
    x_tmp = np.concatenate(
        (data_augmentation(x_train),
         data_augmentation(x_train),
         data_augmentation(x_train))
    )

    # Create tf.data.Dataset from augmented data
    x_tmp_ds = Dataset.from_tensor_slices((x_tmp, y_tmp))\
                      .shuffle(4 * batch_size)\
                      .batch(batch_size)

    # Fit the model for one epoch
    history = model.fit(
        x_tmp_ds,
        verbose=1,
        callbacks=callbacks,
        epochs=1
    )

    # Return the training accuracy for that epoch
    train_acc = history.history['accuracy'][-1]
    return train_acc

# -------------------------------------------------------------------------
# Auxiliary Functions
# -------------------------------------------------------------------------
def predict_in_batches(model, X, batch_size=32):
    """
    Memory-efficient batch prediction
    """
    dataset = tf.data.Dataset.from_tensor_slices(X)\
        .batch(batch_size)\
        .prefetch(tf.data.AUTOTUNE)
    
    predictions = []
    for batch in dataset:
        batch_pred = model.predict(batch, verbose=0)
        predictions.append(batch_pred)
    
    return np.vstack(predictions)


def statistics(model):
    """
    Logs model statistics
    """
    n_params = model.count_params()
    n_filters = func.count_filters(model)
    flops, _ = func.compute_flops(model)
    blocks = rl.count_blocks(model)
    memory = func.memory_usage(1, model)
    
    print(f'Blocks {blocks} Parameters [{n_params}] Filters [{n_filters}] '
          f'FLOPS [{flops}] Memory [{memory:.6f}]', flush=True)
    return flops


def prune(model, p_filter, p_layer, criterion_filter, criterion_layer, X_train, y_train):
    """
    Prunes filters and layers
    """
    # Clear memory before pruning
    gc.collect()
    tf.keras.backend.clear_session()
    
    # Filter pruning
    allowed_layers_filters = rf.layer_to_prune_filters(model)
    filter_method = cf.criteria(criterion_filter)
    scores_filter = filter_method.scores(model, X_train, y_train, allowed_layers_filters)
    pruned_model_filter = rf.rebuild_network(model, scores_filter, p_filter)
    
    # Layer pruning
    allowed_layers = rl.blocks_to_prune(model)
    layer_method = cl.criteria(criterion_layer)
    scores_layer = layer_method.scores(model, X_train, y_train, allowed_layers)
    pruned_model_layer = rl.rebuild_network(model, scores_layer, p_layer)
    
    return pruned_model_filter, pruned_model_layer


def winner_pruned_model(pruned_model_filter, pruned_model_layer, best_pruned_criteria='flops'):
    """
    Selects better pruned model
    """
    layers_current_flops, _ = func.compute_flops(pruned_model_layer)
    layers_filter_flops, _ = func.compute_flops(pruned_model_filter)
    if best_pruned_criteria == 'flops':
        if layers_current_flops < layers_filter_flops:
            return pruned_model_layer, 'layer'
        return pruned_model_filter, 'filter'
    raise ValueError(f'Unknown best_pruned_criteria [{best_pruned_criteria}]')

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
if __name__ == '__main__':
    np.random.seed(2)
    print(f'Architecture [{ARCHITECTURE_NAME}] p_filter[{P_FILTER}] p_layer[{P_LAYER}]', flush=True)

    # ---------------------------------------------------------------------
    # Load Data & Model
    # ---------------------------------------------------------------------
    X_train, y_train, X_test, y_test, X_val, y_val = func.cifar_resnet_data(
        debug=False, validation_set=True
    )
    
    model = func.load_model(ARCHITECTURE_NAME)
    rf.architecture_name = ARCHITECTURE_NAME
    rl.architecture_name = ARCHITECTURE_NAME

    # ---------------------------------------------------------------------
    # Initial Stats
    # ---------------------------------------------------------------------
    init_flops = statistics(model)
    y_pred_test_init = predict_in_batches(model, X_test, BATCH_SIZE)
    initial_acc_test = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred_test_init, axis=1))
    print(f'Unpruned [{ARCHITECTURE_NAME}] Test Accuracy [{initial_acc_test}]')

    y_pred_val_init = predict_in_batches(model, X_val, BATCH_SIZE)
    initial_acc_val = accuracy_score(
        np.argmax(y_val, axis=1),
        np.argmax(y_pred_val_init, axis=1)
    )
    print(f'Unpruned [{ARCHITECTURE_NAME}] Validation Accuracy [{initial_acc_val}]')

    # ---------------------------------------------------------------------
    # Rotation Tracker
    # ---------------------------------------------------------------------
    rotation_tracker = LayerRotationTracker(
        model=model,
        layer_names=None,
        start_epoch=ROTATION_START_EPOCH,
        angle_threshold=ROTATION_ANGLE_THRESHOLD,
        stable_epochs_needed=ROTATION_STABLE_EPOCHS
    )

    best_val_acc = initial_acc_val
    log_file = open("train_log.jsonl", "w", encoding="utf-8")

    # ---------------------------------------------------------------------
    # Setup Fine Tuning (Compile + Callbacks) - only once
    # ---------------------------------------------------------------------
    model, callbacks = setup_fine_tuning_cnn(model, LEARNING_RATE)

    # ---------------------------------------------------------------------
    # Main Training Loop (fine-tuning + pruning + rotation check)
    # ---------------------------------------------------------------------
    try:
        final_model_for_each_epoch = []
        for epoch_idx in range(1, MAX_EPOCHS + 1):
            print(f"\n==== EPOCH {epoch_idx} / {MAX_EPOCHS} ====")

            # Memory cleanup every 5 epochs
            if epoch_idx % 5 == 0:
                gc.collect()
                tf.keras.backend.clear_session()

            # 1) Fine-Tuning for 1 Epoch
            current_acc_train = fine_tuning_epoch(model, X_train, y_train, BATCH_SIZE, callbacks)

            # 2) Compute Test Accuracy
            y_pred_test = predict_in_batches(model, X_test, BATCH_SIZE)
            current_acc_test = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred_test, axis=1))
            print(f"Epoch [{epoch_idx}] Train Accuracy: {current_acc_train:.4f} | Test Accuracy: {current_acc_test:.4f}")

            # 3) Compute Validation Accuracy
            y_pred_val = predict_in_batches(model, X_val, BATCH_SIZE)
            current_acc_val = accuracy_score(np.argmax(y_val, axis=1), np.argmax(y_pred_val, axis=1))
            print(f"Epoch [{epoch_idx}] Validation Accuracy: {current_acc_val:.4f}")

            # 4) Check if better than best_val_acc & save model
            if current_acc_val > best_val_acc:
                best_val_acc = current_acc_val
                model.save("best_model.h5")
                print(f"Validation accuracy improved to {current_acc_val:.4f}. Model saved.")

            # 5) Rotation check
            stable, angle = rotation_tracker.update_and_check_stability(
                model=model,
                current_epoch=epoch_idx,
                total_epochs=MAX_EPOCHS
            )

            # 6) Compute FLOPs
            epoch_flops, _ = func.compute_flops(model)
            train_record = {
                "epoch": epoch_idx,
                "train_acc": float(current_acc_train),
                "val_acc": float(current_acc_val),
                "test_acc": float(current_acc_test),
                "flops": int(epoch_flops),
                "angle": float(angle),
                "pruning_info": None
            }

            # 7) If stable, prune
            if stable:
                print(f"[Epoch {epoch_idx}] Rotation is stable! Initiating pruning...", flush=True)
                
                gc.collect()
                tf.keras.backend.clear_session()
                
                pruned_model_filter, pruned_model_layer = prune(
                    model, P_FILTER, P_LAYER,
                    CRITERION_FILTER, CRITERION_LAYER,
                    X_train, y_train
                )
                
                model, chosen_structure = winner_pruned_model(
                    pruned_model_filter, pruned_model_layer,
                    best_pruned_criteria='flops'
                )

                model.save("pruned_model.h5")
                print("Pruned model saved as pruned_model.h5.")

                # Re-check val/test after pruning
                new_acc_test = accuracy_score(
                    np.argmax(y_test, axis=1),
                    np.argmax(predict_in_batches(model, X_test, BATCH_SIZE), axis=1)
                )
                new_acc_val = accuracy_score(
                    np.argmax(y_val, axis=1),
                    np.argmax(predict_in_batches(model, X_val, BATCH_SIZE), axis=1)
                )

                current_flops = statistics(model)
                best_val_acc = new_acc_val
                model.save("best_model.h5")

                # Re-init rotation tracker
                rotation_tracker = LayerRotationTracker(
                    model=model,
                    layer_names=None,
                    start_epoch=ROTATION_START_EPOCH,
                    angle_threshold=ROTATION_ANGLE_THRESHOLD,
                    stable_epochs_needed=ROTATION_STABLE_EPOCHS
                )

                train_record["pruning_info"] = (
                    f"Pruned with structure={chosen_structure}, new_val_acc={new_acc_val:.4f}"
                )

            # Write JSON log
            log_file.write(json.dumps(train_record) + "\n")
            log_file.flush()

            final_model_for_each_epoch.append(model)

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        # Save the final model in case of crash
        model.save("final_model_crash.h5")
    
    finally:
        log_file.close()
