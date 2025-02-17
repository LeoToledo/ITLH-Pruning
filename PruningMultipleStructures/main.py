# main.py

import argparse
import sys
import json
import gc
import os
import numpy as np
from keras.activations import *
from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.python.data import Dataset
import tensorflow as tf
from keras import backend as K

import rebuild_filters as rf
import rebuild_layers as rl
from pruning_criteria import criteria_filter as cf
from pruning_criteria import criteria_layer as cl
import random

sys.path.insert(0, '../utils')
import custom_functions as func
from layer_rotation import LayerRotationTracker
from custom_functions import data_augmentation

# Set seeds for reproducibility
random.seed(2)
np.random.seed(2)
tf.random.set_seed(2)

# Ensure that the "models" folder exists
if not os.path.exists("models"):
    os.makedirs("models")

# -------------------------------------------------------------------------
# Memory management setup
# -------------------------------------------------------------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
            )
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

set_global_policy('mixed_float16')

# -------------------------------------------------------------------------
# Global variables
# -------------------------------------------------------------------------
ARCHITECTURE_NAME = 'ResNet56'
CRITERION_FILTER = 'CKA'
CRITERION_LAYER = 'CKA'
P_FILTER = 0.08
P_LAYER = 2
MAX_EPOCHS = 1000
LEARNING_RATE = 0.01
MOMENTUM = 0.9
BATCH_SIZE = 64

# New globals for pruning timing:
MIN_INITIAL_TRAIN_EPOCHS = 1  # Minimum epochs before the first pruning check
MIN_POST_PRUNE_TRAIN_EPOCHS = 70  # Minimum epochs after each prune before checking again

# Parameters for the rotation tracker derivative:
ROTATION_STABLE_EPOCHS = 5              # Number of consecutive epochs with derivative below threshold
ROTATION_DERIVATIVE_THRESHOLD = 0.1       # Threshold (in degrees) for the derivative of weight rotation

# New globals for waiting after LR drop before pruning:
TRAIN_EPOCHS_AFTER_LR_DROP_10x = 15  # Number of epochs to train after dropping LR to LEARNING_RATE/10
TRAIN_EPOCHS_AFTER_LR_DROP_100x = 10  # Number of epochs to train after dropping LR further to LEARNING_RATE/100

# Global state variables for the LR drop process.
# lr_drop_stage: 0 = no drop, 1 = dropped to LEARNING_RATE/10, 2 = dropped to LEARNING_RATE/100.
lr_drop_stage = 0
epochs_since_lr_drop = 0

# -------------------------------------------------------------------------
# Modularized Model Saving Function
# -------------------------------------------------------------------------
def save_model_with_description(model, epoch, model_category, val_acc, flops):
    """
    Saves the given model to the "models" folder with a descriptive filename.

    Args:
        model: Keras model to be saved.
        epoch (int): Current epoch number.
        model_category (str): Descriptor for the model version (e.g. "final" or "pruned").
        val_acc (float): Validation accuracy.
        flops (int): FLOPS count.
    
    Returns:
        The path to which the model was saved.
    """
    filename = f"{model_category}_model_epoch_{epoch}_valacc_{val_acc:.4f}_flops_{flops}.h5"
    path = os.path.join("models", filename)
    model.save(path)
    print(f"{model_category.capitalize()} model for epoch {epoch} saved to {path}")
    return path

# -------------------------------------------------------------------------
# Other Utility Functions
# -------------------------------------------------------------------------
def setup_fine_tuning_cnn(model, learning_rate):
    """Setup the model for fine-tuning."""
    import keras
    sgd = keras.optimizers.SGD(
        learning_rate=learning_rate,
        decay=1e-6,
        momentum=0.9,
        nesterov=True
    )
    model.compile(
        loss='categorical_crossentropy',
        optimizer=sgd,
        metrics=['accuracy']
    )
    return model, []

def fine_tuning_epoch(model, x_train, y_train, batch_size, callbacks, epoch, recovery=False):
    """
    Train the model for one epoch using data augmentation.
    
    Args:
        model: Keras model.
        x_train, y_train: Training data.
        batch_size: Batch size.
        callbacks: Callbacks for model.fit.
        epoch: Current epoch.
        recovery (bool): If True, force the learning rate to LEARNING_RATE (i.e. a higher LR) for recovery.
        
    Returns:
        The training accuracy from this epoch.
    """
    global lr_drop_stage
    if recovery:
        current_lr = LEARNING_RATE
        print(f"Recovery mode active. Forcing learning rate: {current_lr}")
    else:
        current_lr = float(K.get_value(model.optimizer.learning_rate))
    K.set_value(model.optimizer.learning_rate, current_lr)
    print(f"Current learning rate: {current_lr}")

    # Data augmentation: replicate training data k times.
    k = 3
    X_aug = np.tile(x_train, (k, 1, 1, 1))
    y_aug = np.tile(y_train, (k, 1))
    datagen = func.generate_data_augmentation(x_train)

    history = model.fit(
        datagen.flow(X_aug, y_aug, batch_size=batch_size, seed=2, shuffle=True),
        verbose=1,
        callbacks=callbacks,
        epochs=1
    )
    return history.history['accuracy'][-1]

def predict_in_batches(model, X, batch_size=32):
    """Perform prediction in batches for memory efficiency."""
    dataset = tf.data.Dataset.from_tensor_slices(X)\
        .batch(batch_size)\
        .prefetch(tf.data.AUTOTUNE)
    predictions = []
    for batch in dataset:
        batch_pred = model.predict(batch, verbose=0)
        predictions.append(batch_pred)
    return np.vstack(predictions)

def statistics(model):
    """Display model statistics."""
    n_params = model.count_params()
    n_filters = func.count_filters(model)
    flops, _ = func.compute_flops(model)
    blocks = rl.count_blocks(model)
    memory = func.memory_usage(1, model)
    print(f'Blocks {blocks} Parameters [{n_params}] Filters [{n_filters}] '
          f'FLOPS [{flops}] Memory [{memory:.6f}]', flush=True)
    return flops

# -------------------------------------------------------------------------
# Modified Prune Function
# -------------------------------------------------------------------------
def prune(model, p_filter, p_layer, criterion_filter, criterion_layer, X_train, y_train):
    """Perform pruning on filters and layers and return the computed scores as well."""
    gc.collect()
    tf.keras.backend.clear_session()
    
    allowed_layers_filters = rf.layer_to_prune_filters(model)
    filter_method = cf.criteria(criterion_filter)
    scores_filter = filter_method.scores(model, X_train, y_train, allowed_layers_filters)
    pruned_model_filter = rf.rebuild_network(model, scores_filter, p_filter)
    
    allowed_layers = rl.blocks_to_prune(model)
    layer_method = cl.criteria(criterion_layer)
    scores_layer = layer_method.scores(model, X_train, y_train, allowed_layers)
    pruned_model_layer = rl.rebuild_network(model, scores_layer, p_layer)
    
    return pruned_model_filter, pruned_model_layer, scores_filter, scores_layer

# -------------------------------------------------------------------------
# Modified Winner Pruned Model Function with Extra "CKA" Criteria
# -------------------------------------------------------------------------
import numpy as np  # Ensure numpy is imported

def winner_pruned_model(pruned_model_filter, pruned_model_layer, scores_filter, scores_layer, best_pruned_criteria='flops'):
    """
    Select the best pruned model based on either FLOPS or the CKA criteria.
    
    For 'flops', the model with the lower FLOPS is returned.
    For 'CKA', the aggregated (mean) CKA score from the criteria is used.
    """
    if best_pruned_criteria == 'flops':
        layers_current_flops, _ = func.compute_flops(pruned_model_layer)
        layers_filter_flops, _ = func.compute_flops(pruned_model_filter)
        if layers_current_flops < layers_filter_flops:
            return pruned_model_layer, 'layer'
        return pruned_model_filter, 'filter'
    
    elif best_pruned_criteria == 'CKA':
        # Aggregate CKA scores from the layer pruning
        # Here, scores_layer is a list of tuples: (layer_idx, score) where score is a scalar.
        layer_scores = [score for (_, score) in scores_layer]
        agg_layer_score = np.mean(layer_scores) if layer_scores else float('inf')
        
        # Aggregate CKA scores from filter pruning
        # Here, scores_filter is a list of tuples: (layer_idx, list_of_scores) for each filter.
        filter_scores = []
        for (_, score_list) in scores_filter:
            filter_scores.extend(score_list)
        agg_filter_score = np.mean(filter_scores) if filter_scores else float('inf')
        
        print(f"[Winner Selection: CKA] Aggregated layer score: {agg_layer_score:.4f}, Aggregated filter score: {agg_filter_score:.4f}")
        if agg_layer_score < agg_filter_score:
            return pruned_model_layer, 'layer'
        else:
            return pruned_model_filter, 'filter'
    
    else:
        raise ValueError(f'Unknown best_pruned_criteria [{best_pruned_criteria}]')

# -------------------------------------------------------------------------
# Main Training Loop
# -------------------------------------------------------------------------
if __name__ == '__main__':
    print(f'Architecture [{ARCHITECTURE_NAME}] p_filter[{P_FILTER}] p_layer[{P_LAYER}]', flush=True)

    # Load data and model.
    X_train, y_train, X_val, y_val = func.cifar_resnet_data(validation_split=0.2, random_state=42)
    model = func.load_model(ARCHITECTURE_NAME)
    rf.architecture_name = ARCHITECTURE_NAME
    rl.architecture_name = ARCHITECTURE_NAME

    # Initial statistics.
    init_flops = statistics(model)
    y_pred_val_init = predict_in_batches(model, X_val, BATCH_SIZE)
    initial_acc_val = accuracy_score(
        np.argmax(y_val, axis=1),
        np.argmax(y_pred_val_init, axis=1)
    )
    print(f'Unpruned [{ARCHITECTURE_NAME}] Validation Accuracy [{initial_acc_val}]')

    # Configure the rotation tracker with the derivative logic.
    rotation_tracker = LayerRotationTracker(
        model=model,
        layer_names=None,
        derivative_threshold=ROTATION_DERIVATIVE_THRESHOLD,
        stable_epochs_needed=ROTATION_STABLE_EPOCHS
    )

    # For logging purposes.
    best_val_acc = initial_acc_val
    log_file = open("train_log.jsonl", "w", encoding="utf-8")

    # Write hyperparameters header (except the learning rate, which changes) as the first block.
    hyperparams = {
        "ARCHITECTURE_NAME": ARCHITECTURE_NAME,
        "CRITERION_FILTER": CRITERION_FILTER,
        "CRITERION_LAYER": CRITERION_LAYER,
        "P_FILTER": P_FILTER,
        "P_LAYER": P_LAYER,
        "MAX_EPOCHS": MAX_EPOCHS,
        "MOMENTUM": MOMENTUM,
        "BATCH_SIZE": BATCH_SIZE,
        "MIN_INITIAL_TRAIN_EPOCHS": MIN_INITIAL_TRAIN_EPOCHS,
        "MIN_POST_PRUNE_TRAIN_EPOCHS": MIN_POST_PRUNE_TRAIN_EPOCHS,
        "ROTATION_STABLE_EPOCHS": ROTATION_STABLE_EPOCHS,
        "ROTATION_DERIVATIVE_THRESHOLD": ROTATION_DERIVATIVE_THRESHOLD,
        "TRAIN_EPOCHS_AFTER_LR_DROP_10x": TRAIN_EPOCHS_AFTER_LR_DROP_10x,
        "TRAIN_EPOCHS_AFTER_LR_DROP_100x": TRAIN_EPOCHS_AFTER_LR_DROP_100x
    }
    log_file.write(json.dumps({"hyperparameters": hyperparams}) + "\n")

    # Initial model setup for fine-tuning.
    model, callbacks = setup_fine_tuning_cnn(model, LEARNING_RATE)

    # Variable to track the epoch of the last prune.
    last_prune_epoch = 0

    # Main training loop.
    for epoch_idx in range(1, MAX_EPOCHS + 1):
        print(f"\n==== EPOCH {epoch_idx} / {MAX_EPOCHS} ====")

        if epoch_idx % 5 == 0:
            gc.collect()
            tf.keras.backend.clear_session()

        # Variable to hold pruning info to log (only when pruning happens).
        pruning_info_to_log = None

        # Determine recovery mode: within MIN_POST_PRUNE_TRAIN_EPOCHS after last prune.
        recovery = (last_prune_epoch > 0 and epoch_idx < last_prune_epoch + MIN_POST_PRUNE_TRAIN_EPOCHS)

        # Training epoch.
        current_acc_train = fine_tuning_epoch(model, X_train, y_train, BATCH_SIZE, callbacks, epoch_idx, recovery=recovery)

        # Validation.
        y_pred_val = predict_in_batches(model, X_val, BATCH_SIZE)
        current_acc_val = accuracy_score(np.argmax(y_val, axis=1), np.argmax(y_pred_val, axis=1))
        print(f"Epoch [{epoch_idx}] Train Accuracy: {current_acc_train:.4f} | Validation Accuracy: {current_acc_val:.4f}")

        # Update best_val_acc for logging.
        if current_acc_val > best_val_acc:
            best_val_acc = current_acc_val
            print(f"New best validation accuracy: {current_acc_val:.4f}")

        # Update the tracker to compute the derivative and check for stability.
        stable, angle = rotation_tracker.update_and_check_stability(model=model)
        # Prevent pruning if the minimum training epochs have not passed.
        if (last_prune_epoch == 0 and epoch_idx < MIN_INITIAL_TRAIN_EPOCHS) or \
           (last_prune_epoch > 0 and epoch_idx < last_prune_epoch + MIN_POST_PRUNE_TRAIN_EPOCHS):
            stable = False

        # Compute derivative from the tracker's log.
        if len(rotation_tracker.angles) > 1:
            derivative = abs(rotation_tracker.angles[-1] - rotation_tracker.angles[-2])
        else:
            derivative = None

        # Get the current learning rate from the model's optimizer.
        current_lr_value = float(K.get_value(model.optimizer.learning_rate))

        # Check and manage the LR drop and waiting period for pruning.
        if stable:
            if lr_drop_stage == 0:
                print(f"[Epoch {epoch_idx}] Stability triggered. Dropping learning rate by 10x (to {LEARNING_RATE/10}) for {TRAIN_EPOCHS_AFTER_LR_DROP_10x} epochs before further drop.")
                lr_drop_stage = 1
                epochs_since_lr_drop = 0
                K.set_value(model.optimizer.learning_rate, LEARNING_RATE / 10)
            elif lr_drop_stage == 1:
                epochs_since_lr_drop += 1
                print(f"[Epoch {epoch_idx}] Waiting after 10x LR drop: {epochs_since_lr_drop}/{TRAIN_EPOCHS_AFTER_LR_DROP_10x} epochs.")
                if epochs_since_lr_drop >= TRAIN_EPOCHS_AFTER_LR_DROP_10x:
                    print(f"[Epoch {epoch_idx}] Waiting period complete for 10x drop. Dropping learning rate further by 10x (to {LEARNING_RATE/100}) for {TRAIN_EPOCHS_AFTER_LR_DROP_100x} epochs.")
                    lr_drop_stage = 2
                    epochs_since_lr_drop = 0
                    K.set_value(model.optimizer.learning_rate, LEARNING_RATE / 100)
            elif lr_drop_stage == 2:
                epochs_since_lr_drop += 1
                print(f"[Epoch {epoch_idx}] Waiting after 100x LR drop: {epochs_since_lr_drop}/{TRAIN_EPOCHS_AFTER_LR_DROP_100x} epochs.")
                if epochs_since_lr_drop >= TRAIN_EPOCHS_AFTER_LR_DROP_100x:
                    print(f"[Epoch {epoch_idx}] Waiting period complete. Proceeding with pruning.")
                    
                    # Perform pruning and also obtain the scores from the criteria.
                    pruned_model_filter, pruned_model_layer, scores_filter, scores_layer = prune(
                        model, P_FILTER, P_LAYER,
                        CRITERION_FILTER, CRITERION_LAYER,
                        X_train, y_train
                    )
                    
                    # Use the extra 'CKA' option if desired.
                    # Set best_pruned_criteria to 'CKA' to use the CKA aggregation.
                    model, chosen_structure = winner_pruned_model(
                        pruned_model_filter, pruned_model_layer, scores_filter, scores_layer,
                        best_pruned_criteria='CKA'
                    )
                    pruning_info_to_log = chosen_structure

                    # Re-compile the newly pruned model and reset the learning rate.
                    model, callbacks = setup_fine_tuning_cnn(model, LEARNING_RATE)
                    K.set_value(model.optimizer.learning_rate, LEARNING_RATE)
                    
                    # Re-evaluate after pruning.
                    new_acc_val = accuracy_score(
                        np.argmax(y_val, axis=1),
                        np.argmax(predict_in_batches(model, X_val, BATCH_SIZE), axis=1)
                    )
                    current_flops = statistics(model)
                    print(f"Post-prune validation accuracy: {new_acc_val:.4f}")
                    
                    # Save the pruned model.
                    save_model_with_description(model, epoch_idx, "pruned", new_acc_val, current_flops)
                    
                    # Update last_prune_epoch.
                    last_prune_epoch = epoch_idx

                    # Reset LR drop state.
                    lr_drop_stage = 0
                    epochs_since_lr_drop = 0

                    # Reinitialize the rotation tracker with the pruned model.
                    rotation_tracker = LayerRotationTracker(
                        model=model,
                        layer_names=None,
                        derivative_threshold=ROTATION_DERIVATIVE_THRESHOLD,
                        stable_epochs_needed=ROTATION_STABLE_EPOCHS
                    )
        else:
            if lr_drop_stage > 0:
                print(f"[Epoch {epoch_idx}] Stability lost. Reverting learning rate to original value.")
                lr_drop_stage = 0
                epochs_since_lr_drop = 0
                K.set_value(model.optimizer.learning_rate, LEARNING_RATE)

        # Compute FLOPS and prepare a training log record.
        epoch_flops, _ = func.compute_flops(model)
        train_record = {
            "epoch": epoch_idx,
            "train_acc": round(float(current_acc_train), 4),
            "val_acc": round(float(current_acc_val), 4),
            "flops": int(epoch_flops),
            "angle": round(float(angle), 4) if angle is not None else None,
            "derivative": round(float(derivative), 4) if derivative is not None else None,
            "current_lr": round(current_lr_value, 4),
            "pruning_info": pruning_info_to_log
        }
        log_file.write(json.dumps(train_record) + "\n")
        log_file.flush()
