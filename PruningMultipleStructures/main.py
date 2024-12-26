import argparse
import sys
import json
import numpy as np
from keras.activations import *
from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers import SGD

import rebuild_filters as rf
import rebuild_layers as rl
from pruning_criteria import criteria_filter as cf
from pruning_criteria import criteria_layer as cl

# Insert your custom utils folder
sys.path.insert(0, '../utils')
import custom_functions as func

# Import our new layer rotation tracker
from layer_rotation import LayerRotationTracker

# Global variables
ARCHITECTURE_NAME = 'ResNet56'
CRITERION_FILTER = 'random'
CRITERION_LAYER = 'random'
P_FILTER = 0.08
P_LAYER = 2
MAX_EPOCHS = 500
LEARNING_RATE = 0.005
MOMENTUM = 0.9
BATCH_SIZE = 256
ROTATION_start_epoch = 2
ROTATION_ANGLE_THRESHOLD = 50
ROTATION_STABLE_EPOCHS = 2

def statistics(model):
    """
    Logs model statistics: #params, #filters, FLOPS, memory, etc.
    """
    n_params = model.count_params()
    n_filters = func.count_filters(model)
    flops, _ = func.compute_flops(model)
    blocks = rl.count_blocks(model)

    memory = func.memory_usage(1, model)
    print('Blocks {} Number of Parameters [{}] Number of Filters [{}] FLOPS [{}] '
          'Memory [{:.6f}]'.format(blocks, n_params, n_filters, flops, memory), flush=True)
    return flops  # Return flops so we can log it

def prune(model, p_filter, p_layer, criterion_filter, criterion_layer, X_train, y_train):
    """
    Prunes filters and layers in two separate pruned models, then returns both.
    """
    # 1) Filter-level pruning
    allowed_layers_filters = rf.layer_to_prune_filters(model)
    filter_method = cf.criteria(criterion_filter)
    scores_filter = filter_method.scores(model, X_train, y_train, allowed_layers_filters)
    pruned_model_filter = rf.rebuild_network(model, scores_filter, p_filter)
    
    # 2) Layer-level pruning
    allowed_layers = rl.blocks_to_prune(model)
    layer_method = cl.criteria(criterion_layer)
    scores_layer = layer_method.scores(model, X_train, y_train, allowed_layers)
    pruned_model_layer = rl.rebuild_network(model, scores_layer, p_layer)
    
    return pruned_model_filter, pruned_model_layer

def winner_pruned_model(pruned_model_filter, pruned_model_layer, best_pruned_criteria='flops'):
    """
    Picks whichever pruned model is 'better', typically by having fewer FLOPs.
    """
    layers_current_flops, _ = func.compute_flops(pruned_model_layer)
    layers_filter_flops, _ = func.compute_flops(pruned_model_filter)
    if best_pruned_criteria == 'flops':
        if layers_current_flops < layers_filter_flops:
            return pruned_model_layer, 'layer'
        return pruned_model_filter, 'filter'
    else:
        raise ValueError('Unknown best_pruned_criteria [{}]'.format(best_pruned_criteria))

def fine_tune(model, X_train, y_train, X_test, y_test):
    """
    Trains the model for a single epoch and returns:
      model, test_accuracy, train_accuracy
    """
    optimizer = SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM, nesterov=True)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(
        X_train, y_train,
        epochs=1,
        batch_size=BATCH_SIZE,
        verbose=1
    )
    # Train accuracy from last epoch
    train_acc = history.history['accuracy'][-1]

    y_pred_test = model.predict(X_test, verbose=0)
    test_acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred_test, axis=1))
    return model, test_acc, train_acc

def should_prune(current_iteration, should_prune_criteria='fixed'):
    """
    Optional function that returns True if we should prune at this iteration.
    """
    if should_prune_criteria == 'fixed':
        prune_interval = 5
        return (current_iteration % prune_interval == 0)
    return False

if __name__ == '__main__':
    np.random.seed(2)

    print('Architecture [{}] p_filter[{}] p_layer[{}]'.format(ARCHITECTURE_NAME, P_FILTER, P_LAYER), flush=True)

    # Load CIFAR data, including validation
    X_train, y_train, X_test, y_test, X_val, y_val = func.cifar_resnet_data(debug=False, validation_set=True)
    
    # Load an (untrained or pre-trained) model
    model = func.load_model(ARCHITECTURE_NAME)

    # Set the architecture name in the rebuild modules
    rf.architecture_name = ARCHITECTURE_NAME
    rl.architecture_name = ARCHITECTURE_NAME

    # Initial stats
    init_flops = statistics(model)  # This returns the flops
    y_pred_test_init = model.predict(X_test, verbose=0)
    initial_acc_test = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred_test_init, axis=1))
    print('Unpruned [{}] Test Accuracy [{}]'.format(ARCHITECTURE_NAME, initial_acc_test))

    # Initial validation accuracy
    y_pred_val_init = model.predict(X_val, verbose=0)
    initial_acc_val = accuracy_score(
        np.argmax(y_val, axis=1),
        np.argmax(y_pred_val_init, axis=1)
    )
    print(f'Unpruned [{ARCHITECTURE_NAME}] Validation Accuracy [{initial_acc_val}]')

    # Initialize the layer rotation tracker
    rotation_tracker = LayerRotationTracker(
        model=model,
        layer_names=None,
        start_epoch=ROTATION_start_epoch,
        angle_threshold=ROTATION_ANGLE_THRESHOLD,
        stable_epochs_needed=ROTATION_STABLE_EPOCHS
    )

    # For best validation accuracy saving
    best_val_acc = initial_acc_val

    # Open JSONL file in append mode
    log_file = open("train_log.jsonl", "w", encoding="utf-8")

    final_model_for_each_epoch = []
    for epoch_idx in range(1, MAX_EPOCHS + 1):
        print(f"\n==== EPOCH {epoch_idx} / {MAX_EPOCHS} ====")

        # 1) Fine-tune model for one epoch
        model, current_acc_test, current_acc_train = fine_tune(model, X_train, y_train, X_test, y_test)
        print(f"Epoch [{epoch_idx}] Test Accuracy: {current_acc_test:.4f}")

        # 2) Compute accuracy on validation set
        y_pred_val = model.predict(X_val, verbose=0)
        current_acc_val = accuracy_score(np.argmax(y_val, axis=1), np.argmax(y_pred_val, axis=1))
        print(f"Epoch [{epoch_idx}] Validation Accuracy: {current_acc_val:.4f}")

        # 3) Check if validation accuracy improved => save model
        if current_acc_val > best_val_acc:
            best_val_acc = current_acc_val
            model.save("best_model.h5")
            print(f"Validation accuracy improved to {current_acc_val:.4f}. Model saved.")

        # 4) Rotation stability check => also retrieve angle
        stable, angle = rotation_tracker.update_and_check_stability(
            model=model,
            current_epoch=epoch_idx,
            total_epochs=MAX_EPOCHS
        )

        # 5) Get FLOPs for logging
        epoch_flops, _ = func.compute_flops(model)

        # 6) Prepare JSON record
        train_record = {
            "epoch": epoch_idx,
            "train_acc": float(current_acc_train),
            "val_acc": float(current_acc_val),
            "test_acc": float(current_acc_test),
            "flops": int(epoch_flops),
            "angle": float(angle),
            "pruning_info": None
        }

        # 7) If rotation is stable => prune
        if stable:
            print(f"[Epoch {epoch_idx}] Rotation is stable! Initiating pruning...", flush=True)
            pruned_model_filter, pruned_model_layer = prune(
                model, P_FILTER, P_LAYER,
                CRITERION_FILTER, CRITERION_LAYER,
                X_train, y_train
            )
            model, chosen_structure = winner_pruned_model(
                pruned_model_filter, pruned_model_layer,
                best_pruned_criteria='flops'
            )

            # Save the pruned model (the new "current" model)
            model.save("pruned_model.h5")
            print("Pruned model saved as pruned_model.h5.")

            # Compute new accuracies after pruning
            new_acc_test = accuracy_score(
                np.argmax(y_test, axis=1),
                np.argmax(model.predict(X_test, verbose=0), axis=1)
            )
            y_pred_val_after_pruning = model.predict(X_val, verbose=0)
            new_acc_val = accuracy_score(
                np.argmax(y_val, axis=1),
                np.argmax(y_pred_val_after_pruning, axis=1)
            )

            # Show stats
            current_flops = statistics(model)

            best_val_acc = new_acc_val
            model.save("best_model.h5")
            print(f"Pruned model saved as best_model.h5.")

            # Re-initialize rotation tracker with the pruned model
            rotation_tracker = LayerRotationTracker(
                model=model,
                layer_names=None,
                start_epoch=ROTATION_start_epoch,
                angle_threshold=ROTATION_ANGLE_THRESHOLD,
                stable_epochs_needed=ROTATION_STABLE_EPOCHS
            )

            # Update log record with pruning info
            train_record["pruning_info"] = f"Pruned with structure={chosen_structure}, new_val_acc={new_acc_val:.4f}"

        # 8) Write record to JSONL
        log_file.write(json.dumps(train_record) + "\n")
        log_file.flush()

        final_model_for_each_epoch.append(model)

    # Close the JSONL file after training ends
    log_file.close()