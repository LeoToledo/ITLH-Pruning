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

# Configuração das seeds
random.seed(2)
np.random.seed(2)
tf.random.set_seed(2)

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
CRITERION_FILTER = 'random'
CRITERION_LAYER = 'random'
P_FILTER = 0.08
P_LAYER = 2
MAX_EPOCHS = 600
LEARNING_RATE = 0.01
MOMENTUM = 0.9
BATCH_SIZE = 64
ROTATION_START_EPOCH = 50
ROTATION_ANGLE_THRESHOLD = 100
ROTATION_STABLE_EPOCHS = 5

def setup_fine_tuning_cnn(model, learning_rate):
    """Setup do modelo para fine tuning"""
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

def fine_tuning_epoch(model, x_train, y_train, batch_size, callbacks, epoch,
                      pruning_recovery_epochs_left):
    """Treina o modelo por uma época com data augmentation"""

    # Se ainda estamos no período de recovery pós-poda, força LR original
    if pruning_recovery_epochs_left > 0:
        current_lr = LEARNING_RATE
    else:
        # Caso contrário, segue o schedule normal
        schedule = [
            (0, LEARNING_RATE),
            (100, LEARNING_RATE/10),
            (150, LEARNING_RATE/100),
        ]
        current_lr = schedule[0][1]
        for epoch_threshold, lr in schedule:
            if epoch >= epoch_threshold:
                current_lr = lr

    K.set_value(model.optimizer.learning_rate, current_lr)
    print(f"Current learning rate: {current_lr}")

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
    """Predição em batches para eficiência de memória"""
    dataset = tf.data.Dataset.from_tensor_slices(X)\
        .batch(batch_size)\
        .prefetch(tf.data.AUTOTUNE)
    
    predictions = []
    for batch in dataset:
        batch_pred = model.predict(batch, verbose=0)
        predictions.append(batch_pred)
    
    return np.vstack(predictions)

def statistics(model):
    """Estatísticas do modelo"""
    n_params = model.count_params()
    n_filters = func.count_filters(model)
    flops, _ = func.compute_flops(model)
    blocks = rl.count_blocks(model)
    memory = func.memory_usage(1, model)
    
    print(f'Blocks {blocks} Parameters [{n_params}] Filters [{n_filters}] '
          f'FLOPS [{flops}] Memory [{memory:.6f}]', flush=True)
    return flops

def prune(model, p_filter, p_layer, criterion_filter, criterion_layer, X_train, y_train):
    """Poda filtros e camadas"""
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
    
    return pruned_model_filter, pruned_model_layer

def winner_pruned_model(pruned_model_filter, pruned_model_layer, best_pruned_criteria='flops'):
    """Seleciona o melhor modelo podado"""
    layers_current_flops, _ = func.compute_flops(pruned_model_layer)
    layers_filter_flops, _ = func.compute_flops(pruned_model_filter)
    if best_pruned_criteria == 'flops':
        if layers_current_flops < layers_filter_flops:
            return pruned_model_layer, 'layer'
        return pruned_model_filter, 'filter'
    raise ValueError(f'Unknown best_pruned_criteria [{best_pruned_criteria}]')

if __name__ == '__main__':
    print(f'Architecture [{ARCHITECTURE_NAME}] p_filter[{P_FILTER}] p_layer[{P_LAYER}]', flush=True)

    # Carrega dados e modelo
    X_train, y_train, X_val, y_val = func.cifar_resnet_data(validation_split=0.2, random_state=42)
    
    model = func.load_model(ARCHITECTURE_NAME)
    rf.architecture_name = ARCHITECTURE_NAME
    rl.architecture_name = ARCHITECTURE_NAME

    # Estatísticas iniciais
    init_flops = statistics(model)
    y_pred_val_init = predict_in_batches(model, X_val, BATCH_SIZE)
    initial_acc_val = accuracy_score(
        np.argmax(y_val, axis=1),
        np.argmax(y_pred_val_init, axis=1)
    )
    print(f'Unpruned [{ARCHITECTURE_NAME}] Validation Accuracy [{initial_acc_val}]')

    # Configuração do rotation tracker
    rotation_tracker = LayerRotationTracker(
        model=model,
        layer_names=None,
        start_epoch=ROTATION_START_EPOCH,
        angle_threshold=ROTATION_ANGLE_THRESHOLD,
        stable_epochs_needed=ROTATION_STABLE_EPOCHS
    )

    best_val_acc = initial_acc_val
    log_file = open("train_log.jsonl", "w", encoding="utf-8")

    # Setup inicial do modelo
    model, callbacks = setup_fine_tuning_cnn(model, LEARNING_RATE)

    # Loop principal de treinamento
    PRUNING_RECOVERY_EPOCHS = 0
    for epoch_idx in range(1, MAX_EPOCHS + 1):
        print(f"\n==== EPOCH {epoch_idx} / {MAX_EPOCHS} ====")

        if epoch_idx % 5 == 0:
            gc.collect()
            tf.keras.backend.clear_session()

        # Treinamento: aqui passamos a variável com as épocas de recuperação que faltam
        current_acc_train = fine_tuning_epoch(
            model, X_train, y_train, BATCH_SIZE, callbacks, epoch_idx, 
            PRUNING_RECOVERY_EPOCHS
        )

        # Se ainda estamos em recovery, decrementa
        if PRUNING_RECOVERY_EPOCHS > 0:
            PRUNING_RECOVERY_EPOCHS -= 1

        # Validação
        y_pred_val = predict_in_batches(model, X_val, BATCH_SIZE)
        current_acc_val = accuracy_score(np.argmax(y_val, axis=1), np.argmax(y_pred_val, axis=1))
        print(f"Epoch [{epoch_idx}] Train Accuracy: {current_acc_train:.4f} | Validation Accuracy: {current_acc_val:.4f}")

        # Checa se melhorou e salva modelo
        if current_acc_val > best_val_acc:
            best_val_acc = current_acc_val
            model.save("best_model.h5")
            print(f"Validation accuracy improved to {current_acc_val:.4f}. Model saved.")

        # Verifica se teve estabilidade de rotação para podar
        stable, angle = rotation_tracker.update_and_check_stability(
            model=model,
            current_epoch=epoch_idx,
            total_epochs=MAX_EPOCHS
        )

        # Computa flops e etc.
        epoch_flops, _ = func.compute_flops(model)
        train_record = {
            "epoch": epoch_idx,
            "train_acc": float(current_acc_train),
            "val_acc": float(current_acc_val),
            "flops": int(epoch_flops),
            "angle": float(angle),
            "pruning_info": None
        }

        # Poda quando estável
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

            model.save("pruned_model.h5")

            # [FIX ADDED HERE] -- Re-compile the newly pruned model
            model, callbacks = setup_fine_tuning_cnn(model, LEARNING_RATE)

            # Reavalia após poda
            new_acc_val = accuracy_score(
                np.argmax(y_val, axis=1),
                np.argmax(predict_in_batches(model, X_val, BATCH_SIZE), axis=1)
            )

            current_flops = statistics(model)
            best_val_acc = new_acc_val
            model.save("best_model.h5")

            # Reinicializa rotation tracker
            rotation_tracker = LayerRotationTracker(
                model=model,
                layer_names=None,
                start_epoch=int(ROTATION_START_EPOCH/2),
                angle_threshold=ROTATION_ANGLE_THRESHOLD,
                stable_epochs_needed=ROTATION_STABLE_EPOCHS
            )

            train_record["pruning_info"] = f"Pruned with structure={chosen_structure}, new_val_acc={new_acc_val:.4f}"

            # **Define que por 10 épocas usaremos a LR original**
            PRUNING_RECOVERY_EPOCHS = 10

        # Salva log
        log_file.write(json.dumps(train_record) + "\n")
        log_file.flush()

