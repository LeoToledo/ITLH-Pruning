import os
import random
import sys

import keras
import numpy as np
import tensorflow as tf
from keras import layers
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

import pruning_criteria.criteria_head as ch
import pruning_criteria.criteria_layer as cl
import rebuild_heads as rh
import rebuild_layers as rl
from checkpoints_config import setup_checkpoint_directories, save_checkpoint
from template_architectures import TransformerTabular
from load_tabular_data import load_covertype_data

sys.path.insert(0, '../utils')
import custom_functions as func

# Global Parameters
NUM_EPOCHS = 100
BATCH_SIZE = 512  
LEARNING_RATE = 0.001
PROJECTION_DIM = 128  
HEADS_PER_LAYER = [16, 16, 16, 16]
PRUNING_FREQUENCY = 3
PRUNING_START = 3
CRITERION_HEAD = 'CKA'
CRITERION_LAYER = 'CKA'
P_HEAD = 10
P_LAYER = 1


def flops(model, verbose=False):
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(
        [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs]
    )
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)

    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        if not verbose:
            opts['output'] = 'none'
        flops = tf.compat.v1.profiler.profile(
            graph=graph, run_meta=run_meta, cmd='op', options=opts
        )
    return flops.total_float_ops

def count_num_heads(model):
    """
    Count the number of attention heads in a model by recursively traversing all layers
    """
    def count_heads_recursive(layer):
        try:
            num_heads = 0
            if isinstance(layer, tf.keras.layers.MultiHeadAttention):
                if hasattr(layer, '_num_heads'):
                    num_heads += layer._num_heads
                elif hasattr(layer, 'num_heads'):
                    num_heads += layer.num_heads
                else:
                    query_weights = layer.get_weights()[0]
                    key_dim = layer._key_dim
                    total_dims = query_weights.shape[-1]
                    estimated_heads = total_dims // (key_dim * 4)
                    num_heads += estimated_heads

            if hasattr(layer, 'layers'):
                for sublayer in layer.layers:
                    num_heads += count_heads_recursive(sublayer)

            return num_heads
        except Exception as e:
            return 0

    try:
        total_heads = count_heads_recursive(model)
        return total_heads
    except Exception as e:
        return 0

def get_attention_layers(model):
    """
    Get all MultiHeadAttention layers in the model
    """
    attention_layers = []

    def get_layers_recursive(layer):
        if isinstance(layer, tf.keras.layers.MultiHeadAttention):
            attention_layers.append(layer)

        if hasattr(layer, 'layers'):
            for sublayer in layer.layers:
                get_layers_recursive(sublayer)

    get_layers_recursive(model)
    return attention_layers


def itlh_pruner(model, x_train, y_train, x_test, y_test, criterion_head='CKA', criterion_layer='CKA', p_head=2, p_layer=1):
   """
   Iterative Two-Level Hierarchical Pruner (ITLH).
   """
   try:
       print("\n=== INICIALIZAÇÃO DA PODA ===")
       total_params = model.count_params()
       original_flops = flops(model)
       print(f"Parâmetros totais do modelo: {total_params:,}")
       print(f"FLOPS originais: {original_flops:,}")

       attention_params = 0
       other_params = 0
       for layer in model.layers:
           layer_params = layer.count_params()
           if isinstance(layer, tf.keras.layers.MultiHeadAttention):
               attention_params += layer_params
           else:
               other_params += layer_params

       print(f"\nTotal de parâmetros em camadas de atenção: {attention_params:,} ({attention_params/total_params*100:.2f}%)")
       print(f"Total de parâmetros em outras camadas: {other_params:,} ({other_params/total_params*100:.2f}%)")

       num_heads_before = count_num_heads(model)
       attention_layers = get_attention_layers(model)

       print(f"\nNúmero de cabeças antes da poda: {num_heads_before}")
       print(f"Número de camadas de atenção encontradas: {len(attention_layers)}")

       # Evaluate head pruning
       head_method = ch.criteria(criterion_head)
       head_scores = head_method.scores(model, x_train, y_train, rh.heads_to_prune(model))

       try:
           layer_indices = [idx for idx, layer in enumerate(model.layers) 
                          if isinstance(layer, tf.keras.layers.MultiHeadAttention)]

           formatted_scores = []
           for score_tuple in head_scores[0]:
               layer_idx = layer_indices[len(formatted_scores)]
               formatted_scores.append((layer_idx, score_tuple[1]))

           heads_to_prune = rh.get_heads_to_prune(model, formatted_scores, p_head)
           print(f"\n=== CABEÇAS A SEREM PODADAS ===")
           print(f"Índices das cabeças a serem podadas: {heads_to_prune}")

           model_head_pruned = rh.rebuild_network(model, heads_to_prune)

           num_heads_after = count_num_heads(model_head_pruned)
           print(f"\nNúmero de cabeças após a poda: {num_heads_after}")

           # FLOPS analysis
           head_flops = flops(model_head_pruned)
           head_reduction = (original_flops - head_flops) / original_flops * 100

           print("\n=== ANÁLISE DE REDUÇÃO ===")
           print(f"FLOPS originais: {original_flops:,}")
           print(f"FLOPS após poda de cabeças: {head_flops:,}")
           print(f"Redução total de FLOPS: {original_flops - head_flops:,}")
           print(f"Porcentagem de redução: {head_reduction:.2f}%")
           print(f"Redução média por cabeça: {(original_flops - head_flops)/(num_heads_before - num_heads_after):,.2f} FLOPS/cabeça")

       except Exception as e:
           raise RuntimeError(f"Head pruning failed: {str(e)}")

       # Layer pruning
       layer_method = cl.criteria(criterion_layer)
       layer_scores = layer_method.scores(model, x_train, y_train, rl.layers_to_prune(model))
       try:
           blocks_tmp, mask, allowed_layers = rl.get_layers_to_prune(model, layer_scores[0], p_layer)
           model_layer_pruned = rl.rebuild_network(model, blocks_tmp, mask, allowed_layers)
           layer_flops = flops(model_layer_pruned)
           layer_reduction = (original_flops - layer_flops) / original_flops * 100
       except Exception as e:
           raise RuntimeError(f"Layer pruning failed: {str(e)}")

       # Compare results and print detailed information
       print("\n=== RESULTADOS FINAIS DA PODA ===")
       print(f"FLOPS originais: {original_flops:,}")
       print(f"Redução por poda de cabeças: {head_reduction:.2f}%")
       print(f"Redução por poda de camadas: {layer_reduction:.2f}%")

       if head_reduction >= layer_reduction:
           print("Poda de cabeças oferece melhor redução. Selecionando poda de cabeças.")
           return model_head_pruned, 'head', head_reduction
       else:
           print("Poda de camadas oferece melhor redução. Selecionando poda de camadas.")
           return model_layer_pruned, 'layer', layer_reduction

   except Exception as e:
       raise RuntimeError(f"ITLH Pruner failed: {str(e)}")

if __name__ == '__main__':
    try:
        # Configuration of seeds
        np.random.seed(12227)
        random.seed(12227)

        # GPU configuration
        os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
        os.environ['TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32'] = '1'
        os.environ['TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32'] = '1'
        physical_devices = tf.config.list_physical_devices('GPU')

        # Setup checkpoint directories
        run_dir, checkpoint_dirs = setup_checkpoint_directories()

        # Load data
        x_train, y_train, x_test, y_test = load_covertype_data(debug=True, sample_size=1_000)

        input_shape = x_train.shape[1:]  # Now includes the sequence dimension
        n_classes = y_train.shape[1]  # Binary classification

        # Create tabular transformer model
        model = TransformerTabular(
            input_shape=input_shape,
            projection_dim=PROJECTION_DIM,
            num_heads=HEADS_PER_LAYER,
            n_classes=n_classes
        )

        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # History dictionary
        history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'test_acc': [],
            'flops_reduction': [],
            'pruning_type': []
        }

        # Training loop
        for epoch in range(NUM_EPOCHS):
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
            # Train for one epoch
            train_results = model.fit(
                x_train, y_train,
                batch_size=BATCH_SIZE,
                epochs=1,
                verbose=1
            )

            # Evaluate on test set
            y_pred = model.predict(x_test, verbose=0)
            test_acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

            # Update history
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(train_results.history['loss'][0])
            history['train_acc'].append(train_results.history['accuracy'][0])
            history['test_acc'].append(test_acc)

            # Print epoch results
            print(f"Train Loss: {history['train_loss'][-1]:.4f}, Train Acc: {history['train_acc'][-1]:.4f}, Test Acc: {test_acc:.4f}")

            # Prune the model when needed
            if (epoch + 1) >= PRUNING_START and (epoch + 1) % PRUNING_FREQUENCY == 0:
                print("\nIniciando processo de poda...")
                try:
                    # Save checkpoint before pruning
                    training_params = {
                        'NUM_EPOCHS': NUM_EPOCHS,
                        'BATCH_SIZE': BATCH_SIZE,
                        'LEARNING_RATE': LEARNING_RATE,
                        'PROJECTION_DIM': PROJECTION_DIM,
                        'HEADS_PER_LAYER': HEADS_PER_LAYER,
                        'PRUNING_FREQUENCY': PRUNING_FREQUENCY,
                        'PRUNING_START': PRUNING_START,
                        'CRITERION_HEAD': CRITERION_HEAD,
                        'CRITERION_LAYER': CRITERION_LAYER,
                        'P_HEAD': P_HEAD,
                        'P_LAYER': P_LAYER
                    }
                    save_checkpoint(model, history, checkpoint_dirs, 'pruning', epoch + 1, 'before', training_params)

                    # Perform pruning
                    model_pruned, pruning_type, flops_reduction = itlh_pruner(
                        model=model,
                        x_train=x_train,
                        y_train=y_train,
                        x_test=x_test,
                        y_test=y_test,
                        criterion_head=CRITERION_HEAD,
                        criterion_layer=CRITERION_LAYER,
                        p_head=P_HEAD,
                        p_layer=P_LAYER
                    )

                    # Update the model
                    model = model_pruned
                    model.compile(
                        optimizer=optimizer,
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )

                    # Update history
                    history['flops_reduction'].append(flops_reduction)
                    history['pruning_type'].append(pruning_type)

                    # Print pruning results
                    print(f"Tipo de poda: {pruning_type}, Redução de FLOPS: {flops_reduction:.2f}%")

                    # Save checkpoint after pruning
                    save_checkpoint(model, history, checkpoint_dirs, 'pruning', epoch + 1, 'after', training_params)

                except Exception as e:
                    print(f"Error during pruning: {str(e)}")
                    # Save current state safely
                    save_checkpoint(model, history, checkpoint_dirs, 'error', None, None, training_params)
                    sys.exit(1)
            else:
                history['flops_reduction'].append(0.0)
                history['pruning_type'].append(None)

            # Save regular checkpoints
            if (epoch + 1) % 10 == 0:
                save_checkpoint(model, history, checkpoint_dirs, 'regular', epoch + 1, None, training_params)

        # Save final results
        save_checkpoint(model, history, checkpoint_dirs, 'final', None, None, training_params)

    except Exception as e:
        print(f"Critical error in main execution: {str(e)}")
        # Try to save current state even in case of error
        try:
            training_params = {
                'NUM_EPOCHS': NUM_EPOCHS,
                'BATCH_SIZE': BATCH_SIZE,
                'LEARNING_RATE': LEARNING_RATE,
                'PROJECTION_DIM': PROJECTION_DIM,
                'HEADS_PER_LAYER': HEADS_PER_LAYER,
                'PRUNING_FREQUENCY': PRUNING_FREQUENCY,
                'PRUNING_START': PRUNING_START,
                'CRITERION_HEAD': CRITERION_HEAD,
                'CRITERION_LAYER': CRITERION_LAYER,
                'P_HEAD': P_HEAD,
                'P_LAYER': P_LAYER
            }
            save_checkpoint(model, history, checkpoint_dirs, 'error', None, None, training_params)
        except Exception as e:
            print(f"Error saving final checkpoint: {str(e)}")

        sys.exit(1)
