import os
import random
import sys

import keras
import numpy as np
import tensorflow as tf
from keras import layers
from sklearn.metrics import accuracy_score
from tensorflow.python.data import Dataset

import rebuild_heads as rh
import rebuild_layers as rl
import template_architectures
from pruning_criteria import criteria_head as ch
from pruning_criteria import criteria_layer as cl
from tqdm import tqdm

# Configurações para remover warnings
import logging
import os


sys.path.insert(0, '../utils')
import custom_functions as func
import custom_callbacks

# Global variables to replace command line arguments
ARCHITECTURE_NAME = ''
CRITERION_HEAD = 'random'
CRITERION_LAYER = 'random'
P_HEAD = 1.0
P_LAYER = 1

# Pruning schedule variables
PRUNING_START_EPOCH = 2  # Começa a podar após 50 épocas
PRUNING_INTERVAL = 2     # Poda a cada 10 épocas

# Variáveis globais adicionais
SEED_VALUE = 12227
DATA_FILE = 'FaciesClassificationYananGasField'
N_EPOCHS = 200
LR = 0.001
BATCH_SIZE = 1024
PROJECTION_DIM = 64
NUM_HEADS = [256, 128, 64, 16, 32, 8]
SCHEDULE = [(100, LR / 10), (150, LR / 100)]

def load_transformer_data(file='synthetic'):
    """
    loads premade data for transformer model
    Args:
        file: name of the file to load data from; 'synthetic' generates random data for tests
    Returns:
        x_train, x_test, y_train, y_test, n_classes
    """
    if file == 'synthetic':
        # Synthetic data"
        samples, features, n_classes = 1000, 200, 3
        x_train, x_test = np.random.rand(samples, features), np.random.rand(int(samples / 10),
                                                                            features)  # samples x features
        y_train = np.random.randint(0, n_classes, len(x_train))
        y_test = np.random.randint(0, n_classes, len(x_test))
        n_classes = len(np.unique(y_train, axis=0))
    else:
        # Real data -- DecaLearn
        if '.npz' not in file:
            file += '.npz'

        tmp = np.load(file)
        x_train, x_test, y_train, y_test = tmp['X_train'], tmp['X_test'], tmp['y_train'], tmp['y_test']

        x_train = np.expand_dims(x_train, axis=1)
        x_test = np.expand_dims(x_test, axis=1)

        n_classes = len(np.unique(y_train, axis=0))
        y_train = np.eye(n_classes)[y_train]
        y_test = np.eye(n_classes)[y_test]

    return x_train, x_test, y_train, y_test, n_classes


def flops(model, verbose=False):
    """
    Calculate FLOPS used by the model
    Args:
        model: model to calculate perfomance of
        verbose: if True, returns extra information (I.E.: flops per type of operation among others)

    Returns:
        numbers of flops used by the model
    """
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function([tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        if not verbose:
            opts['output'] = 'none'
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
    return flops.total_float_ops


def statistics(model):
    """
    prints statistics of the model: number of heads, parameters, FLOPS and memory
    Args:
        model: model from which the statistics are calculated
    """
    n_params = model.count_params()
    n_heads = func.count_filters(model)
    memory = func.memory_usage(1, model)
    tmp = [layer._num_heads for layer in model.layers if isinstance(layer, layers.MultiHeadAttention)]

    print('#Heads {} Params [{}]  FLOPS [{}] Memory [{:.6f}]'.format(tmp, n_params, flops(model), memory), flush=True)


def fine_tuning(model, x_train, y_train, x_test, y_test, current_epoch):
    """
    Realiza o fine tuning do modelo para uma época
    
    Args:
        model: modelo a ser treinado
        x_train: dados de treino
        y_train: labels de treino
        x_test: dados de teste
        y_test: labels de teste
        current_epoch: época atual para o scheduler
        
    Returns:
        model: modelo treinado
        accuracy: acurácia da época atual
    """
    # Configura o learning rate scheduler
    lr_scheduler = custom_callbacks.LearningRateScheduler(init_lr=LR, schedule=SCHEDULE)
    callbacks = [lr_scheduler]

    # Configura o otimizador SGD com momentum
    sgd = keras.optimizers.SGD(
        learning_rate=LR, 
        decay=1e-6, 
        momentum=0.9, 
        nesterov=True
    )
    
    # Compila o modelo
    model.compile(
        loss='categorical_crossentropy',
        optimizer=sgd,
        metrics=['accuracy']
    )

    # Cria um dataset TensorFlow com shuffle e batch
    dataset = Dataset.from_tensor_slices((x_train, y_train))\
        .shuffle(4 * BATCH_SIZE)\
        .batch(BATCH_SIZE)

    # Treina por uma época
    model.fit(
        dataset,
        batch_size=BATCH_SIZE,
        verbose=2,
        callbacks=callbacks,
        epochs=current_epoch + 1,
        initial_epoch=current_epoch
    )

    # Calcula acurácia
    y_pred = model.predict(x_test, verbose=0)
    accuracy = accuracy_score(
        np.argmax(y_test, axis=1),
        np.argmax(y_pred, axis=1)
    )
    
    return model, accuracy


def perform_pruning(model, x_train, y_train, x_test, y_test):
    """
    Realiza tanto a poda de heads quanto de layers e aplica a que obtiver melhor resultado
        
    Args:
        model: modelo original
        x_train, y_train: dados de treino
        x_test, y_test: dados de teste
        
    Returns:
        best_model: modelo após a melhor poda
        best_accuracy: acurácia do melhor modelo
    """
    # Calcula métricas do modelo original
    original_flops = flops(model)
    original_heads = [layer._num_heads for layer in model.layers if isinstance(layer, layers.MultiHeadAttention)]
    original_params = model.count_params()
    
    y_pred = model.predict(x_test, verbose=0)
    original_accuracy = accuracy_score(
        np.argmax(y_test, axis=1),
        np.argmax(y_pred, axis=1)
    )
    
    # Tenta poda de heads
    head_method = ch.criteria(CRITERION_HEAD)
    head_scores = head_method.scores(model, x_train, y_train, rh.heads_to_prune(model))
    head_pruned_model = rh.rebuild_network(model, head_scores, P_HEAD)
    
    # Métricas após poda de heads
    head_flops = flops(head_pruned_model)
    head_heads = [layer._num_heads for layer in head_pruned_model.layers if isinstance(layer, layers.MultiHeadAttention)]
    head_params = head_pruned_model.count_params()
    
    y_pred = head_pruned_model.predict(x_test, verbose=0)
    head_accuracy = accuracy_score(
        np.argmax(y_test, axis=1),
        np.argmax(y_pred, axis=1)
    )
    
    # Tenta poda de layers
    layer_method = cl.criteria(CRITERION_LAYER)
    layer_scores = layer_method.scores(model, x_train, y_train, rl.layers_to_prune(model))
    layer_pruned_model = rl.rebuild_network(model, layer_scores, P_LAYER)
    
    # Métricas após poda de layers
    layer_flops = flops(layer_pruned_model)
    layer_heads = [layer._num_heads for layer in layer_pruned_model.layers if isinstance(layer, layers.MultiHeadAttention)]
    layer_params = layer_pruned_model.count_params()
    
    y_pred = layer_pruned_model.predict(x_test, verbose=0)
    layer_accuracy = accuracy_score(
        np.argmax(y_test, axis=1),
        np.argmax(y_pred, axis=1)
    )
    
    # Imprime resultados detalhados
    print("\nResultados da Tentativa de Poda:")
    print(f"Modelo Original:")
    print(f"- Accuracy: {original_accuracy:.4f}")
    print(f"- FLOPS: {original_flops:,}")
    print(f"- Heads: {original_heads}")
    print(f"- Params: {original_params:,}")
    
    print(f"\nPoda de Heads:")
    print(f"- Accuracy: {head_accuracy:.4f}")
    print(f"- FLOPS: {head_flops:,} ({((original_flops - head_flops)/original_flops)*100:.2f}% redução)")
    print(f"- Heads: {head_heads}")
    print(f"- Params: {head_params:,} ({((original_params - head_params)/original_params)*100:.2f}% redução)")
    
    print(f"\nPoda de Layers:")
    print(f"- Accuracy: {layer_accuracy:.4f}")
    print(f"- FLOPS: {layer_flops:,} ({((original_flops - layer_flops)/original_flops)*100:.2f}% redução)")
    print(f"- Heads: {layer_heads}")
    print(f"- Params: {layer_params:,} ({((original_params - layer_params)/original_params)*100:.2f}% redução)")
    
    # Escolhe a melhor poda
    if head_accuracy > layer_accuracy:
        print("\nEscolhida poda de heads - Melhor accuracy com:")
        print(f"- Redução de FLOPS: {((original_flops - head_flops)/original_flops)*100:.2f}%")
        print(f"- Redução de Params: {((original_params - head_params)/original_params)*100:.2f}%")
        return head_pruned_model, head_accuracy
    else:
        print("\nEscolhida poda de layers - Melhor accuracy com:")
        print(f"- Redução de FLOPS: {((original_flops - layer_flops)/original_flops)*100:.2f}%")
        print(f"- Redução de Params: {((original_params - layer_params)/original_params)*100:.2f}%")
        return layer_pruned_model, layer_accuracy


if __name__ == '__main__':
    np.random.seed(SEED_VALUE)
    random.seed(SEED_VALUE)

    # Configuração do ambiente
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32'] = '1'
    os.environ['TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32'] = '1'
    physical_devices = tf.config.list_physical_devices('GPU')

    print('Architecture [{}] p_head[{}] p_layer[{}]'.format(
        ARCHITECTURE_NAME, P_HEAD, P_LAYER), flush=True)

    # Carrega e prepara dados tabulares
    x_train, x_test, y_train, y_test, n_classes = load_transformer_data(
        file=DATA_FILE)

    # Configuração do modelo
    input_shape = (x_train.shape[1:])
    projection_dim = PROJECTION_DIM
    num_heads = NUM_HEADS

    # Cria e compila o modelo inicial
    model = template_architectures.TransformerTabular(
        input_shape, projection_dim, num_heads, n_classes)

    # Treinamento inicial
    n_epochs = N_EPOCHS
    print("Iniciando treinamento...")
    
    # Guarda estado inicial do modelo para comparação
    initial_flops = flops(model)
    initial_params = model.count_params()
    initial_heads = [layer._num_heads for layer in model.layers if isinstance(layer, layers.MultiHeadAttention)]
    
    for epoch in tqdm(range(n_epochs)):
        model, accuracy = fine_tuning(
            model, x_train, y_train, x_test, y_test, epoch)
        
        # Verifica se é momento de podar
        if epoch >= PRUNING_START_EPOCH and (epoch - PRUNING_START_EPOCH) % PRUNING_INTERVAL == 0:
            print(f"\nRealizando tentativa de poda na época {epoch}")
            
            # Guarda métricas antes da poda
            pre_pruning_flops = flops(model)
            pre_pruning_params = model.count_params()
            pre_pruning_heads = [layer._num_heads for layer in model.layers if isinstance(layer, layers.MultiHeadAttention)]
            
            # Realiza a poda
            model, accuracy = perform_pruning(model, x_train, y_train, x_test, y_test)
            
            # Confirma mudanças após a poda
            current_flops = flops(model)
            current_params = model.count_params()
            current_heads = [layer._num_heads for layer in model.layers if isinstance(layer, layers.MultiHeadAttention)]
            
            print("\nConfirmação das mudanças após poda:")
            print(f"FLOPS: {pre_pruning_flops:,} -> {current_flops:,} (Redução de {((pre_pruning_flops - current_flops)/pre_pruning_flops)*100:.2f}%)")
            print(f"Params: {pre_pruning_params:,} -> {current_params:,} (Redução de {((pre_pruning_params - current_params)/pre_pruning_params)*100:.2f}%)")
            print(f"Heads: {pre_pruning_heads} -> {current_heads}")
            print(f"Redução total desde o início:")
            print(f"- FLOPS: Redução de {((initial_flops - current_flops)/initial_flops)*100:.2f}%")
            print(f"- Params: Redução de {((initial_params - current_params)/initial_params)*100:.2f}%")
            print("-"*50)
            
        # Imprime resultados a cada 5 épocas
        if epoch % 5 == 0:
            print(f'Época [{epoch}] Accuracy [{accuracy:.4f}]')
            statistics(model)

    # Avaliação final do modelo
    y_pred = model.predict(x_test)
    acc = accuracy_score(np.argmax(y_pred, axis=1), np.argmax(y_test, axis=1))
    print('Accuracy Final [{}]'.format(acc))
    statistics(model)
