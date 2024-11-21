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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all messages, 1 = INFO, 2 = WARNING, 3 = ERROR
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # Só mostra erros

# Desabilita avisos específicos do TensorFlow
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

sys.path.insert(0, '../utils')
import custom_functions as func
import custom_callbacks

# Global variables to replace command line arguments
ARCHITECTURE_NAME = ''
CRITERION_HEAD = 'random'
CRITERION_LAYER = 'random'
P_HEAD = 1.0
P_LAYER = 1

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


def fine_tuning(model, x_train, y_train, x_test, y_test, current_epoch, batch_size=1024, lr=0.001):
    """
    Realiza o fine tuning do modelo para uma época
    
    Args:
        model: modelo a ser treinado
        x_train: dados de treino
        y_train: labels de treino
        x_test: dados de teste
        y_test: labels de teste
        current_epoch: época atual para o scheduler
        batch_size: tamanho do batch
        lr: learning rate inicial
    
    Returns:
        model: modelo treinado
        accuracy: acurácia da época atual
    """
    # Configura o learning rate scheduler
    schedule = [(100, lr / 10), (150, lr / 100)]
    lr_scheduler = custom_callbacks.LearningRateScheduler(init_lr=lr, schedule=schedule)
    callbacks = [lr_scheduler]

    # Configura o otimizador SGD com momentum
    sgd = keras.optimizers.SGD(
        learning_rate=lr, 
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
        .shuffle(4 * batch_size)\
        .batch(batch_size)

    # Treina por uma época
    model.fit(
        dataset,
        batch_size=batch_size,
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


if __name__ == '__main__':
    np.random.seed(12227)
    random.seed(12227)

    # Configuração do ambiente
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32'] = '1'
    os.environ['TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32'] = '1'
    physical_devices = tf.config.list_physical_devices('GPU')

    print('Architecture [{}] p_head[{}] p_layer[{}]'.format(
        ARCHITECTURE_NAME, P_HEAD, P_LAYER), flush=True)

    # Carrega e prepara dados tabulares
    x_train, x_test, y_train, y_test, n_classes = load_transformer_data(
        file='FaciesClassificationYananGasField')

    # Configuração do modelo
    input_shape = (x_train.shape[1:])
    projection_dim = 64
    num_heads = [256, 128, 64, 16, 32, 8]

    # Cria e compila o modelo inicial
    model = template_architectures.TransformerTabular(
        input_shape, projection_dim, num_heads, n_classes)

    # Treinamento inicial - 200 épocas
    n_epochs = 200
    print("Iniciando treinamento inicial...")
    for epoch in tqdm(range(n_epochs)):
        model, accuracy = fine_tuning(
            model, x_train, y_train, x_test, y_test, epoch)
        
        # Imprime resultados a cada 5 épocas
        if epoch % 5 == 0:
            print(f'Época [{epoch}] Accuracy [{accuracy:.4f}]')
            statistics(model)

    # Avaliação final do modelo inicial
    y_pred = model.predict(x_test)
    acc = accuracy_score(np.argmax(y_pred, axis=1), np.argmax(y_test, axis=1))
    print('Accuracy Modelo Inicial [{}]'.format(acc))
    statistics(model)

    # Processo de poda iterativa
    print("\nIniciando processo de poda...")
    for i in range(40):
        prob = random.random()  # Probabilidade para escolher tipo de poda
        
        if prob >= 0.5:
            # Poda de heads
            head_method = ch.criteria(CRITERION_HEAD)
            scores = head_method.scores(model, x_train, y_train, rh.heads_to_prune(model))
            model = rh.rebuild_network(model, scores, P_HEAD)
        else:
            # Poda de layers
            layer_method = cl.criteria(CRITERION_LAYER)
            scores = layer_method.scores(model, x_train, y_train, rl.layers_to_prune(model))
            model = rl.rebuild_network(model, scores, P_LAYER)

        # Retreinamento após poda
        print(f"\nRetreinamento após poda {i+1}")
        for epoch in range(n_epochs):
            model, accuracy = fine_tuning(
                model, x_train, y_train, x_test, y_test, epoch)
            
            if epoch % 5 == 0:
                print(f'Poda [{i+1}] Época [{epoch}] Accuracy [{accuracy:.4f}]')
                statistics(model)

        print(f'Iteração de Poda [{i}] Accuracy Final [{accuracy}]')