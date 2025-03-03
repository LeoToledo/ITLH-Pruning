# custom_functions.py

import random
import custom_callbacks
import numpy as np
from sklearn.metrics import accuracy_score
import keras
from tensorflow.python.data import Dataset


def fine_tuning_cnn(model, x_train, y_train, x_test, y_test):

    batch_size = 1024
    lr = 0.001
    schedule = [(100, lr / 10), (150, lr / 100)]
    lr_scheduler = custom_callbacks.LearningRateScheduler(init_lr=lr, schedule=schedule)
    callbacks = [lr_scheduler]

    sgd = keras.optimizers.SGD(learning_rate=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    for ep in range(0, 200):
        y_tmp = np.concatenate((y_train, y_train, y_train))
        x_tmp = np.concatenate(
            (data_augmentation(x_train),
                data_augmentation(x_train),
                data_augmentation(x_train)))

        x_tmp = Dataset.from_tensor_slices((x_tmp, y_tmp)).shuffle(4 * batch_size).batch(batch_size)

        model.fit(x_tmp, batch_size=batch_size, verbose=2,
                    callbacks=callbacks,
                    epochs=ep, initial_epoch=ep - 1)

        if ep % 5 == 0:
            acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(model.predict(x_test, verbose=0), axis=1))
            print('Accuracy [{:.4f}]'.format(acc), flush=True)

    return model

def load_transformer_model(architecture_file='', weights_file=''):
    """
    loads a premade transformer model from a set of files
    Args:
        architecture_file: name of the file containing architecture of the model
        weights_file: name of the file containing weights of the model
    Returns:
        loaded transformer model
    """
    import keras
    from custom_classes import Patches, PatchEncoder
    from keras.utils import CustomObjectScope

    if '.json' not in architecture_file:
        architecture_file = architecture_file + '.json'

    with open(architecture_file, 'r') as f:
        with CustomObjectScope({'PatchEncoder': PatchEncoder},
                               {'Patches': Patches}):
            model = keras.models.model_from_json(f.read())

    if weights_file != '':
        if '.h5' not in weights_file:
            weights_file = weights_file + '.h5'
        model.load_weights(weights_file)
        print('Load architecture [{}]. Load weights [{}]'.format(architecture_file, weights_file), flush=True)
    else:
        print('Load architecture [{}]'.format(architecture_file), flush=True)

    return model


def load_model(architecture_file='', weights_file=''):
    """
    loads a premade neural network model from a set of files
    Args:
        architecture_file: name of the file containing architecture of the model
        weights_file: name of the file containing weights of the model
    Returns:
        loaded neural network model
    """
    import keras
    from keras.utils.generic_utils import CustomObjectScope
    from keras import backend as K
    from keras import layers

    def _hard_swish(x):
        return x * K.relu(x + 3.0, max_value=6.0) / 6.0

    def _relu6(x):
        return K.relu(x, max_value=6)

    if '.json' not in architecture_file:
        architecture_file = architecture_file + '.json'

    with open(architecture_file, 'r') as f:
        # Not compatible with keras 2.4.x and TF 2.0
        # with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,
        #                         'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D,
        #                        '_hard_swish': _hard_swish}):
        with CustomObjectScope({'relu6': _relu6,
                                'DepthwiseConv2D': layers.DepthwiseConv2D,
                                '_hard_swish': _hard_swish}):
            model = keras.models.model_from_json(f.read())

    if weights_file != '':
        if '.h5' not in weights_file:
            weights_file = weights_file + '.h5'
        model.load_weights(weights_file)
        print('Load architecture [{}]. Load weights [{}]'.format(architecture_file, weights_file), flush=True)
    else:
        print('Load architecture [{}]'.format(architecture_file), flush=True)

    return model


def save_model(file_name='', model=None):
    """
    saves a model into architecture and weights file
    Args:
        file_name: name of the file the model is saved into
        model: model to be saved
    """
    print('Saving architecture and weights in {}'.format(file_name))

    model.save_weights(file_name + '.h5')
    with open(file_name + '.json', 'w') as f:
        f.write(model.to_json())


def generate_data_augmentation(x_train):
    print('Using real-time data augmentation.')
    from keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    datagen.fit(x_train)
    return datagen


def cifar_resnet_data(validation_split=0.2, random_state=42):
    """
    Carrega e prepara os dados do CIFAR-10 para treinamento e validação.
    
    Args:
        validation_split (float): Proporção dos dados para validação (entre 0 e 1)
        random_state (int): Semente para reproducibilidade
        
    Returns:
        tuple: (x_train, y_train, x_val, y_val)
    """
    import tensorflow as tf
    import numpy as np
    
    # Configurar seed para reproducibilidade
    np.random.seed(random_state)
    tf.random.set_seed(random_state)
    
    # Carregar dados CIFAR-10
    (x_all, y_all), _ = tf.keras.datasets.cifar10.load_data()
    
    # Embaralhar os índices
    total_samples = len(x_all)
    indices = np.random.permutation(total_samples)
    
    # Calcular o ponto de divisão para validação
    val_size = int(total_samples * validation_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    # Separar dados de treino e validação
    x_train = x_all[train_indices]
    y_train = y_all[train_indices]
    x_val = x_all[val_indices]
    y_val = y_all[val_indices]
    
    # Converter para float32 e normalizar para [0,1]
    x_train = x_train.astype('float32') / 255.0
    x_val = x_val.astype('float32') / 255.0
    
    # Calcular média e desvio padrão apenas do conjunto de treino
    train_mean = np.mean(x_train, axis=0)
    train_std = np.std(x_train, axis=0) + 1e-7  # Evitar divisão por zero
    
    # Normalizar (z-score normalization)
    x_train = (x_train - train_mean) / train_std
    x_val = (x_val - train_mean) / train_std
    
    # Converter labels para one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_val = tf.keras.utils.to_categorical(y_val, 10)
    
    return x_train, y_train, x_val, y_val

def count_filters(model):
    import keras
    # from keras.applications.mobilenet import DepthwiseConv2D
    from keras.layers import DepthwiseConv2D
    n_filters = 0
    # Model contains only Conv layers
    for layer_idx in range(1, len(model.layers)):

        layer = model.get_layer(index=layer_idx)

        if isinstance(layer, keras.layers.Conv2D) and not isinstance(layer, DepthwiseConv2D):
            config = layer.get_config()
            n_filters += config['filters']

        if isinstance(layer, DepthwiseConv2D):
            n_filters += layer.output_shape[-1]

    # Todo: Model contains Conv and Fully Connected layers
    # for layer_idx in range(1, len(model.get_layer(index=1))):
    #     layer = model.get_layer(index=1).get_layer(index=layer_idx)
    #     if isinstance(layer, keras.layers.Conv2D) == True:
    #         config = layer.get_config()
    #     n_filters += config['filters']
    return n_filters


def count_filters_layer(model):
    import keras
    # from keras.applications.mobilenet import DepthwiseConv2D
    from keras.layers import DepthwiseConv2D
    n_filters = ''
    # Model contains only Conv layers
    for layer_idx in range(1, len(model.layers)):

        layer = model.get_layer(index=layer_idx)
        if isinstance(layer, keras.layers.Conv2D) and not isinstance(layer, DepthwiseConv2D):
            config = layer.get_config()
            n_filters += str(config['filters']) + ' '

        if isinstance(layer, DepthwiseConv2D):
            n_filters += str(layer.output_shape[-1])

    return n_filters


def compute_flops(model):
    # useful link https://www.programmersought.com/article/27982165768/
    import keras
    # from keras.applications.mobilenet import DepthwiseConv2D
    from keras.layers import DepthwiseConv2D
    total_flops = 0
    flops_per_layer = []

    for layer_idx in range(1, len(model.layers)):
        layer = model.get_layer(index=layer_idx)
        if isinstance(layer, DepthwiseConv2D) is True:
            _, output_map_H, output_map_W, current_layer_depth = layer.output_shape

            _, _, _, previous_layer_depth = layer.input_shape
            kernel_H, kernel_W = layer.kernel_size

            # Computed according to https://arxiv.org/pdf/1704.04861.pdf Eq.(5)
            flops = (kernel_H * kernel_W * previous_layer_depth * output_map_H * output_map_W) + (
                    previous_layer_depth * current_layer_depth * output_map_W * output_map_H)
            total_flops += flops
            flops_per_layer.append(flops)

        elif isinstance(layer, keras.layers.Conv2D) is True:
            _, output_map_H, output_map_W, current_layer_depth = layer.output_shape

            _, _, _, previous_layer_depth = layer.input_shape
            kernel_H, kernel_W = layer.kernel_size

            flops = output_map_H * output_map_W * previous_layer_depth * current_layer_depth * kernel_H * kernel_W
            total_flops += flops
            flops_per_layer.append(flops)

        if isinstance(layer, keras.layers.Dense) is True:
            _, current_layer_depth = layer.output_shape

            _, previous_layer_depth = layer.input_shape

            flops = current_layer_depth * previous_layer_depth
            total_flops += flops
            flops_per_layer.append(flops)

    return total_flops, flops_per_layer


def top_k_accuracy(y_true, y_pred, k):
    top_n = np.argsort(y_pred, axis=1)[:, -k:]
    idx_class = np.argmax(y_true, axis=1)
    hit = 0
    for i in range(idx_class.shape[0]):
        if idx_class[i] in top_n[i, :]:
            hit = hit + 1
    return float(hit) / idx_class.shape[0]


def center_crop(image, crop_size=224):
    h, w, _ = image.shape

    top = (h - crop_size) // 2
    left = (w - crop_size) // 2

    bottom = top + crop_size
    right = left + crop_size

    return image[top:bottom, left:right, :]


def random_crop(img=None, random_crop_size=(64, 64)):
    # Code taken from https://jkjung-avt.github.io/keras-image-cropping/
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y + dy), x:(x + dx), :]


def data_augmentation(X, padding=4):
    X_out = np.zeros(X.shape, dtype=X.dtype)
    n_samples, x, y, _ = X.shape

    padded_sample = np.zeros((x + padding * 2, y + padding * 2, 3), dtype=X.dtype)

    for i in range(0, n_samples):
        p = random.random()
        padded_sample[padding:x + padding, padding:y + padding, :] = X[i][:, :, :]
        if p >= 0.5:  # random crop on the original image
            X_out[i] = random_crop(padded_sample, (x, y))
        else:  # random crop on the flipped image
            X_out[i] = random_crop(np.flip(padded_sample, axis=1), (x, y))

        # import matplotlib.pyplot as plt
        # plt.imshow(X_out[i])

    return X_out


def memory_usage(batch_size, model):
    import tensorflow as tf
    from keras import backend as K
    # Taken from #https://stackoverflow.com/questions/43137288/how-to-determine-needed-memory-of-keras-model

    if tf.__version__.split('.')[0] != '2':
        shapes_mem_count = 0
        for layer in model.layers:
            single_layer_mem = 1
            for s in layer.output_shape:
                if s is None:
                    continue
                single_layer_mem *= s
            shapes_mem_count += single_layer_mem

        trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
        non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])
    else:
        shapes_mem_count = 0
        for layer in model.layers:
            single_layer_mem = 1
            for s in layer.output_shape:
                if s is None or not isinstance(s, int):
                    continue
                single_layer_mem *= s
            shapes_mem_count += single_layer_mem

        trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
        non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = total_memory / (1024.0 ** 3)
    return gbytes


def count_depth(model):
    import keras.layers as layers
    depth = 0
    for i in range(0, len(model.layers)):
        layer = model.get_layer(index=i)
        if isinstance(layer, layers.Conv2D):
            depth = depth + 1
    print('Depth: [{}]'.format(depth))
    return depth
