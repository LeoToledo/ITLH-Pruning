import argparse
import os
import random
import sys

import keras
import numpy as np
import tensorflow as tf
from keras import layers
from sklearn.metrics import accuracy_score
from tensorflow.python.data import Dataset

sys.path.insert(0, '../utils')
import custom_functions as func

def create_transformer_model(input_shape, num_classes, heads_per_layer, projection_dim=64):
    """
    Creates a transformer model with specified number of heads per layer.
    
    Args:
        input_shape: Shape of input data
        num_classes: Number of classes for classification
        heads_per_layer: List specifying number of heads for each layer
        projection_dim: Dimension of the projection space
    """
    inputs = keras.Input(shape=input_shape)
    
    # Create patches
    patches = layers.Conv2D(projection_dim, (3, 3), strides=(2, 2), padding="same")(inputs)
    patches = layers.Reshape((-1, patches.shape[-1]))(patches)
    
    # Position embeddings
    positions = tf.range(start=0, limit=patches.shape[1], delta=1)
    position_embedding = layers.Embedding(input_dim=patches.shape[1], output_dim=projection_dim)(positions)
    x = patches + position_embedding
    
    # Transformer blocks
    for num_heads in heads_per_layer:
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=projection_dim // num_heads
        )(x, x)
        
        # Skip connection 1
        x = layers.Add()([attention_output, x])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # MLP
        ffn = layers.Dense(projection_dim * 2, activation="gelu")(x)
        ffn = layers.Dense(projection_dim)(ffn)
        
        # Skip connection 2
        x = layers.Add()([ffn, x])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Classification head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    return keras.Model(inputs, outputs)

def flops(model, verbose=False):
    """
    Calculate FLOPS used by the model.
    
    Args:
        model: Model to calculate performance of.
        verbose: If True, returns extra information.
    """
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

def statistics(model):
    """
    Prints statistics of the model: number of heads, parameters, FLOPS, and memory.
    """
    n_params = model.count_params()
    memory = func.memory_usage(1, model)
    tmp = [
        layer._num_heads
        for layer in model.layers
        if isinstance(layer, layers.MultiHeadAttention)
    ]

    print(
        '#Heads {} Params [{}] FLOPS [{}] Memory [{:.6f}]'.format(
            tmp, n_params, flops(model), memory
        ),
        flush=True
    )

if __name__ == '__main__':
    np.random.seed(12227)
    random.seed(12227)

    # GPU configuration
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32'] = '1'
    os.environ['TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32'] = '1'
    physical_devices = tf.config.list_physical_devices('GPU')

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', type=str, default='12,12,12,12,12',
                       help='Comma-separated list of heads per layer (e.g., "12,12,12,12,12")')
    parser.add_argument('--projection_dim', type=int, default=64,
                       help='Projection dimension for transformer')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for optimizer')
    
    args = parser.parse_args()
    
    # Convert architecture string to list of integers
    heads_per_layer = [int(x) for x in args.architecture.split(',')]
    print(f"Creating model with architecture: {heads_per_layer}")

    # Load CIFAR data
    x_train, y_train, x_test, y_test = func.cifar_resnet_data(debug=True)
    input_shape = x_train.shape[1:]
    n_classes = y_train.shape[1]

    # Create model
    model = create_transformer_model(
        input_shape=input_shape,
        num_classes=n_classes,
        heads_per_layer=heads_per_layer,
        projection_dim=args.projection_dim
    )

    # Compile model
    optimizer = keras.optimizers.Adam(  # Mudado de AdamW para Adam
        learning_rate=args.learning_rate
    )
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Print initial statistics
    print("\nInitial model statistics:")
    statistics(model)

    # Training loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train for one epoch
        history = model.fit(
            x_train, y_train,
            batch_size=args.batch_size,
            epochs=1,
            verbose=1
        )
        
        # Evaluate on test set
        if (epoch + 1) % 5 == 0:  # Evaluate every 5 epochs
            y_pred = model.predict(x_test, verbose=0)
            acc = accuracy_score(
                np.argmax(y_pred, axis=1),
                np.argmax(y_test, axis=1)
            )
            print(f"\nTest accuracy at epoch {epoch + 1}: {acc:.4f}")
            print("Current model statistics:")
            statistics(model)