import os
import gzip
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import keras
import urllib.request
import io

def load_covertype_data(debug=False, sample_size=100_000):
    """
    Load and preprocess the Forest Cover Type dataset
    Features: 54 (10 continuous, 44 binary)
    Classes: 7 tipos de cobertura florestal
    Total samples: 581,012
    """
    try:
        # Download if not exists
        filename = "covtype.data.gz"
        if not os.path.exists(filename):
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filename)

        # Load data
        print("Loading Cover Type dataset...")
        if debug:
            # Load only a subset in debug mode
            with gzip.open(filename, 'rt') as f:
                data = pd.read_csv(f, nrows=sample_size, header=None)
            print(f"Debug mode: Loaded {sample_size:,} rows")
        else:
            with gzip.open(filename, 'rt') as f:
                data = pd.read_csv(f, header=None)
            print(f"Loaded full dataset: {len(data):,} rows")

        print("Starting preprocessing...")
        
        # Separate features and target
        X = data.iloc[:, :-1].values.astype('float32')  # all columns except last
        y = data.iloc[:, -1].values.astype('int32') - 1  # labels são 1-7, convertendo para 0-6
        
        # Normalize features
        print("Normalizing features...")
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Reshape para o transformer (adiciona dimensão de sequência)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        print("Splitting dataset...")
        # Split mantendo a proporção de classes
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Converter para categorical
        y_train = keras.utils.to_categorical(y_train, num_classes=7)
        y_test = keras.utils.to_categorical(y_test, num_classes=7)

        return X_train, y_train, X_test, y_test

    except Exception as e:
        print(f"Error loading Cover Type data: {str(e)}")
        raise