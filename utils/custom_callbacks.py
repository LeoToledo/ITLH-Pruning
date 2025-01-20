import numpy as np
import keras
from keras.callbacks import Callback
from keras import backend as K


class LearningRateScheduler(Callback):
    def __init__(self, init_lr=0.01, schedule=None):
        super(Callback, self).__init__()
        self.init_lr = init_lr
        self.schedule = schedule or [(25, 1e-2), (50, 1e-3), (100, 1e-4)]

    def on_epoch_begin(self, epoch, logs=None):
        # Encontra o learning rate apropriado para a época atual
        current_lr = self.init_lr
        for epoch_threshold, lr in self.schedule:
            if epoch >= epoch_threshold:
                current_lr = lr
        
        # Atualiza o learning rate
        K.set_value(self.model.optimizer.lr, current_lr)
        print(f'Learning rate: {current_lr}')


class SavelModelScheduler(Callback):
    def __init__(self, file_name='', schedule=None):
        super(Callback, self).__init__()
        self.schedule = schedule or [1, 25, 50, 75, 100, 150]
        self.file_name = file_name

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if epoch in self.schedule:
            model_name = f"{self.file_name}epoch{epoch}"
            print(f'Epoch {epoch:05d}: saving model to {model_name}')
            self.model.save_weights(model_name, overwrite=True)
            with open(f"{model_name}.json", 'w') as f:
                f.write(self.model.to_json())


class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_acc', value=0.95, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        
        if current is None:
            warnings.warn(
                f"Early stopping requires {self.monitor} available!", 
                RuntimeWarning
            )
            return
        
        if current < self.value:
            if self.verbose > 0:
                print(f"Epoch {epoch:05d}: early stopping THR")
            self.model.stop_training = True


def custom_stopping(value=0.5, verbose=0):
    """Helper function to create early stopping callback"""
    return EarlyStoppingByLossVal(
        monitor='val_loss',
        value=value,
        verbose=verbose
    )