import os
import datetime
import pandas as pd
from tensorflow import keras
import traceback

def setup_checkpoint_directories():
    """
    Creates a structured directory system for checkpoints and logs.
    Returns the base directory path for the current training run.
    """
    # Create base checkpoints directory if it doesn't exist
    base_dir = 'checkpoints'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Create a timestamped directory for this training run
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(base_dir, f'run_{timestamp}')
    
    # Create subdirectories for different types of checkpoints
    subdirs = {
        'regular': os.path.join(run_dir, 'regular_checkpoints'),
        'pruning': os.path.join(run_dir, 'pruning_checkpoints'),
        'final': os.path.join(run_dir, 'final'),
        'history': os.path.join(run_dir, 'history')
    }
    
    for dir_path in subdirs.values():
        os.makedirs(dir_path)
    
    return run_dir, subdirs

def save_history_safely(history, filename):
    """
    Salva o histórico garantindo que todas as listas tenham o mesmo tamanho
    """
    # Encontra o menor tamanho entre todas as listas
    min_length = min(len(v) for v in history.values())
    
    # Trunca todas as listas para o mesmo tamanho
    safe_history = {
        k: v[:min_length] for k, v in history.items()
    }
    
    # Salva o histórico
    hist_df = pd.DataFrame(safe_history)
    hist_df.to_csv(filename, index=False)

def save_checkpoint(model, history, checkpoint_dirs, checkpoint_type, epoch=None, status=None, training_params=None):
    """
    Saves model and history checkpoints in the appropriate directory.
    """
    try:
        if checkpoint_type == 'regular':
            model_path = os.path.join(
                checkpoint_dirs['regular'],
                f'model_checkpoint_epoch_{epoch}.h5'
            )
            history_path = os.path.join(
                checkpoint_dirs['history'],
                f'history_checkpoint_epoch_{epoch}.csv'
            )
        
        elif checkpoint_type == 'pruning':
            assert status in ['before', 'after'], "Status must be 'before' or 'after' for pruning checkpoints"
            model_path = os.path.join(
                checkpoint_dirs['pruning'],
                f'model_{status}_pruning_epoch_{epoch}.h5'
            )
            history_path = os.path.join(
                checkpoint_dirs['history'],
                f'history_{status}_pruning_epoch_{epoch}.csv'
            )
        
        elif checkpoint_type == 'final':
            model_path = os.path.join(checkpoint_dirs['final'], 'model_final.h5')
            history_path = os.path.join(checkpoint_dirs['final'], 'history_final.csv')
        
        elif checkpoint_type == 'error':
            model_path = os.path.join(checkpoint_dirs['final'], 'model_at_error.h5')
            history_path = os.path.join(checkpoint_dirs['final'], 'history_at_error.csv')
        
        else:
            raise ValueError(f"Invalid checkpoint type: {checkpoint_type}")
        
        # Save model and history
        model.save(model_path)
        save_history_safely(history, history_path)
        
        print(f"Checkpoint saved successfully:\n - Model: {model_path}\n - History: {history_path}")
        
        # Save experiment parameters for final checkpoints
        if checkpoint_type == 'final' and training_params is not None:
            params_path = os.path.join(checkpoint_dirs['final'], 'experiment_params.txt')
            with open(params_path, 'w') as f:
                for key, value in training_params.items():
                    f.write(f'{key}: {value}\n')
            
    except Exception as e:
        print(f"\nError saving checkpoint: {str(e)}")
        print("Stack trace:")
        traceback.print_exc()
        raise

def load_latest_checkpoint(checkpoint_dirs, checkpoint_type='regular'):
    """
    Loads the latest checkpoint of the specified type.
    """
    try:
        if checkpoint_type == 'regular':
            checkpoint_dir = checkpoint_dirs['regular']
        elif checkpoint_type == 'pruning':
            checkpoint_dir = checkpoint_dirs['pruning']
        else:
            raise ValueError(f"Invalid checkpoint type for loading: {checkpoint_type}")
        
        # Find the latest checkpoint
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.h5')]
        if not checkpoints:
            return None, None, None
        
        latest_checkpoint = max(checkpoints)
        epoch_number = int(latest_checkpoint.split('_')[-1].split('.')[0])
        
        # Load model
        model_path = os.path.join(checkpoint_dir, latest_checkpoint)
        model = keras.models.load_model(model_path)
        
        # Load corresponding history
        history_path = os.path.join(
            checkpoint_dirs['history'],
            f'history_checkpoint_epoch_{epoch_number}.csv'
        )
        history = pd.read_csv(history_path).to_dict('list')
        
        print(f"Loaded checkpoint from epoch {epoch_number}")
        return model, history, epoch_number
        
    except Exception as e:
        print(f"\nError loading checkpoint: {str(e)}")
        print("Stack trace:")
        traceback.print_exc()
        return None, None, None