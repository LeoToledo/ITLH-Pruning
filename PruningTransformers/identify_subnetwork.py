# identify_subnetwork.py

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from typing import Dict, Tuple, List

# Variável global para guardar top_indices da última chamada
last_top_indices_by_layer = {}

def get_subnetwork(model: tf.keras.Model, 
                   prune_ratio: float, 
                   epoch: int) -> Tuple[Dict, Dict]:
    """
    Gets the dominant sub-network architecture based on neuron importance using magnitude criterion
    as described in 'When to Prune? A Policy towards Early Structural Pruning'
    
    Args:
        model: current transformer model state
        prune_ratio: ratio of neurons to prune (α in the paper), between 0 and 1
        epoch: current epoch index (for debugging/printing)
        
    Returns:
        structure: dict with number of active neurons per layer and their masks
        importance_scores: dict with importance scores for each neuron
    """
    if not 0 <= prune_ratio < 1:
        raise ValueError("prune_ratio must be between 0 and 1")
        
    # Get attention layers
    attention_layers = [
        layer for layer in model.layers 
        if isinstance(layer, layers.MultiHeadAttention)
    ]
    
    if not attention_layers:
        raise ValueError("No attention layers found in model")
    
    importance_scores = {}
    structure = {}
    
    # Explicação da mudança no top_indices:
    # O conjunto top_indices muda quando os pesos dos neurônios mudam ao longo do treinamento,
    # alterando o valor I_n^l = ||W_n^l||_2/sqrt(P^l). Isso ocorre porque a otimização (via SGD ou outro otimizador)
    # atualiza W_n^l a cada batch/época, possivelmente mudando o ranking dos neurônios.
    
    for idx, layer in enumerate(attention_layers):
        try:
            # Get weights of attention layer
            weights = layer.get_weights()  

            if len(weights) < 8:
                raise ValueError(f"Layer {idx} does not have expected 8 weights (Q, K, V, Output kernels and biases)")
                
            # Extract Query, Key, Value kernels (skip biases)
            query_weight = weights[0]  # Query Kernel
            key_weight = weights[2]    # Key Kernel
            value_weight = weights[4]  # Value Kernel
            
            # Calculate total parameters per head considering all weights (Q,K,V)
            P_l = (np.prod(query_weight.shape) + 
                   np.prod(key_weight.shape) + 
                   np.prod(value_weight.shape))

            # Reshape weights to (num_heads, -1) for each component
            # Assuming weights are shaped [input_dim, key_dim] etc., adjust if necessary
            q_reshaped = query_weight.reshape(layer._num_heads, -1)
            k_reshaped = key_weight.reshape(layer._num_heads, -1)
            v_reshaped = value_weight.reshape(layer._num_heads, -1)
            
            # Calculate magnitude-based importance according to the paper's equation (2)
            # I_n^l = ||W_n^l||_2/sqrt(P^l)
            head_scores = (np.linalg.norm(q_reshaped, axis=1) + 
                           np.linalg.norm(k_reshaped, axis=1) + 
                           np.linalg.norm(v_reshaped, axis=1)) / np.sqrt(P_l)
            
            # Store raw importance scores
            importance_scores[f'layer_{idx}'] = head_scores
            
            # Calculate number of heads to keep based on prune_ratio
            n_heads = layer._num_heads
            n_keep = max(1, int(n_heads * (1 - prune_ratio)))  # Keep at least 1 head
        
            # Get top n_keep heads by importance score
            top_indices = np.argpartition(head_scores, -n_keep)[-n_keep:]
            active_mask = np.zeros_like(head_scores, dtype=bool)
            active_mask[top_indices] = True
            
            structure[f'layer_{idx}'] = {
                'total_heads': n_heads,
                'active_heads': n_keep,
                'head_scores': head_scores,
                'active_mask': active_mask,
                'parameters_per_head': P_l,
                'top_indices': top_indices,
                'mean_importance': np.mean(head_scores[top_indices]),
                'min_importance': np.min(head_scores[top_indices]),
                'max_importance': np.max(head_scores[top_indices])
            }
            
            # [NOVO PRINT] Calcular porcentagem de mudança no top_indices em relação à época anterior
            # Apenas se tivermos histórico
            if idx in last_top_indices_by_layer:
                old_top = last_top_indices_by_layer[idx]
                # Quantos heads não estavam no top antes?
                # Convertemos em sets para facilitar a comparação
                old_set = set(old_top)
                new_set = set(top_indices)
                changed = len(new_set - old_set)  # quantos são novos agora
                perc_changed = (changed / len(new_set)) * 100
                print(f"[DEBUG-TOP-CHANGE] Epoch {epoch}, Layer {idx}: {perc_changed:.2f}% of top indices changed.")
            else:
                print(f"[DEBUG-TOP-CHANGE] Epoch {epoch}, Layer {idx}: No previous data to compare.")
            
            # Atualiza histórico
            last_top_indices_by_layer[idx] = top_indices.copy()
            
        except Exception as e:
            print(f"Error processing layer {idx}: {str(e)}")
            raise
    
    # Print network structure overview
    print("\nNetwork Structure Overview:")
    print("Layer\t\tTotal Heads\tActive Heads\tRetention %\tMean Importance")
    print("-" * 80)
    
    for idx in range(len(attention_layers)):
        info = structure[f'layer_{idx}']
        total = info['total_heads']
        active = info['active_heads']
        retention = (active / total) * 100
        mean_imp = info['mean_importance']
        
        print(f"Layer {idx}\t{total}\t\t{active}\t\t{retention:.2f}%\t\t{mean_imp:.4f}")
    
    # Validation check
    total_neurons = sum(s['total_heads'] for s in structure.values())
    kept_neurons = sum(s['active_heads'] for s in structure.values())
    actual_ratio = 1 - (kept_neurons / total_neurons)
    
    print(f"\nOverall Statistics:")
    print(f"Total neurons: {total_neurons}")
    print(f"Kept neurons: {kept_neurons}")
    print(f"Target prune ratio: {prune_ratio:.2f}")
    print(f"Actual prune ratio: {actual_ratio:.2f}")
    
    return structure, importance_scores
