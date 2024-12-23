import numpy as np
from collections import deque
from typing import Dict, List, Tuple

class EPI:
    """
    Implementation of Early Pruning Indicator (EPI) as described in 
    'When to Prune? A Policy towards Early Structural Pruning'
    """
    def __init__(self, r: int = 20, tau_magnitude: float = 0.983, tau_gradient: float = 0.944):
        """
        Initialize EPI calculator
        Args:
            r: number of past epochs to consider for stability comparison
            tau_magnitude: threshold for magnitude-based pruning (default from paper)
            tau_gradient: threshold for gradient-based pruning (default from paper)
        """
        self.r = r
        self.tau_magnitude = tau_magnitude
        self.tau_gradient = tau_gradient
        self.subnetwork_history = deque(maxlen=r)
        self.current_method = None

    def calculate_layer_distance(self, n1: int, n2: int) -> float:
        """
        Calculates normalized difference between layers as per equation (4) in paper:
        d_l(N1,N2) = |n1 - n2| / (n1 + n2)
        
        Args:
            n1: number of active neurons in first network's layer
            n2: number of active neurons in second network's layer
            
        Returns:
            Normalized difference between 0 and 1
        """
        if n1 + n2 == 0:  # Handle edge case of both layers being empty
            return 0.0
        return abs(n1 - n2) / (n1 + n2)

    def calculate_psi(self, network1: Dict, network2: Dict) -> float:
        """
        Calculates structural similarity (Ψ) between two sub-networks as per equation (5):
        Ψ(N1,N2) = 1 - (1/L)∑(d_l(N1,N2))
        
        Args:
            network1: first network structure with layer statistics
            network2: second network structure with layer statistics
            
        Returns:
            Similarity score between 0 and 1, where 1 indicates identical structures
        """
        total_distance = 0
        n_layers = len(network1)
        
        for layer_idx in range(n_layers):
            n1 = network1[f'layer_{layer_idx}']['active_heads']
            n2 = network2[f'layer_{layer_idx}']['active_heads']
            total_distance += self.calculate_layer_distance(n1, n2)
            
        psi = 1 - (total_distance / n_layers)
        return psi

    def calculate_epi(self, current_network: Dict) -> float:
        """
        Calculates EPI score as per equation (6):
        EPI_t = (1/r)∑(j=1 to r)Ψ(N_t, N_(t-j))
        
        Args:
            current_network: current sub-network structure
            
        Returns:
            EPI score between 0 and 1
        """
        if len(self.subnetwork_history) == 0:
            self.subnetwork_history.append(current_network)
            return 0.0
            
        # Calculate Ψ between current network and past r networks
        psi_values = []
        for past_network in self.subnetwork_history:
            psi = self.calculate_psi(current_network, past_network)
            psi_values.append(psi)

        # Add current network to history
        self.subnetwork_history.append(current_network)
        
        # Calculate EPI as average of Ψ values
        return np.mean(psi_values)

    def should_prune(self, current_network: Dict, method: str = 'magnitude') -> bool:
        """
        Determines if pruning should occur based on EPI value and method-specific threshold
        
        Args:
            current_network: current sub-network structure
            method: 'magnitude' or 'gradient' pruning method
            
        Returns:
            bool: True if network structure is stable enough for pruning
        """
        self.current_method = method
        threshold = self.tau_magnitude if method == 'magnitude' else self.tau_gradient
        
        current_epi = self.calculate_epi(current_network)
        
        # As specified in the paper, EPI should be above threshold
        # and we need enough history for stability comparison
        if len(self.subnetwork_history) < self.r:
            return False
            
        # Check if current EPI exceeds threshold and is greater than
        # previous values (indicating stability)
        if current_epi < threshold:
            return False
            
        # Check if EPI has been increasing (indicating convergence to stability)
        prev_epis = [
            self.calculate_epi(net) 
            for net in list(self.subnetwork_history)[-5:]
        ]
        if not all(current_epi >= prev_epi for prev_epi in prev_epis):
            return False
            
        return True

    def get_network_statistics(self, network: Dict) -> Dict:
        """
        Calculates and returns useful statistics about the network structure
        
        Args:
            network: network structure
            
        Returns:
            Dict containing various network statistics
        """
        total_heads = 0
        active_heads = 0
        
        for layer_name, layer_info in network.items():
            total_heads += layer_info['total_heads']
            active_heads += layer_info['active_heads']
            
        return {
            'total_heads': total_heads,
            'active_heads': active_heads,
            'pruning_ratio': 1 - (active_heads / total_heads),
            'n_layers': len(network)
        }

    def reset(self):
        """
        Resets EPI calculator state
        """
        self.subnetwork_history.clear()
        self.current_method = None