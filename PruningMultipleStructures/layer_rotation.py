# layer_rotation.py

import numpy as np
import math

class LayerRotationTracker:
    """
    Tracks how much each layer's weights have rotated from their initialization.
    The "angle" here is computed as the average (over tracked layers) of the angle (in degrees)
    between the weight vectors at initialization and the current weight vectors.
    When the derivative (i.e. the change in the mean angle from the previous epoch) remains below a
    set threshold for a number of consecutive epochs, the tracker signals that the model is stable enough
    to consider pruning.
    """
    def __init__(self, model, layer_names=None, derivative_threshold=0.1, stable_epochs_needed=2):
        """
        Args:
            model: Keras model.
            layer_names: Optional list of layer names to track. If None, all layers with weights are tracked.
            derivative_threshold: Threshold (in degrees) for the change in angle between epochs below which is considered stable.
            stable_epochs_needed: Number of consecutive epochs with derivative below the threshold needed to trigger pruning.
        """
        self.derivative_threshold = derivative_threshold
        self.stable_epochs_needed = stable_epochs_needed
        self.consecutive_stable_count = 0
        self.previous_angle = None

        # If no specific layers are provided, track all layers that have weights.
        if layer_names is None:
            self.layer_names = [layer.name for layer in model.layers if len(layer.get_weights()) > 0]
        else:
            self.layer_names = layer_names

        # Store the initial weights from the model as the baseline.
        self.initial_weights = [model.get_layer(name).get_weights()[0] for name in self.layer_names]

        # Optional: Log of computed angles for later analysis.
        self.angles = []

    def compute_epoch_angle(self, model):
        """
        Computes the average angle (in degrees) between the initial weights and the current weights.
        
        Args:
            model: The current Keras model.
        
        Returns:
            mean_angle (float): The average angle (in degrees) computed over all tracked layers.
            current_weights (list): A list containing the current weights for each tracked layer.
        """
        # Retrieve current weights for each tracked layer.
        current_weights = [model.get_layer(name).get_weights()[0] for name in self.layer_names]

        angles = []
        # Compute the angle for each layer between the initial and current weights.
        for w_init, w_curr in zip(self.initial_weights, current_weights):
            # Flatten the weight arrays.
            w_init_flat = w_init.flatten()
            w_curr_flat = w_curr.flatten()
            # Compute the norms (adding a small constant for numerical stability).
            norm_init = np.linalg.norm(w_init_flat) + 1e-8
            norm_curr = np.linalg.norm(w_curr_flat) + 1e-8
            # Compute the cosine similarity.
            cos_sim = np.dot(w_init_flat, w_curr_flat) / (norm_init * norm_curr)
            # Clip the cosine similarity to the interval [-1, 1] to avoid numerical errors.
            cos_sim = np.clip(cos_sim, -1.0, 1.0)
            # Compute the angle (in degrees) from the cosine similarity.
            angle_deg = np.degrees(np.arccos(cos_sim))
            angles.append(angle_deg)
        mean_angle = np.mean(angles)
        return mean_angle, current_weights

    def update_and_check_stability(self, model):
        """
        Updates the tracker using the current model and checks whether the derivative (change) in the angle
        from the previous epoch has been below the threshold for enough consecutive epochs to trigger pruning.
        
        Args:
            model: The current Keras model.
        
        Returns:
            stable (bool): True if the derivative has remained below the threshold for the required number of consecutive epochs.
            mean_angle (float): The computed average angle for the current epoch.
        """
        mean_angle, current_weights = self.compute_epoch_angle(model)

        # Log the computed angle.
        self.angles.append(mean_angle)
        print(f"[LayerRotationTracker] Mean Angle (from initialization): {mean_angle:.2f}°")

        # If this is the first epoch, initialize previous_angle and do not count stability.
        if self.previous_angle is None:
            self.previous_angle = mean_angle
            return False, mean_angle

        # Calculate the derivative (absolute difference between the current and previous mean angle).
        derivative = abs(mean_angle - self.previous_angle)
        print(f"[LayerRotationTracker] Angle Derivative: {derivative:.2f}°")

        # Update previous_angle for the next epoch.
        self.previous_angle = mean_angle

        # Check whether the derivative is below the threshold.
        if derivative < self.derivative_threshold:
            self.consecutive_stable_count += 1
        else:
            self.consecutive_stable_count = 0

        # If the derivative has remained below the threshold for enough consecutive epochs, signal stability.
        stable = self.consecutive_stable_count >= self.stable_epochs_needed

        return stable, mean_angle
