import numpy as np
from scipy.spatial.distance import cosine
import math
from sklearn.preprocessing import MinMaxScaler

class LayerRotationTracker:
    """
    Tracks how much each layer's weights have "rotated" (changed in direction)
    compared to the initial weights, using cosine distance, replicating
    the logic of angle (slope) calculation from the original article.
    """

    def __init__(
        self,
        model,
        layer_names=None,
        start_epoch=5,
        angle_threshold=45.0,
        stable_epochs_needed=2
    ):
        """
        Args:
            model: Keras model
            layer_names: list of layer names (optional).
            start_epoch: number of epochs used in linear regression (5 in the article).
            angle_threshold: angle threshold (45° by default, as in the article).
            stable_epochs_needed: number of consecutive epochs that must be below
                                  the angle threshold before returning True.
        """
        self.start_epoch = start_epoch
        self.angle_threshold = angle_threshold
        self.stable_epochs_needed = stable_epochs_needed
        self.consecutive_stable_count = 0

        # If layer_names is not specified, we use all layers that have weights
        if layer_names is None:
            self.layer_names = [layer.name for layer in model.layers if len(layer.get_weights()) > 0]
        else:
            self.layer_names = layer_names

        # Save initial weights
        self.initial_weights = []
        for name in self.layer_names:
            w_init = model.get_layer(name).get_weights()[0]
            self.initial_weights.append(w_init.copy())

        # List for storing mean cosine distances
        self.cosine_distances_mean = []

    def calculate_mean_cosine_distance(self, model):
        """
        Calculates the average cosine distance between current weights and initial weights.
        """
        current_weights = [model.get_layer(name).get_weights()[0] for name in self.layer_names]
        distances = []
        for w_init, w_curr in zip(self.initial_weights, current_weights):
            dist = cosine(w_init.flatten(), w_curr.flatten())
            distances.append(dist)
        mean_distance = np.mean(distances)
        
        # (Prints removed as requested)
        # print(f"[LayerRotationTracker] Mean Cosine Distance: {mean_distance:.4f}")
        
        return mean_distance

    def update_and_check_stability(self, model, current_epoch, total_epochs):
        """
        Returns:
            stable (bool): True if rotation is considered "stable" this epoch.
            angle (float): The computed angle this epoch (0 if insufficient data).
        """
        # 1) Calculate mean cosine distance
        distance_mean = self.calculate_mean_cosine_distance(model)
        self.cosine_distances_mean.append(distance_mean)

        # Default angle = 0.0 (if we don't have enough data)
        angle = 0.0
        stable = False

        # 2) Only compute slope/angle if we have at least `start_epoch` points
        if len(self.cosine_distances_mean) >= self.start_epoch:
            # Take the last 'start_epoch' distances
            recent_distances = self.cosine_distances_mean[-self.start_epoch:]
            recent_distances = np.array(recent_distances).reshape(-1, 1)

            # Create an array of epochs [1..total_epochs] and normalize with MinMaxScaler
            epochs_ndarray = np.arange(1, total_epochs + 1).reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            epochs_norm = scaler.fit_transform(epochs_ndarray)
            epochs_norm_1d = epochs_norm.flatten()

            # Select the window of normalized epochs
            # E.g., if current_epoch=10 and start_epoch=5 => we take [5..10) => indices 5..9
            start_index = current_epoch - self.start_epoch
            end_index = current_epoch
            if start_index < 0:
                start_index = 0

            epochs_window = epochs_norm_1d[start_index:end_index]

            # Linear fit (degree=1) using epochs_window (x) and the cosine distances (y)
            recent_distances_norm_1d = recent_distances.flatten()
            slope, intercept = np.polyfit(epochs_window, recent_distances_norm_1d, 1)

            # Convert slope to angle in degrees
            angle = np.degrees(np.arctan(slope))
            print(f"[LayerRotationTracker] Angle: {angle:.2f}° (slope-based)")

            # 3) If angle < threshold => increment self.consecutive_stable_count
            if angle < self.angle_threshold:
                self.consecutive_stable_count += 1
            else:
                self.consecutive_stable_count = 0

            # 4) If we reached 'stable_epochs_needed', mark stable as True
            if self.consecutive_stable_count >= self.stable_epochs_needed:
                stable = True

        return stable, angle
