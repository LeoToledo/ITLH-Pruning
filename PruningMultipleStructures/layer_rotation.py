import numpy as np
from scipy.spatial.distance import cosine
import math
from sklearn.preprocessing import MinMaxScaler


class LayerRotationTracker:
    """
    Tracks how much each layer's weights have "rotated" (changed in direction)
    compared to the initial weights, using cosine distance.

    Now enhanced with:
      - A 'window_size' to focus on only the last N epochs (local slope).
      - A simpler local slope calculation if we have just a few data points.
    """

    def __init__(
        self,
        model,
        layer_names=None,
        start_epoch=5,
        angle_threshold=45.0,
        stable_epochs_needed=2,
        window_size=10,            # Nova configuração para definir a quantidade de épocas recentes na janela
        use_local_normalization=True  # Se True, normaliza apenas a janela (pode desativar se quiser)
    ):
        """
        Args:
            model: Keras model
            layer_names: list of layer names to track (optional).
            start_epoch: only start checking stability after these many epochs.
            angle_threshold: slope-based angle threshold (in degrees).
            stable_epochs_needed: number of consecutive epochs that must be below
                                  angle_threshold to declare "stable".
            window_size: how many recent epochs to consider when computing slope.
            use_local_normalization: if True, normalizes epochs in [0,1] within the
                                     window; if False, uses epochs "as is".
        """
        self.start_epoch = start_epoch
        self.angle_threshold = angle_threshold
        self.stable_epochs_needed = stable_epochs_needed
        self.window_size = window_size
        self.use_local_normalization = use_local_normalization

        self.consecutive_stable_count = 0

        # Se layer_names não for especificado, rastreia todas as camadas com pesos
        if layer_names is None:
            self.layer_names = [layer.name for layer in model.layers if len(layer.get_weights()) > 0]
        else:
            self.layer_names = layer_names

        # Salva pesos iniciais para referência
        self.initial_weights = []
        for name in self.layer_names:
            w_init = model.get_layer(name).get_weights()[0]
            self.initial_weights.append(w_init.copy())

        # Lista que armazena a média da distância cosseno a cada época
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
        return mean_distance

    def compute_local_slope_angle(self, distances, epochs):
        """
        Faz um cálculo "local" do slope baseando-se SOMENTE nos últimos points da janela.
        - distances: array/list dos valores de distância cosseno, restringidos à janela
        - epochs: array/list das épocas correspondentes, também restrito à janela

        Retorna o ângulo em graus baseado no slope. Se só houver 2 pontos, calcula diferença direta.
        """
        # Se tivermos menos de 2 pontos, não é possível calcular slope
        if len(distances) < 2:
            return 0.0

        # Se forem apenas 2 pontos, cálculo direto do "slope" = (dist2 - dist1) / (epoch2 - epoch1)
        if len(distances) == 2:
            d1, d2 = distances
            e1, e2 = epochs
            # slope é a variação em função do delta de épocas
            slope = (d2 - d1) / (max(e2 - e1, 1e-5))  # evita divisão por zero
            # Converte slope -> ângulo
            angle = np.degrees(np.arctan(slope))
            return angle

        # Caso geral (>= 3 pontos), podemos usar np.polyfit
        if self.use_local_normalization:
            # Normaliza epochs dentro da própria janela
            scaler = MinMaxScaler(feature_range=(0, 1))
            epochs_reshaped = np.array(epochs).reshape(-1, 1)
            epochs_norm = scaler.fit_transform(epochs_reshaped).flatten()
            slope, _ = np.polyfit(epochs_norm, distances, 1)
        else:
            # Usa epochs como estão
            slope, _ = np.polyfit(epochs, distances, 1)

        angle = np.degrees(np.arctan(slope))
        return angle

    def calculate_angle(self):
        """
        Calcula o ângulo (slope) usando apenas as últimas `window_size` distâncias.
        Retorna 0.0 caso não haja dados suficientes.
        """
        if len(self.cosine_distances_mean) < 2:
            return 0.0

        # Pega apenas as últimas N (self.window_size) distâncias
        recent_distances = self.cosine_distances_mean[-self.window_size:]
        # Cria um vetor de épocas que corresponde a esses pontos finais
        # (Por exemplo, se já passamos de 50 épocas, e window_size=10,
        #  vamos criar algo como [41,42,...,50] ou algo nesse sentido)
        last_epoch = len(self.cosine_distances_mean)  # total de épocas já registradas
        start_index = max(last_epoch - self.window_size, 0)  # índice para pegar as distâncias
        epochs_window = list(range(start_index + 1, last_epoch + 1))  # +1 pois época começa em 1,2,...

        # Converte para arrays
        recent_distances = np.array(recent_distances)
        epochs_window = np.array(epochs_window)

        # Calcula slope local (ângulo) nessa janela
        angle = self.compute_local_slope_angle(recent_distances, epochs_window)
        return angle

    def update_and_check_stability(self, model, current_epoch, total_epochs):
        """
        Passo principal a ser chamado a cada época. 
        Retorna:
            stable (bool): True se consideramos que estabilizou nessa época.
            angle (float): O ângulo calculado (para debug/log).
        """
        # 1) Calcula e armazena a distância cosseno média
        distance_mean = self.calculate_mean_cosine_distance(model)
        self.cosine_distances_mean.append(distance_mean)

        # 2) Calcula o ângulo local (slope baseado somente na janela)
        angle = self.calculate_angle()

        # Inicialmente, assume que não está estável
        stable = False

        # Só começa a checar estabilidade se len >= start_epoch
        if len(self.cosine_distances_mean) >= self.start_epoch:
            # Basicamente, se angle < threshold, incrementa; senão, zera
            if abs(angle) < self.angle_threshold:
                self.consecutive_stable_count += 1
            else:
                self.consecutive_stable_count = 0

            # Se já ficamos 'stable_epochs_needed' épocas consecutivas com ângulo baixo, definimos como estável
            if self.consecutive_stable_count >= self.stable_epochs_needed:
                stable = True

            print(f"[LayerRotationTracker] local angle: {angle:.2f}°  "
                  f"(consecutive_stable_count={self.consecutive_stable_count}, stable={stable})")

        return stable, angle
