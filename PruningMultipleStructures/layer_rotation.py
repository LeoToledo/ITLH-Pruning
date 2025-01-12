# layer_rotation.py

import numpy as np
from scipy.spatial.distance import cosine
from sklearn.preprocessing import MinMaxScaler

class LayerRotationTracker:
    """
    Classe que rastreia a variação (distância cosseno) dos pesos de determinadas camadas
    ao longo do treinamento, calculando o ângulo de inclinação via regressão linear
    numa janela de épocas. Se o ângulo for menor que um limiar (angle_threshold)
    por 'stable_epochs_needed' épocas consecutivas, consideramos que a rotação está "estável".
    """

    def __init__(
        self,
        model,
        layer_names=None,
        start_epoch=600,
        angle_threshold=1.0,
        stable_epochs_needed=200,
        window_size=5
    ):
        """
        Parâmetros:
            model: O modelo Keras/TensorFlow do qual se extrairão os pesos iniciais.
            layer_names: Lista de nomes de camadas a serem rastreadas. Se None, pega
                         todas as camadas com pesos (excluindo bias).
            start_epoch: Época a partir da qual começa a verificar a rotação.
            angle_threshold: Valor do ângulo (em graus) abaixo do qual consideramos
                             "estável" na janela corrente.
            stable_epochs_needed: Quantas épocas seguidas precisam estar "estáveis"
                                  para considerarmos a rotação verdadeiramente estável.
            window_size: Tamanho da janela de épocas usada no cálculo da regressão linear.
        """
        self.start_epoch = start_epoch
        self.angle_threshold = angle_threshold
        self.stable_epochs_needed = stable_epochs_needed
        self.window_size = window_size

        # Se não for especificado, usaremos todas as camadas que tiverem pesos (matriz de pesos)
        if layer_names is None:
            layer_names = [
                layer.name for layer in model.layers
                if layer.get_weights() and len(layer.get_weights()[0].shape) > 0
            ]
        self.layer_names = layer_names

        # Guarda pesos iniciais das camadas
        self.initial_weights = [model.get_layer(name).get_weights()[0] for name in self.layer_names]

        # Listas para armazenar o histórico das distâncias médias ao longo das épocas
        self.cosine_distances_mean = []

        # Contador de épocas seguidas em que a rotação permanece estável
        self.stable_epochs_count = 0

    def update_and_check_stability(self, model, current_epoch, total_epochs):
        """
        Atualiza a distância cosseno com base nos pesos atuais do modelo
        e verifica se a rotação está estável, retornando:
            stable (bool), angle (float)

        Parâmetros:
            model: O modelo Keras/TensorFlow atual.
            current_epoch: Época atual (1-based).
            total_epochs: Número total de épocas (para normalização, ex. 500).

        Retorna:
            stable (bool): Se atingiu ou não estabilidade (rotação < limiar).
            angle (float): O ângulo calculado na janela recente.
        """
        # Antes de 'start_epoch', não fazemos o cálculo (retorna False, angle = 0.0)
        if current_epoch < self.start_epoch:
            return False, 0.0

        # Calcula a distância média e armazena
        cosine_distance_mean = np.mean(
            calculate_cosine_distance(self.initial_weights, self._get_current_weights(model))
        )
        self.cosine_distances_mean.append(cosine_distance_mean)

        # Se não temos ainda a janela mínima de épocas, retorna instável
        if len(self.cosine_distances_mean) < self.window_size:
            return False, 0.0

        # Caso contrário, calculamos a inclinação e o ângulo
        stable_now, angle = self._check_rotation_angle_criterion(
            model=model,
            layer_names=self.layer_names,
            window_size=self.window_size,
            epoch=current_epoch,
            epochs=total_epochs
        )

        # Se está estável nesta época, incrementa o contador; senão, reseta
        if stable_now:
            self.stable_epochs_count += 1
        else:
            self.stable_epochs_count = 0

        # Se o contador de épocas estáveis atingiu o necessário, retornamos True
        is_stable = (self.stable_epochs_count >= self.stable_epochs_needed)
        return is_stable, angle

    def _get_current_weights(self, model):
        """
        Retorna a lista de pesos (sem bias) para as camadas especificadas em self.layer_names.
        """
        return [model.get_layer(name).get_weights()[0] for name in self.layer_names]

    def _check_rotation_angle_criterion(self, model, layer_names, window_size, epoch, epochs):
  
        # usamos self._get_current_weights, mas mantemos a consistência com o seu snippet)
        current_weights = [model.get_layer(name).get_weights()[0] for name in layer_names]

        # Calcula a distância cosseno (média) entre os pesos atuais e os pesos iniciais
        cosine_distance_mean = np.mean(
            calculate_cosine_distance(self.initial_weights, current_weights)
        )
        print(f"\nMean cosine distance: {cosine_distance_mean:.4f}", flush=True)

        # Verifica se temos pelo menos 'window_size' distâncias para analisar
        if len(self.cosine_distances_mean) >= window_size:
            # Pega os últimos 'window_size' pontos
            recent_distances = self.cosine_distances_mean[-window_size:]
            recent_distances = np.array(recent_distances).reshape(-1, 1)

            # Cria um array de épocas de 1 até 'epochs' e normaliza
            epochs_ndarray = np.arange(1, epochs + 1).reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            epochs_norm = scaler.fit_transform(epochs_ndarray)

            # Flatten
            recent_distances_norm_1d = recent_distances.flatten()
            epochs_norm_1d = epochs_norm.flatten()

            # Selecionar a janela atual de 'window_size' pontos
            epochs_window = epochs_norm_1d[epoch - window_size : epoch]

            print("Epochs window (normalized):", epochs_window)
            print("Recent distances (original scale):", recent_distances_norm_1d)

            # Ajusta uma linha de regressão linear aos pontos
            slope, intercept = np.polyfit(epochs_window, recent_distances_norm_1d, 1)

            # Converte a inclinação para um ângulo em graus
            angle = np.degrees(np.arctan(slope))
            print(f"\nAngle: {angle:.2f}°", flush=True)

            # Verifica se o ângulo é menor que o limiar definido (angle_threshold)
            if angle < self.angle_threshold:
                return True, angle

            return False, angle

        # Se não há janela suficiente, definimos que está instável e ângulo 0
        return False, 0.0


def calculate_cosine_distance(weights1, weights2):

    distances = []
    distance = np.array([
        cosine(w1.flatten(), w2.flatten()) for w1, w2 in zip(weights1, weights2)
    ])
    distances.append(distance)
    return distances
