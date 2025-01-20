import matplotlib.pyplot as plt
import json
import numpy as np
import os

# Configuração para estilo mais adequado para publicação
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.figsize': (8, 6),  # Ajustado para plots individuais
    'axes.grid': True,
    'grid.alpha': 0.3
})

# Função para detectar pontos de poda
def detect_pruning_points(data):
    pruning_epochs = []
    prev_flops = None
    
    for i, entry in enumerate(data):
        if prev_flops is not None and entry['flops'] < prev_flops:
            pruning_epochs.append(i)
        prev_flops = entry['flops']
    
    return pruning_epochs

# Definir o caminho do arquivo
file_path = './pruning/train_log.jsonl'
#file_path = './no_pruning/train_log.jsonl'

if not os.path.exists(file_path):
    raise FileNotFoundError(f"O arquivo {file_path} não foi encontrado!")

# Extrair o diretório do arquivo de entrada
output_dir = os.path.dirname(file_path)
print(f"Usando arquivo: {file_path}")
print(f"Salvando plots em: {output_dir}")

# Carregar e processar dados
data = []
epochs = []
train_acc = []
val_acc = []
flops = []

with open(file_path, 'r') as f:
    for line in f:
        entry = json.loads(line)
        data.append(entry)
        epochs.append(entry['epoch'])
        train_acc.append(entry['train_acc'])
        val_acc.append(entry['val_acc'])
        flops.append(entry['flops'])

pruning_epochs = detect_pruning_points(data)

# Função para criar e salvar plot individual
def create_accuracy_plot(epochs, acc_data, title, ylabel, filename, pruning_epochs):
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    
    # Plot principal com linha mais fina
    plt.plot(epochs, acc_data, linewidth=1.5, label=ylabel)
    
    # Linhas verticais para eventos de poda
    for ep in pruning_epochs:
        plt.axvline(x=epochs[ep], color='#d62728', linestyle='--', alpha=0.5, linewidth=1)
    
    # Configuração dos eixos com mais detalhes no eixo y
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    
    # Ajustar limites e ticks do eixo y para mais detalhes na região de interesse
    min_acc = min(acc_data)
    y_min = max(0, min_acc - 0.1)  # Garantir que não fique negativo
    plt.ylim(y_min, 1.02)  # Limite superior um pouco acima de 1 para visualização
    
    # Mais ticks no eixo y, especialmente na região de interesse (0.8-1.0)
    y_ticks = np.concatenate([
        np.arange(round(y_min, 1), 0.8, 0.1),  # Ticks espaçados de 0.1 até 0.8
        np.arange(0.8, 1.02, 0.02)  # Ticks mais detalhados de 0.8 até 1.0
    ])
    plt.yticks(y_ticks)
    
    # Grade e legenda
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    
    # Ajustar layout e salvar
    plt.tight_layout()
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, bbox_inches='tight', dpi=600)  # Aumentada a resolução
    plt.close()

# Criar plots individuais
create_accuracy_plot(
    epochs, 
    train_acc, 
    'Training Accuracy vs. Epoch', 
    'Training Accuracy', 
    'training_accuracy.png',
    pruning_epochs
)

create_accuracy_plot(
    epochs, 
    val_acc, 
    'Validation Accuracy vs. Epoch', 
    'Validation Accuracy', 
    'validation_accuracy.png',
    pruning_epochs
)