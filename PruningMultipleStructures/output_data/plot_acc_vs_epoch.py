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
    'figure.figsize': (12, 4),
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

# Carregar e processar dados
data = []
epochs = []
train_acc = []
val_acc = []
test_acc = []
flops = []

# Tentar diferentes caminhos possíveis para o arquivo
file_paths = [
    '../train_log.jsonl',  # Caminho relativo mencionado
    'train_log.jsonl',     # Na pasta atual
    'paste.txt'            # Nome original
]

file_path = None
for path in file_paths:
    if os.path.exists(path):
        file_path = path
        break

if file_path is None:
    raise FileNotFoundError(f"Não foi possível encontrar o arquivo de dados. Tentei os seguintes caminhos: {file_paths}")

print(f"Usando arquivo: {file_path}")

with open(file_path, 'r') as f:
    for line in f:
        entry = json.loads(line)
        data.append(entry)
        epochs.append(entry['epoch'])
        train_acc.append(entry['train_acc'])
        val_acc.append(entry['val_acc'])
        test_acc.append(entry['test_acc'])
        flops.append(entry['flops'])

pruning_epochs = detect_pruning_points(data)

# Criar os três subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
fig.subplots_adjust(wspace=0.3)

# Plot 1: Training Accuracy
ax1.plot(epochs, train_acc, color='#1f77b4', linewidth=2, label='Training Acc.')
for ep in pruning_epochs:
    ax1.axvline(x=epochs[ep], color='#d62728', linestyle='--', alpha=0.5)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Accuracy')
ax1.set_title('(a) Training Accuracy vs. Epoch')
ax1.grid(True, linestyle='--', alpha=0.3)
ax1.set_ylim(0, 1.1)

# Plot 2: Validation Accuracy
ax2.plot(epochs, val_acc, color='#2ca02c', linewidth=2, label='Validation Acc.')
for ep in pruning_epochs:
    ax2.axvline(x=epochs[ep], color='#d62728', linestyle='--', alpha=0.5)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Validation Accuracy')
ax2.set_title('(b) Validation Accuracy vs. Epoch')
ax2.grid(True, linestyle='--', alpha=0.3)
ax2.set_ylim(0, 1.1)

# Plot 3: Test Accuracy
ax3.plot(epochs, test_acc, color='#ff7f0e', linewidth=2, label='Test Acc.')
for ep in pruning_epochs:
    ax3.axvline(x=epochs[ep], color='#d62728', linestyle='--', alpha=0.5)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Test Accuracy')
ax3.set_title('(c) Test Accuracy vs. Epoch')
ax3.grid(True, linestyle='--', alpha=0.3)
ax3.set_ylim(0, 1.1)

# Adicionar legenda nos três gráficos
for ax in [ax1, ax2, ax3]:
    ax.legend()

# Ajustar o layout
plt.tight_layout()

# Salvar a figura
#plt.savefig('accuracy_plots.pdf', bbox_inches='tight', dpi=300)
plt.savefig('accuracy_plots.png', bbox_inches='tight', dpi=300)
plt.close()