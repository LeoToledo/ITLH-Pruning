import matplotlib.pyplot as plt
import json
import numpy as np
import os

# Configurações do plot
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.figsize': (10, 6),
    'axes.grid': True,
    'grid.alpha': 0.3
})

# Definir o caminho do arquivo
file_path = '../train_log.jsonl'

if not os.path.exists(file_path):
    raise FileNotFoundError(f"O arquivo {file_path} não foi encontrado!")

# Carregar e processar dados
data = []
epochs = []
angles = []
pruning_events = []
pruning_epochs = []

with open(file_path, 'r') as f:
    for line in f:
        entry = json.loads(line)
        data.append(entry)
        epochs.append(entry['epoch'])
        angles.append(entry['angle'])
        
        # Detectar eventos de poda
        if entry['pruning_info'] is not None:
            pruning_epochs.append(entry['epoch'])
            pruning_events.append(entry['pruning_info'])

# Criar o plot
fig, ax = plt.subplots()

# Plot principal do ângulo
ax.plot(epochs, angles, color='#1f77b4', linewidth=2, label='Angle')

# Adicionar marcadores para eventos de poda
for epoch, event in zip(pruning_epochs, pruning_events):
    # Extrair o tipo de poda do texto do evento
    pruning_type = event.split('structure=')[1].split(',')[0] if 'structure=' in event else 'unknown'
    
    # Encontrar o valor do ângulo para esta época
    angle_value = angles[epochs.index(epoch)]
    
    # Adicionar marcador e anotação
    ax.scatter(epoch, angle_value, color='red', s=100, zorder=5)
    ax.annotate(f'Pruning: {pruning_type}',
               xy=(epoch, angle_value),
               xytext=(10, 10),
               textcoords='offset points',
               ha='left',
               va='bottom',
               bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

# Configurar o plot
ax.set_xlabel('Epoch')
ax.set_ylabel('Angle (radians)')
ax.set_title('Angle Evolution with Pruning Events')
ax.grid(True, linestyle='--', alpha=0.3)
ax.legend()

# Ajustar o layout
plt.tight_layout()

# Salvar a figura
#plt.savefig('angle_plot.pdf', bbox_inches='tight', dpi=300)
plt.savefig('angle_plot.png', bbox_inches='tight', dpi=300)
plt.close()