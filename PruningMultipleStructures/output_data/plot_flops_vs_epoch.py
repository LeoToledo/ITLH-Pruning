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
epochs = []
flops = []
pruning_events = []
pruning_epochs = []
pruning_flops = []

# Para encontrar pontos de transição nos FLOPs
prev_flops = None

with open(file_path, 'r') as f:
    for line in f:
        entry = json.loads(line)
        epochs.append(entry['epoch'])
        current_flops = entry['flops']
        flops.append(current_flops)
        
        # Detectar mudanças nos FLOPs
        if prev_flops is not None and current_flops != prev_flops:
            pruning_epochs.append(entry['epoch'])
            pruning_flops.append(current_flops)
            if entry['pruning_info'] is not None:
                pruning_type = entry['pruning_info'].split('structure=')[1].split(',')[0] if 'structure=' in entry['pruning_info'] else 'unknown'
                pruning_events.append(f"Pruning: {pruning_type}")
            else:
                pruning_events.append("FLOPs changed")
        
        prev_flops = current_flops

# Criar o plot
fig, ax = plt.subplots()

# Plot principal dos FLOPs
ax.plot(epochs, flops, color='#2ca02c', linewidth=2, label='FLOPs')

# Adicionar marcadores para eventos de mudança nos FLOPs
for epoch, flop_value, event in zip(pruning_epochs, pruning_flops, pruning_events):
    ax.scatter(epoch, flop_value, color='red', s=100, zorder=5)
    
    # Calcular a redução percentual de FLOPs
    idx = epochs.index(epoch)
    if idx > 0:
        prev_flops_value = flops[idx - 1]
        reduction = (prev_flops_value - flop_value) / prev_flops_value * 100
        annotation_text = f"{event}\n({reduction:.1f}% reduction)"
    else:
        annotation_text = event
    
    # Adicionar anotação
    ax.annotate(annotation_text,
               xy=(epoch, flop_value),
               xytext=(10, 10),
               textcoords='offset points',
               ha='left',
               va='bottom',
               bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

# Configurar o plot
ax.set_xlabel('Epoch')
ax.set_ylabel('FLOPs')
ax.set_title('FLOPs Evolution with Pruning Events')
ax.grid(True, linestyle='--', alpha=0.3)
ax.legend()

# Formatar o eixo y para usar notação científica mais legível
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))

# Ajustar o layout
plt.tight_layout()

# Salvar a figura
#plt.savefig('flops_plot.pdf', bbox_inches='tight', dpi=300)
plt.savefig('flops_plot.png', bbox_inches='tight', dpi=300)
plt.close()