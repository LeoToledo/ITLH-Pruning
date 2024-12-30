import matplotlib.pyplot as plt
import json
import numpy as np
import os
from matplotlib.transforms import Bbox

# Configurações do plot com estilo mais limpo
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (12, 6),
    'axes.grid': True,
    'grid.alpha': 0.2
})

def get_non_overlapping_position(ax, x, y, existing_boxes, box_height=30, box_width=80):
    """
    Encontra uma posição não sobreposta para a caixa de anotação
    """
    positions = [(10, 10), (10, -20), (-80, 10), (-80, -20)]  # Possíveis posições relativas
    text_box = None
    
    for dx, dy in positions:
        # Calcula as coordenadas da caixa
        display_coords = ax.transData.transform((x, y))
        text_coords = (display_coords[0] + dx, display_coords[1] + dy)
        box = Bbox([[text_coords[0], text_coords[1]], 
                   [text_coords[0] + box_width, text_coords[1] + box_height]])
        
        # Verifica sobreposição com caixas existentes
        overlap = False
        for existing_box in existing_boxes:
            if existing_box.overlaps(box):
                overlap = True
                break
                
        if not overlap:
            text_box = box
            return (dx, dy), text_box
            
    # Se todas as posições estiverem ocupadas, retorna a última tentativa
    return positions[0], box

def main():
    # Definir o caminho do arquivo
    file_path = './pruning/train_log.jsonl'
    #file_path = './no_pruning/train_log.jsonl'

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
            
            if entry['pruning_info'] is not None:
                pruning_epochs.append(entry['epoch'])
                pruning_events.append(entry['pruning_info'])

    # Criar o plot
    fig, ax = plt.subplots(dpi=100)

    # Plot principal do ângulo
    ax.plot(epochs, angles, color='#1f77b4', linewidth=1.5, label='Angle')

    # Rastrear caixas de anotação existentes
    existing_boxes = []

    # Adicionar marcadores para eventos de poda
    for epoch, event in zip(pruning_epochs, pruning_events):
        # Extrair o tipo de poda
        pruning_type = event.split('structure=')[1].split(',')[0] if 'structure=' in event else 'unknown'
        angle_value = angles[epochs.index(epoch)]
        
        # Adicionar marcador menor
        ax.scatter(epoch, angle_value, color='red', s=40, zorder=5, alpha=0.7)
        
        # Encontrar posição não sobreposta para a anotação
        (dx, dy), text_box = get_non_overlapping_position(ax, epoch, angle_value, existing_boxes)
        
        # Adicionar anotação com estilo mais limpo
        ax.annotate(f'Pruning: {pruning_type}',
                   xy=(epoch, angle_value),
                   xytext=(dx, dy),
                   textcoords='offset points',
                   ha='left' if dx > 0 else 'right',
                   va='bottom' if dy > 0 else 'top',
                   bbox=dict(boxstyle='round,pad=0.3', 
                           fc='#fff7d1',  # Amarelo mais suave
                           ec='#666666',   # Borda mais escura
                           alpha=0.7,
                           linewidth=0.5),
                   arrowprops=dict(arrowstyle='->', 
                                 connectionstyle='arc3,rad=0.2',
                                 color='#666666',
                                 linewidth=0.5),
                   fontsize=8)
        
        existing_boxes.append(text_box)

    # Configurar o plot
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Angle (radians)')
    ax.set_title('Angle Evolution with Pruning Events')
    ax.grid(True, linestyle='--', alpha=0.2)
    ax.legend(loc='upper left')

    # Ajustar o layout
    plt.tight_layout()

    # Salvar a figura
    plt.savefig('angle_plot.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    main()