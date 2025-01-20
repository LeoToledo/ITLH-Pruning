import matplotlib.pyplot as plt
import json
import numpy as np
import os
from matplotlib.transforms import Bbox
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib.units import inch

# Configuração para estilo mais adequado para publicação
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.figsize': (8, 6),
    'axes.grid': True,
    'grid.alpha': 0.3
})

def get_non_overlapping_position(ax, x, y, existing_boxes, box_height=25, box_width=70):
    positions = [
        (10, 15), (10, -25),  # direita
        (-70, 15), (-70, -25),  # esquerda
        (10, 40), (-70, 40),   # mais acima
        (10, -50), (-70, -50)  # mais abaixo
    ]
    text_box = None
    
    for dx, dy in positions:
        display_coords = ax.transData.transform((x, y))
        text_coords = (display_coords[0] + dx, display_coords[1] + dy)
        box = Bbox([[text_coords[0], text_coords[1]], 
                   [text_coords[0] + box_width, text_coords[1] + box_height]])
        
        overlap = False
        for existing_box in existing_boxes:
            if existing_box.overlaps(box):
                overlap = True
                break
                
        if not overlap:
            text_box = box
            return (dx, dy), text_box
            
    return positions[0], box

def format_flops(x, p):
    return f'{x/1e6:.1f}M'

def create_accuracy_flops_table(data_points, output_dir):
    """
    Cria uma tabela PDF com os dados de acurácia máxima vs redução de FLOPs
    """
    output_path = os.path.join(output_dir, "accuracy_vs_flops.pdf")
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    elements = []

    table_data = [['FLOPs (M)', 'Reduction (%)', 'Max Val Acc (%)']]
    
    for flops, reduction, acc in data_points:
        table_data.append([
            f'{flops/1e6:.1f}',
            f'{reduction:.1f}',
            f'{acc*100:.2f}'
        ])

    t = Table(table_data)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
    ]))
    
    elements.append(t)
    doc.build(elements)

def create_flops_plot(epochs, flops, pruning_epochs, pruning_flops, pruning_events, 
                     initial_flops, output_dir, existing_boxes):
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    
    plt.plot(epochs, flops, color='#2ca02c', linewidth=1.5, label='FLOPs')

    for epoch, flop_value, event in zip(pruning_epochs, pruning_flops, pruning_events):
        plt.scatter(epoch, flop_value, color='red', s=30, zorder=5, alpha=0.7)
        
        reduction = (initial_flops - flop_value) / initial_flops * 100
        annotation_text = f'-{reduction:.1f}%'

        (dx, dy), text_box = get_non_overlapping_position(ax, epoch, flop_value, existing_boxes)
        
        plt.annotate(annotation_text,
                   xy=(epoch, flop_value),
                   xytext=(dx, dy),
                   textcoords='offset points',
                   ha='left' if dx > 0 else 'right',
                   va='bottom' if dy > 0 else 'top',
                   bbox=dict(boxstyle='round,pad=0.2', 
                           fc='#fff7d1',
                           ec='#666666',
                           alpha=0.7,
                           linewidth=0.5),
                   arrowprops=dict(arrowstyle='->', 
                                 connectionstyle='arc3,rad=0.2',
                                 color='#666666',
                                 linewidth=0.5),
                   fontsize=8)
        
        existing_boxes.append(text_box)

    plt.xlabel('Epoch')
    plt.ylabel('FLOPs')
    plt.title('FLOPs Evolution with Pruning Events')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='upper right')
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_flops))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'flops_plot.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=600)
    plt.close()

def main():
    file_path = './pruning/train_log.jsonl'
    #file_path = './no_pruning/train_log.jsonl'

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"O arquivo {file_path} não foi encontrado!")

    output_dir = os.path.dirname(file_path)
    print(f"Usando arquivo: {file_path}")
    print(f"Salvando plots em: {output_dir}")

    epochs = []
    flops = []
    val_accs = []
    pruning_events = []
    pruning_epochs = []
    pruning_flops = []
    prev_flops = None
    initial_flops = None
    stage_data = []
    
    # Variáveis para rastrear o período atual
    current_period_start = 0
    current_max_val_acc = 0
    
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
        
    for i, entry in enumerate(data):
        epochs.append(entry['epoch'])
        current_flops = entry['flops']
        val_accs.append(entry['val_acc'])
        flops.append(current_flops)
        
        if initial_flops is None:
            initial_flops = current_flops
            
        # Atualizar máximo do período atual
        current_max_val_acc = max(current_max_val_acc, entry['val_acc'])
            
        # Detectar mudança nos FLOPs ou último epoch
        if (prev_flops is not None and current_flops != prev_flops) or i == len(data) - 1:
            if i == len(data) - 1 and current_flops == prev_flops:
                # Incluir o último ponto na máxima do período atual
                current_max_val_acc = max(current_max_val_acc, entry['val_acc'])
                
            pruning_epochs.append(entry['epoch'])
            pruning_flops.append(current_flops)
            
            # Calcular redução em relação ao inicial
            reduction = (initial_flops - (prev_flops or current_flops)) / initial_flops * 100
            
            # Adicionar dados do período que está terminando
            stage_data.append((prev_flops or current_flops, reduction, current_max_val_acc))
            
            # Resetar para o próximo período
            current_period_start = i
            current_max_val_acc = entry['val_acc']
            
            if entry['pruning_info'] is not None:
                pruning_type = entry['pruning_info'].split('structure=')[1].split(',')[0] if 'structure=' in entry['pruning_info'] else 'unknown'
                pruning_events.append(f"Pruning: {pruning_type}")
            else:
                pruning_events.append("FLOPs changed")
                    
        prev_flops = current_flops

    # Criar plots e tabela
    existing_boxes = []
    create_flops_plot(epochs, flops, pruning_epochs, pruning_flops, pruning_events, 
                     initial_flops, output_dir, existing_boxes)
    create_accuracy_flops_table(stage_data, output_dir)

if __name__ == "__main__":
    main()