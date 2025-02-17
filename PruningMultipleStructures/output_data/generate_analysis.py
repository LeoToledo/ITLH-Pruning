#!/usr/bin/env python3
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

# Imports for PDF generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet

# ============================
# Global Path Variables
# ============================
# Path to the input train_log.jsonl file
INPUT_FILE = './final_results/20250210-lr-drop-two-stages/train_log.jsonl'
# Directory where plots and the PDF will be saved
OUTPUT_DIR = './final_results/20250210-lr-drop-two-stages/'

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================
# Global Variables for Analysis
# ============================
sota_acc = 0.923  # Defined state-of-the-art accuracy

# Define a paper style for all graphs (avoid using seaborn)
PAPER_STYLE = {
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
}

# ======================================================
# Code 1: Generation of Accuracy Plots (Training/Validation)
# ======================================================

def detect_pruning_points(data):
    """Detect pruning points by comparing the reduction in FLOPs between epochs."""
    pruning_epochs = []
    prev_flops = None
    for i, entry in enumerate(data):
        if prev_flops is not None and entry['flops'] < prev_flops:
            pruning_epochs.append(i)
        prev_flops = entry['flops']
    return pruning_epochs

def create_accuracy_plot(epochs, acc_data, title, ylabel, filename, pruning_epochs):
    """Creates and saves an accuracy plot with vertical lines at pruning events."""
    with plt.rc_context(PAPER_STYLE):
        plt.figure()
        ax = plt.gca()
        plt.plot(epochs, acc_data, linewidth=1.5, label=ylabel)
        for ep in pruning_epochs:
            plt.axvline(x=epochs[ep], color='#d62728', linestyle='--', alpha=0.5, linewidth=1)
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.title(title)
        min_acc = min(acc_data)
        y_min = max(0, min_acc - 0.1)
        plt.ylim(y_min, 1.02)
        y_ticks = np.concatenate([
            np.arange(round(y_min, 1), 0.8, 0.1),
            np.arange(0.8, 1.02, 0.02)
        ])
        plt.yticks(y_ticks)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(output_path, bbox_inches='tight', dpi=600)
        plt.close()

def generate_accuracy_plots(data):
    """Processes data and generates Training and Validation Accuracy plots."""
    epochs = []
    train_acc = []
    val_acc = []
    for entry in data:
        epochs.append(entry['epoch'])
        train_acc.append(entry['train_acc'])
        val_acc.append(entry['val_acc'])
    pruning_epochs = detect_pruning_points(data)
    create_accuracy_plot(epochs, train_acc, 'Training Accuracy vs. Epoch', 'Training Accuracy', 'training_accuracy.png', pruning_epochs)
    create_accuracy_plot(epochs, val_acc, 'Validation Accuracy vs. Epoch', 'Validation Accuracy', 'validation_accuracy.png', pruning_epochs)

# ======================================================
# Code 2: Angle Evolution Plot with Pruning Events
# ======================================================

def get_non_overlapping_position_angle(ax, x, y, existing_boxes, box_height=30, box_width=80):
    """
    Determines a non-overlapping position for an annotation (used in the angle plot).
    """
    positions = [(10, 10), (10, -20), (-80, 10), (-80, -20)]
    for dx, dy in positions:
        display_coords = ax.transData.transform((x, y))
        text_coords = (display_coords[0] + dx, display_coords[1] + dy)
        box = Bbox([[text_coords[0], text_coords[1]],
                    [text_coords[0] + box_width, text_coords[1] + box_height]])
        overlap = any(existing_box.overlaps(box) for existing_box in existing_boxes)
        if not overlap:
            return (dx, dy), box
    return positions[0], box

def generate_angle_plot(data):
    """Generates the plot of angle evolution with annotations for pruning events."""
    # Use our defined PAPER_STYLE (do not use seaborn)
    with plt.rc_context(PAPER_STYLE):
        epochs = []
        angles = []
        pruning_epochs = []
        pruning_events = []
        for entry in data:
            epochs.append(entry['epoch'])
            angles.append(entry['angle'])
            if entry.get('pruning_info') is not None:
                pruning_epochs.append(entry['epoch'])
                pruning_events.append(entry['pruning_info'])
        fig, ax = plt.subplots(dpi=100)
        ax.plot(epochs, angles, color='#1f77b4', linewidth=1.5, label='Angle')
        existing_boxes = []
        for epoch, event in zip(pruning_epochs, pruning_events):
            try:
                idx = epochs.index(epoch)
            except ValueError:
                idx = 0
            angle_value = angles[idx]
            ax.scatter(epoch, angle_value, color='red', s=40, zorder=5, alpha=0.7)
            (dx, dy), text_box = get_non_overlapping_position_angle(ax, epoch, angle_value, existing_boxes)
            ax.annotate(f'Pruning: {event.split("structure=")[1].split(",")[0] if "structure=" in event else event}',
                        xy=(epoch, angle_value),
                        xytext=(dx, dy),
                        textcoords='offset points',
                        ha='left' if dx > 0 else 'right',
                        va='bottom' if dy > 0 else 'top',
                        bbox=dict(boxstyle='round,pad=0.3',
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
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Angle (degrees)')
        ax.set_title('Angle Evolution with Pruning Events')
        ax.grid(True, linestyle='--', alpha=0.2)
        ax.legend(loc='upper left')
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, 'angle_plot.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

# ======================================================
# Code 2.1: New Derivative Plot (Derivative vs. Epoch)
# ======================================================

def detect_events(data):
    """
    Detects LR drop and pruning events from the JSON log.
    Returns two lists: lr_drop_epochs and prune_epochs.
    """
    lr_drop_epochs = []
    prune_epochs = []
    prev_lr_drop = False
    prev_prune = False
    for entry in data:
        info = entry.get('pruning_info')
        if info is not None:
            if "Waiting for" in info:
                if not prev_lr_drop:
                    lr_drop_epochs.append(entry['epoch'])
                    prev_lr_drop = True
            else:
                prev_lr_drop = False
            if "Pruned with structure=" in info:
                if not prev_prune:
                    prune_epochs.append(entry['epoch'])
                    prev_prune = True
            else:
                prev_prune = False
        else:
            prev_lr_drop = False
            prev_prune = False
    return lr_drop_epochs, prune_epochs

def create_derivative_plot(epochs, derivatives, title, filename, lr_drop_epochs, prune_epochs):
    """Creates and saves a plot of derivative vs. epoch with vertical lines at LR drop and prune events."""
    with plt.rc_context(PAPER_STYLE):
        plt.figure()
        ax = plt.gca()
        plt.plot(epochs, derivatives, linewidth=1.5, label='Derivative')
        # Mark LR drop events with blue dashed vertical lines
        for lr_epoch in lr_drop_epochs:
            plt.axvline(x=lr_epoch, color='blue', linestyle='--', alpha=0.7, linewidth=1, label='LR Drop')
        # Mark Prune events with red dashed vertical lines
        for pr_epoch in prune_epochs:
            plt.axvline(x=pr_epoch, color='red', linestyle='--', alpha=0.7, linewidth=1, label='Prune')
        # To avoid duplicate legend entries, use unique labels.
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.xlabel('Epoch')
        plt.ylabel('Derivative (° change)')
        plt.title(title)
        plt.legend(by_label.values(), by_label.keys())
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(output_path, bbox_inches='tight', dpi=600)
        plt.close()

def generate_derivative_plot(data):
    """Generates the derivative plot using the new JSON log fields."""
    epochs = []
    derivatives = []
    for entry in data:
        epochs.append(entry['epoch'])
        # Use the logged derivative value (or 0 for the first epoch)
        derivatives.append(entry.get('derivative', 0))
    lr_drop_epochs, prune_epochs = detect_events(data)
    create_derivative_plot(epochs, derivatives, 'Derivative Evolution vs. Epoch', 'derivative_plot.png', lr_drop_epochs, prune_epochs)

# ======================================================
# Code 3: FLOPs Plot and Accuracy vs. FLOPs Table (PDF)
# ======================================================

def get_non_overlapping_position_flops(ax, x, y, existing_boxes, box_height=25, box_width=70):
    """
    Determines a non-overlapping position for an annotation (used in the FLOPs plot).
    """
    positions = [
        (10, 15), (10, -25),
        (-70, 15), (-70, -25),
        (10, 40), (-70, 40),
        (10, -50), (-70, -50)
    ]
    for dx, dy in positions:
        display_coords = ax.transData.transform((x, y))
        text_coords = (display_coords[0] + dx, display_coords[1] + dy)
        box = Bbox([[text_coords[0], text_coords[1]],
                    [text_coords[0] + box_width, text_coords[1] + box_height]])
        overlap = any(existing_box.overlaps(box) for existing_box in existing_boxes)
        if not overlap:
            return (dx, dy), box
    return positions[0], box

def format_flops(x, p):
    """Formats the y-axis of FLOPs in millions (M)."""
    return f'{x/1e6:.1f}M'

def create_accuracy_flops_table(data_points):
    """
    Creates a PDF table relating FLOPs, reduction, maximum validation accuracy,
    final validation accuracy, and the delta versus SOTA (state-of-the-art).
    """
    output_path = os.path.join(OUTPUT_DIR, "accuracy_vs_flops.pdf")
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    elements = []
    # New header includes an extra column for delta versus SOTA.
    table_data = [['FLOPs (M)', 'Reduction (%)', 'Max Val Acc (%)', 'Final Val Acc (%)', 'Δ SOTA (%)']]
    for flops_val, reduction, max_acc, final_acc in data_points:
        delta = (final_acc - sota_acc) * 100  # delta in percentage
        table_data.append([
            f'{flops_val/1e6:.1f}',
            f'{reduction:.1f}',
            f'{max_acc*100:.2f}',
            f'{final_acc*100:.2f}',
            f'{delta:+.2f}%'
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
    return table_data  # Return the table data for inclusion in the final report

def generate_flops_plot_and_table(data):
    """Processes data and generates the FLOPs plot and the Accuracy vs. FLOPs PDF table."""
    epochs = []
    flops = []
    val_accs = []
    pruning_events = []
    pruning_epochs = []
    pruning_flops = []
    prev_flops = None
    initial_flops = None
    stage_data = []
    current_max_val_acc = 0
    for i, entry in enumerate(data):
        epoch = entry['epoch']
        current_flops = entry['flops']
        epochs.append(epoch)
        flops.append(current_flops)
        val_accs.append(entry['val_acc'])
        if initial_flops is None:
            initial_flops = current_flops
        current_max_val_acc = max(current_max_val_acc, entry['val_acc'])
        # Detect change in FLOPs or if this is the last point
        if (prev_flops is not None and current_flops != prev_flops) or i == len(data) - 1:
            if i == len(data) - 1 and current_flops == prev_flops:
                current_max_val_acc = max(current_max_val_acc, entry['val_acc'])
            pruning_epochs.append(epoch)
            pruning_flops.append(current_flops)
            reduction = (initial_flops - (prev_flops if prev_flops is not None else current_flops)) / initial_flops * 100
            final_val = entry['val_acc']  # final validation accuracy for this stage
            stage_data.append((prev_flops if prev_flops is not None else current_flops, reduction, current_max_val_acc, final_val))
            current_max_val_acc = entry['val_acc']
            if entry.get('pruning_info') is not None:
                if 'structure=' in entry['pruning_info']:
                    pruning_type = entry['pruning_info'].split('structure=')[1].split(',')[0]
                else:
                    pruning_type = 'unknown'
                pruning_events.append(f"Pruning: {pruning_type}")
            else:
                pruning_events.append("FLOPs changed")
        prev_flops = current_flops
    existing_boxes = []
    create_flops_plot(epochs, flops, pruning_epochs, pruning_flops, pruning_events, initial_flops, existing_boxes)
    create_accuracy_flops_table(stage_data)
    return stage_data

def create_flops_plot(epochs, flops, pruning_epochs, pruning_flops, pruning_events, initial_flops, existing_boxes):
    """Creates and saves the FLOPs evolution plot with annotations."""
    with plt.rc_context(PAPER_STYLE):
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        plt.plot(epochs, flops, color='#2ca02c', linewidth=1.5, label='FLOPs')
        for epoch, flop_value, event in zip(pruning_epochs, pruning_flops, pruning_events):
            plt.scatter(epoch, flop_value, color='red', s=30, zorder=5, alpha=0.7)
            reduction = (initial_flops - flop_value) / initial_flops * 100
            annotation_text = f'-{reduction:.1f}%'
            (dx, dy), text_box = get_non_overlapping_position_flops(ax, epoch, flop_value, existing_boxes)
            plt.annotate(annotation_text,
                         xy=(epoch, flop_value),
                         xytext=(dx, dy),
                         textcoords='offset points',
                         ha='left' if dx > 0 else 'right',
                         va='bottom' if dy > 0 else 'top',
                         bbox=dict(boxstyle='round,pad=0.2', fc='#fff7d1', ec='#666666', alpha=0.7, linewidth=0.5),
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='#666666', linewidth=0.5),
                         fontsize=8)
            existing_boxes.append(text_box)
        plt.xlabel('Epoch')
        plt.ylabel('FLOPs')
        plt.title('FLOPs Evolution with Pruning Events')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend(loc='upper right')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_flops))
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, 'flops_plot.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=600)
        plt.close()

# ======================================================
# Code 4: Final Report PDF
# ======================================================

def generate_final_report_pdf(header, stage_data):
    """
    Generates a final PDF report that displays the hyperparameters,
    the generated plots, and the Accuracy vs. FLOPs table.
    """
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(os.path.join(OUTPUT_DIR, "final_report.pdf"), pagesize=letter)
    elements = []
    
    # Title
    elements.append(Paragraph("Analysis Report", styles['Title']))
    elements.append(Spacer(1, 0.2*inch))
    
    # Hyperparameters Section
    elements.append(Paragraph("Hyperparameters", styles['Heading2']))
    hyper_table_data = [["Parameter", "Value"]]
    for key, value in header.items():
         hyper_table_data.append([str(key), str(value)])
    hyper_table = Table(hyper_table_data, hAlign='CENTER')
    hyper_table.setStyle(TableStyle([
         ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
         ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
         ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
         ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
         ('FONTSIZE', (0, 0), (-1, 0), 12),
         ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
         ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
         ('GRID', (0, 0), (-1, -1), 1, colors.black),
         ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
    ]))
    elements.append(hyper_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # SOTA Section
    elements.append(Paragraph(f"Defined SOTA Accuracy: {sota_acc*100:.2f}%", styles['Heading2']))
    elements.append(Spacer(1, 0.3*inch))
    
    # Images Section
    image_info = [
        ("Training Accuracy", "training_accuracy.png"),
        ("Validation Accuracy", "validation_accuracy.png"),
        ("Angle Evolution", "angle_plot.png"),
        ("Derivative Evolution", "derivative_plot.png"),
        ("FLOPs Evolution", "flops_plot.png")
    ]
    for title_text, img_file in image_info:
        elements.append(Paragraph(title_text, styles['Heading2']))
        elements.append(Spacer(1, 0.1*inch))
        img_path = os.path.join(OUTPUT_DIR, img_file)
        try:
            im = RLImage(img_path)
            im.drawWidth = 6*inch
            im.drawHeight = 4*inch
            elements.append(im)
            elements.append(Spacer(1, 0.3*inch))
        except Exception as e:
            elements.append(Paragraph(f"Error loading image {img_file}: {str(e)}", styles['Normal']))
            elements.append(Spacer(1, 0.3*inch))
    
    # Accuracy vs. FLOPs Table Section
    elements.append(Paragraph("Accuracy vs. FLOPs", styles['Heading2']))
    table_data = [['FLOPs (M)', 'Reduction (%)', 'Max Val Acc (%)', 'Final Val Acc (%)', 'Δ SOTA (%)']]
    for flops_val, reduction, max_acc, final_acc in stage_data:
        delta = (final_acc - sota_acc)*100
        table_data.append([
            f'{flops_val/1e6:.1f}',
            f'{reduction:.1f}',
            f'{max_acc*100:.2f}',
            f'{final_acc*100:.2f}',
            f'{delta:+.2f}%'
        ])
    acc_table = Table(table_data, hAlign='CENTER')
    acc_table.setStyle(TableStyle([
         ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
         ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
         ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
         ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
         ('FONTSIZE', (0, 0), (-1, 0), 12),
         ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
         ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
         ('GRID', (0, 0), (-1, -1), 1, colors.black),
         ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
    ]))
    elements.append(acc_table)
    
    doc.build(elements)

# ======================================================
# Main Function
# ======================================================

def main():
    # Load data from the input JSONL file
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")
    with open(INPUT_FILE, 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Separate the hyperparameters header from the rest of the data
    header = None
    data_entries = []
    for entry in data:
        if "hyperparameters" in entry:
            header = entry["hyperparameters"]
        else:
            data_entries.append(entry)
    
    # Generate accuracy plots (Training and Validation)
    generate_accuracy_plots(data_entries)
    # Generate angle evolution plot
    generate_angle_plot(data_entries)
    # Generate the derivative plot (Derivative vs. Epoch)
    generate_derivative_plot(data_entries)
    # Generate the FLOPs evolution plot and the Accuracy vs. FLOPs table PDF
    stage_data = generate_flops_plot_and_table(data_entries)
    
    # Generate the final comprehensive report PDF
    if header is not None:
        generate_final_report_pdf(header, stage_data)
    
    print("All plots, PDFs, and the final report have been generated in the directory:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
