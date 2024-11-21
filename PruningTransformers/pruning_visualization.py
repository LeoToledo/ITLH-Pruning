# pruning_visualization.py

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import threading
import queue

class PruningVisualization:
    def __init__(self, master):
        self.master = master
        self.master.title("Pruning Visualization")
        self.master.geometry("800x600")

        self.fig, self.axs = plt.subplots(2, 2, figsize=(10, 8))
        self.fig.tight_layout(pad=3.0)

        self.accuracy_line, = self.axs[0, 0].plot([], [], 'r-')
        self.axs[0, 0].set_title("Accuracy vs Iteration")
        self.axs[0, 0].set_xlabel("Iteration")
        self.axs[0, 0].set_ylabel("Accuracy")

        self.complexity_line, = self.axs[0, 1].plot([], [], 'b-')
        self.axs[0, 1].set_title("Complexity vs Iteration")
        self.axs[0, 1].set_xlabel("Iteration")
        self.axs[0, 1].set_ylabel("Complexity")

        self.similarity_line, = self.axs[1, 0].plot([], [], 'g-')
        self.axs[1, 0].set_title("Similarity vs Iteration")
        self.axs[1, 0].set_xlabel("Iteration")
        self.axs[1, 0].set_ylabel("Similarity")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.log_frame = ttk.Frame(self.master)
        self.log_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.log_text = tk.Text(self.log_frame, height=10)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        self.scrollbar = ttk.Scrollbar(self.log_frame, orient="vertical", command=self.log_text.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.log_text.configure(yscrollcommand=self.scrollbar.set)

        self.data_queue = queue.Queue()
        self.update_thread = threading.Thread(target=self.update_plots)
        self.update_thread.daemon = True
        self.update_thread.start()

    def update_plots(self):
        while True:
            try:
                data = self.data_queue.get(timeout=0.1)
                self.accuracy_line.set_data(data['iterations'], data['accuracies'])
                self.complexity_line.set_data(data['iterations'], data['complexities'])
                self.similarity_line.set_data(data['iterations'], data['similarities'])

                for ax in self.axs.flat:
                    ax.relim()
                    ax.autoscale_view()

                self.canvas.draw()

                self.log_text.insert(tk.END, f"Iteration {data['iterations'][-1]}: "
                                             f"Accuracy: {data['accuracies'][-1]:.4f}, "
                                             f"Complexity: {data['complexities'][-1]:.4f}, "
                                             f"Similarity: {data['similarities'][-1]:.4f}\n")
                self.log_text.see(tk.END)
            except queue.Empty:
                pass

    def update_data(self, iteration, accuracy, complexity, similarity):
        self.data_queue.put({
            'iterations': list(range(iteration + 1)),
            'accuracies': [accuracy],
            'complexities': [complexity],
            'similarities': [similarity]
        })

def main():
    root = tk.Tk()
    app = PruningVisualization(root)

    # Simulação de dados para teste
    import time
    import random

    def simulate_pruning():
        for i in range(100):
            accuracy = random.uniform(0.7, 1.0)
            complexity = random.uniform(10000, 100000)
            similarity = random.uniform(0.5, 1.0)
            app.update_data(i, accuracy, complexity, similarity)
            time.sleep(0.5)

    threading.Thread(target=simulate_pruning).start()

    root.mainloop()

if __name__ == "__main__":
    main()