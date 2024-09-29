import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from data_preprocessing import label_encoding, generate_heatmap, normalization, apply_sigmoid
from models import apply_svm, apply_decision_tree, apply_random_forest, naive_bayes
from utils import insert_data, check_inputs

class MyApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sentiment Analysis")
        self.geometry("800x600")
        self.setup_ui()

    def setup_ui(self):
        frame_row1 = tk.Frame(self)
        frame_row1.pack(side=tk.TOP, pady=10)

        self.btn_open = tk.Button(frame_row1, text="1. Label Encoding", command=label_encoding)
        self.btn_open.pack(side=tk.LEFT, padx=10)

        self.btn_open = tk.Button(frame_row1, text="2. Heatmap and Relevant Attributes CSV", command=generate_heatmap)
        self.btn_open.pack(side=tk.LEFT, padx=10)

        self.btn_open = tk.Button(frame_row1, text="3. Normalization", command=normalization)
        self.btn_open.pack(side=tk.LEFT, padx=10)

        self.btn_open = tk.Button(frame_row1, text="4. Sigmoid", command=apply_sigmoid)
        self.btn_open.pack(side=tk.LEFT, padx=10)

        frame_row2 = tk.Frame(self)
        frame_row2.pack(side=tk.TOP, pady=10)

        self.btn_open = tk.Button(frame_row2, text="5. SVM", command=lambda: apply_svm(self))
        self.btn_open.pack(side=tk.LEFT, padx=10)
        
        self.btn_open = tk.Button(frame_row2, text="6. Decision Tree", command=lambda: apply_decision_tree(self))
        self.btn_open.pack(side=tk.LEFT, padx=10)

        self.btn_open = tk.Button(frame_row2, text="7. Random Forest", command=lambda: apply_random_forest(self))
        self.btn_open.pack(side=tk.LEFT, padx=10)

        self.btn_open = tk.Button(frame_row2, text="8. Naive Bayes", command=lambda: naive_bayes(self))
        self.btn_open.pack(side=tk.LEFT, padx=10)

        frame_row3 = tk.Frame(self)
        frame_row3.pack(side=tk.TOP, pady=10)

        self.input_text = self.add_input_field(frame_row3, "Insert Text:")
        self.input_sentiment = self.add_input_field(frame_row3, "Sentiment:")
        self.input_hashtags = self.add_input_field(frame_row3, "Hashtags:")
        self.input_retweets = self.add_input_field(frame_row3, "Retweets:")
        self.input_likes = self.add_input_field(frame_row3, "Likes:")
        self.input_month = self.add_input_field(frame_row3, "Month:")
        self.input_hour = self.add_input_field(frame_row3, "Hour:")

        self.insert_button = tk.Button(frame_row3, text="Insert", command=lambda: insert_data(self))
        self.insert_button.pack(side=tk.LEFT, padx=10)

        self.check_button = tk.Button(frame_row3, text="Check", command=lambda: check_inputs(self))
        self.check_button.pack(side=tk.LEFT, padx=10)

        frame_row4 = tk.Frame(self)
        frame_row4.pack(side=tk.TOP, pady=10)

        self.fig, self.graph_ax = plt.subplots(figsize=(4, 2), dpi=100)
        self.canvas_widget = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_widget.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.canvas_widget.draw()

    def add_input_field(self, frame, label_text):
        label = tk.Label(frame, text=label_text)
        label.pack(side=tk.LEFT, padx=0)
        entry = tk.Entry(frame)
        entry.pack(side=tk.LEFT, padx=0)
        return entry

if __name__ == "__main__":
    app = MyApp()
    app.mainloop()
