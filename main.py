import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np
from utils.plots import generate_plot
from utils.preprocess import preprocess_input
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import logging
# pip install tensorflow
import tensorflow.keras as keras
import sklearn
import xgboost
import tensorflow as tf
print(tf.__version__)  # This should print the installed version of TensorFlow


# Load Models
MODELS = {
    "Logistic Regression": "models/logistic_regression_model.pkl",
    "Neural Network": "models/NN_model.pkl",
    "Random Forest": "models/random_forest_model.pkl",
    "SVM": "models/svm_model.pkl",
    "XGBoost": "models/XGBoost_model.pkl",
    "Decision Tree":"models/decision_tree_model.pkl",
    "Adaboost":"models/Adaboost_model.pkl",
}

class AIModelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Meat Quality Prediction")
        self.root.geometry("1280x1080")
        self.label_mapping = {
            0: "Fresh",
            1: "Spoiled"
        }
        # Input Section
        input_frame = ttk.LabelFrame(root, text="Inputs", padding=10)
        input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.feature_names = ['TVC', 'MQ135', 'MQ2', 'MQ6', 'MQ7', 'MQ9', 'Humidity', 'Temperature']
        self.inputs = {}
        for i, feature in enumerate(self.feature_names):
            ttk.Label(input_frame, text=feature).grid(row=i, column=0, pady=5, sticky="w")
            self.inputs[feature] = ttk.Entry(input_frame, width=20)
            self.inputs[feature].grid(row=i, column=1, pady=5)

        # Model Selection
        model_frame = ttk.LabelFrame(root, text="Model Selector", padding=10)
        model_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        ttk.Label(model_frame, text="Select Model").grid(row=0, column=0, pady=5)
        self.model_choice = ttk.Combobox(model_frame, values=list(MODELS.keys()), state="readonly")
        self.model_choice.grid(row=1, column=0, pady=5)
        self.model_choice.set("Select a Model")

        predict_button = ttk.Button(model_frame, text="Predict", command=self.predict)
        predict_button.grid(row=2, column=0, pady=10)

        # Result Section
        result_frame = ttk.LabelFrame(root, text="Result", padding=10, relief="raised", borderwidth=2)
        result_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # Create a stylish label with background color, border, and font
        self.result_label = tk.Text(result_frame, height=5, wrap=tk.WORD, state="disabled", font=("Times New Roman", 17, "bold"), foreground="red", background="white")
        self.result_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Log Section with Scrollable Canvas
        log_frame = ttk.LabelFrame(root, text="Logs", padding=10)
        log_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

        self.log_canvas = tk.Canvas(log_frame, width=680, height=480, bg="light gray")
        self.log_canvas.pack(fill=tk.BOTH, expand=True)

        # Menu Bar
        self.create_menu()

    def create_menu(self):
        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)

        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="About", command=lambda: messagebox.showinfo("About", "Meat Quality Prediction Prototype v1.0 \n Made By: \n Sadman and Hasin \n"))
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)

    def predict(self):
        try:
        # Validate model selection
            model_name = self.model_choice.get()
            if model_name not in MODELS:
                raise ValueError("Please select a valid model from the dropdown menu.")

        # Validate inputs
            input_data = []
            for feature in self.feature_names:
                value = self.inputs[feature].get()
                if not value:
                    raise ValueError(f"Input for {feature} is missing.")
                input_data.append(float(value))

        # Check input length
            if len(input_data) != len(self.feature_names):
                raise ValueError(f"Expected {len(self.feature_names)} features, got {len(input_data)}.")

        # Load scaler and preprocess input
            scaler = joblib.load("models/scaler.pkl")
            scaled_input = scaler.transform([input_data])  # Shape should be (1, 8)

        # Load model and predict
            model_path = MODELS[model_name]
            model = joblib.load(model_path)
            prediction = model.predict(scaled_input)[0]

            if isinstance(prediction, (list, np.ndarray)):
                prediction = prediction[0]  # Get the first element if it's a list or ndarray

            if self.model_choice.get() in ["Neural Network", "XGBoost"]:
                label = self.label_mapping.get(prediction, "Unknown")
            else:
                # If the model is not NN or XGBoost, assume it's a string label already
                label = prediction  # The label is already in human-readable format

        # Display prediction result
            self.result_label.config(state="normal")
            self.result_label.delete("1.0", tk.END)
            self.result_label.insert(tk.END, f"Prediction: {label}")
            self.result_label.config(state="disabled")

        # Optional: Probability Prediction (if the model supports it)
            # if hasattr(model, "predict_proba"):
            #     probabilities = model.predict_proba(scaled_input)[0]
            #     self.result_label.config(state="normal")
            #     self.result_label.insert(tk.END, f"Probabilities: {probabilities}\n")
            #     self.result_label.config(state="disabled")
            
            generate_plot(input_data, prediction)
            self.log_canvas.delete("all")
            plot_image = tk.PhotoImage(file="assets/plot.png")
            self.log_canvas.create_image(0, 0, anchor="nw", image=plot_image)
            self.root.plot_image = plot_image  # Keep reference to avoid garbage collection

        except ValueError as e:
            # Suppress the error and log it for debugging (without popup)
            if "shape mismatch" in str(e):
                logging.warning("Shape mismatch detected: Input data does not match expected shape.")
            else:
                logging.error(f"Unexpected value error: {e}")
        except Exception as e:
            # Catch other errors and log them (without popups)
            logging.error(f"An error occurred: {e}")
            self.result_label.config(state="normal")
            self.result_label.delete(1.0, tk.END)
            self.result_label.insert(tk.END, "Error Occurred")
            self.result_label.config(state="disabled")


    def display_plot(self, fig):
        for widget in self.log_canvas.winfo_children():
            widget.destroy()  # Clear existing plot

        canvas = FigureCanvasTkAgg(fig, master=self.log_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = AIModelApp(root)
    root.mainloop()



# import tkinter as tk
# from tkinter import ttk, messagebox
# import pandas as pd
# import joblib
# from utils.plots import generate_plot
# from utils.preprocess import preprocess_input
# # from tensorflow.keras.models import load_model



# # Load Models
# MODELS = {
#     "Logistic Regression": "models/logistic_regression_model.pkl",
#     "Neural Network": "models/NN_model.pkl",
#     "Random Forest": "models/random_forest_model.pkl",
#     "SVM": "models/svm_model.pkl",
#     "XGBoost": "models/XGBoost_model.pkl",
# }

# class AIModelApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Meat Quality Prediction")
#         self.root.geometry("1280x1080")

#         # Input Section
#         input_frame = ttk.LabelFrame(root, text="Inputs", padding=10)
#         input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

#         feature_names = ['TVC','MQ135',  'MQ2', 'MQ6', 'MQ7', 'MQ9', 'Humidity','Temperature']
#         self.inputs = {}
#         for i, feature in enumerate(feature_names):
#             ttk.Label(input_frame, text=feature).grid(row=i, column=0, pady=5, sticky="w")
#             self.inputs[feature] = ttk.Entry(input_frame, width=20)
#             self.inputs[feature].grid(row=i, column=1, pady=5)

#         # Model Selection
#         model_frame = ttk.LabelFrame(root, text="Model Selector", padding=10)
#         model_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

#         ttk.Label(model_frame, text="Select Model").grid(row=0, column=0, pady=5)
#         self.model_choice = ttk.Combobox(model_frame, values=list(MODELS.keys()), state="readonly")
#         self.model_choice.grid(row=1, column=0, pady=5)
#         self.model_choice.set("Select a Model")

#         predict_button = ttk.Button(model_frame, text="Predict", command=self.predict)
#         predict_button.grid(row=2, column=0, pady=10)

#         # Result Section
#         result_frame = ttk.LabelFrame(root, text="Result", padding=10)
#         result_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

#         self.result_label = tk.Text(result_frame, height=5, wrap=tk.WORD, state="disabled")
#         self.result_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

#         # Log Section
#         log_frame = ttk.LabelFrame(root, text="Logs", padding=10)
#         log_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

#         self.log_canvas = tk.Canvas(log_frame, width=300, height=200)
#         self.log_canvas.pack(fill=tk.BOTH, expand=True)

#         # Menu Bar
#         self.create_menu()

#     def create_menu(self):
#         menu_bar = tk.Menu(self.root)
#         self.root.config(menu=menu_bar)

#         file_menu = tk.Menu(menu_bar, tearoff=0)
#         file_menu.add_command(label="About", command=lambda: messagebox.showinfo("About", "Meat Quality Prediction Prototype v1.0"))
#         file_menu.add_separator()
#         file_menu.add_command(label="Exit", command=self.root.quit)
#         menu_bar.add_cascade(label="File", menu=file_menu)

#     def predict(self):
#         try:
#             feature_names = ['TVC','MQ135',  'MQ2', 'MQ6', 'MQ7', 'MQ9', 'Humidity','Temperature']

#         # Get inputs from the UI for the specific features
#             input_data = [float(self.inputs[feature].get()) for feature in feature_names]
        
#         # Load the scaler (saved during training)
#             scaler = joblib.load("models/scaler.pkl")

#         # Scale the input data
#             scaled_input = scaler.transform([input_data])  # Ensure 2D shape for scaler

#         # Predict using the selected model
#             # # Load selected model
#             model_path = MODELS[self.model_choice.get()]
#             model = joblib.load(model_path)

#             prediction = model.predict(scaled_input)  # Make prediction
#             label = prediction[0]  # Get the label (e.g., "Excellent", "Good", etc.)

#         # Display the result in the Result Part
#             self.result_label.config(text=f"Predicted Label: {label}")
        
#         # Log the result (e.g., plot probabilities or other metrics)
#         # # Update result text
#             self.result_label.config(state="normal")
#             self.result_label.delete("1.0", tk.END)
#             self.result_label.insert(tk.END, f"Prediction: {prediction}")
#             self.result_label.config(state="disabled")

#             # # Generate and display plot
#             generate_plot(input_data, prediction)
#             self.log_canvas.delete("all")
#             plot_image = tk.PhotoImage(file="assets/plot.png")
#             self.log_canvas.create_image(0, 0, anchor="nw", image=plot_image)
#             self.root.plot_image = plot_image  # Keep reference to avoid garbage collection
            
#             # # Get inputs
#             # input_data = [float(self.inputs[f"Sensor{i}"].get()) for i in range(1, 9)]
#             # preprocessed_data = preprocess_input(input_data)

#             # # Load selected model
#             # model_path = MODELS[self.model_choice.get()]
#             # model = joblib.load(model_path)

#             # # Make prediction
#             # prediction = model.predict([preprocessed_data])[0]

#             # # Update result text
#             # self.result_label.config(state="normal")
#             # self.result_label.delete("1.0", tk.END)
#             # self.result_label.insert(tk.END, f"Prediction: {prediction}")
#             # self.result_label.config(state="disabled")

#             # # Generate and display plot
#             # generate_plot(input_data, prediction)
#             # self.log_canvas.delete("all")
#             # plot_image = tk.PhotoImage(file="assets/plot.png")
#             # self.log_canvas.create_image(0, 0, anchor="nw", image=plot_image)
#             # self.root.plot_image = plot_image  # Keep reference to avoid garbage collection

#         except Exception as e:
#             messagebox.showerror("Error", f"An error occurred: {e}")


# if __name__ == "__main__":
#     root = tk.Tk()
#     app = AIModelApp(root)
#     root.mainloop()
