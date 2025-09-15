import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.font import Font
from perceptron import Perceptron
from utils import cargar_dataset
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results/")
if not os.path.exists(RESULTS_PATH):
    os.makedirs(os.path.join(RESULTS_PATH, "graficas"))

class PerceptronApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Perceptrón Simple")
        self.root.geometry("600x650")
        self.root.resizable(False, False)
        self.root.configure(bg="#f0f0f0")

        self.dataset = None
        self.X = None
        self.y = None
        self.model = None

        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("TButton", padding=6, relief="flat", background="#007BFF", foreground="white", font=('Helvetica', 10, 'bold'))
        self.style.map("TButton", background=[('active', '#0056b3')])
        self.style.configure("TLabel", background="#f0f0f0", font=('Helvetica', 10))
        self.style.configure("Header.TLabel", font=('Helvetica', 12, 'bold'))
        self.style.configure("TEntry", padding=5, font=('Helvetica', 10))

        # Main frame
        main_frame = ttk.Frame(root, padding="10 10 10 10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Dataset Section ---
        dataset_frame = ttk.LabelFrame(main_frame, text="1. Cargar Datos", padding="10 10 10 10")
        dataset_frame.pack(fill=tk.X, pady=5)

        self.load_button = ttk.Button(dataset_frame, text="📂 Cargar Dataset", command=self.cargar)
        self.load_button.pack(pady=5)
        self.dataset_info_label = ttk.Label(dataset_frame, text="No se ha cargado ningún dataset.", justify=tk.LEFT)
        self.dataset_info_label.pack(pady=5)

        # --- Parameters Section ---
        params_frame = ttk.LabelFrame(main_frame, text="2. Configurar Parámetros", padding="10 10 10 10")
        params_frame.pack(fill=tk.X, pady=5)

        ttk.Label(params_frame, text="Tasa de aprendizaje (η):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.lr_entry = ttk.Entry(params_frame, width=15)
        self.lr_entry.grid(row=0, column=1, pady=2)
        self.lr_entry.insert(0, "0.1")

        ttk.Label(params_frame, text="Máximo de iteraciones:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.max_iter_entry = ttk.Entry(params_frame, width=15)
        self.max_iter_entry.grid(row=1, column=1, pady=2)
        self.max_iter_entry.insert(0, "100")

        ttk.Label(params_frame, text="Error máximo permitido (ε):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.tol_entry = ttk.Entry(params_frame, width=15)
        self.tol_entry.grid(row=2, column=1, pady=2)
        self.tol_entry.insert(0, "0.01")

        # --- Training Section ---
        train_frame = ttk.LabelFrame(main_frame, text="3. Entrenamiento", padding="10 10 10 10")
        train_frame.pack(fill=tk.X, pady=5)

        self.train_button = ttk.Button(train_frame, text="🚀 Entrenar Modelo", command=self.entrenar)
        self.train_button.pack(pady=5)
        self.train_status_label = ttk.Label(train_frame, text="El modelo no ha sido entrenado.")
        self.train_status_label.pack(pady=5)

        # --- Simulation Section ---
        sim_frame = ttk.LabelFrame(main_frame, text="4. Probar y Validar", padding="10 10 10 10")
        sim_frame.pack(fill=tk.X, pady=5)

        ttk.Label(sim_frame, text="Nuevo patrón (separado por comas):").pack(pady=2)
        self.pattern_entry = ttk.Entry(sim_frame, width=40)
        self.pattern_entry.pack(pady=2)
        self.test_button = ttk.Button(sim_frame, text="🔮 Probar Patrón", command=self.probar)
        self.test_button.pack(pady=5)
        self.result_label = ttk.Label(sim_frame, text="Resultado de la predicción aparecerá aquí.", font=('Helvetica', 10, 'italic'))
        self.result_label.pack(pady=5)

        # --- Graph Section ---
        self.graph_frame = ttk.LabelFrame(main_frame, text="5. Gráfica de Error", padding="10 10 10 10")
        self.graph_frame.pack(fill=tk.BOTH, expand=True, pady=5)

    def cargar(self):
        try:
            file = filedialog.askopenfilename(initialdir=os.path.join(os.path.dirname(__file__), "../datasets"),
                                              filetypes=[("CSV files", "*.csv")])
            if file:
                nombre = os.path.basename(file)
                self.dataset = cargar_dataset(nombre)
                self.X = self.dataset.iloc[:, :-1].values
                self.y = self.dataset.iloc[:, -1].values
                info_text = (f"📂 Cargado: {nombre}\n"
                             f"➡️ Patrones: {self.dataset.shape[0]}\n"
                             f"➡️ Entradas: {self.X.shape[1]}")
                self.dataset_info_label.config(text=info_text)
                self.train_status_label.config(text="Modelo listo para entrenar.")
                self.pattern_entry.delete(0, tk.END) # Clear previous pattern
                self.result_label.config(text="Resultado de la predicción aparecerá aquí.")
        except Exception as e:
            messagebox.showerror("Error al Cargar", f"No se pudo cargar el archivo: {e}")

    def entrenar(self):
        if self.X is None or self.y is None:
            messagebox.showerror("Error", "Debes cargar un dataset primero.")
            return
        try:
            lr = float(self.lr_entry.get())
            max_iter = int(self.max_iter_entry.get())
            tol = float(self.tol_entry.get())

            self.model = Perceptron(n_inputs=self.X.shape[1], lr=lr, max_iter=max_iter, tol=tol)
            self.model.fit(self.X, self.y)

            self.train_status_label.config(text="✅ Entrenamiento completado.", foreground="green")
            self.mostrar_grafica()
        except ValueError:
            messagebox.showerror("Error de Parámetros", "Asegúrate de que los parámetros sean números válidos.")
        except Exception as e:
            messagebox.showerror("Error en Entrenamiento", f"Ocurrió un error: {e}")

    def mostrar_grafica(self):
        if self.model is None or not self.model.errors:
            messagebox.showerror("Error", "No hay datos de error para graficar. Entrena el modelo primero.")
            return

        for widget in self.graph_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
        ax.plot(self.model.errors, marker='o', linestyle='-', color='b')
        ax.set_title("Evolución del Error de Entrenamiento")
        ax.set_xlabel("Iteraciones")
        ax.set_ylabel("Error Promedio")
        ax.grid(True)

        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def probar(self):
        if self.model is None:
            messagebox.showerror("Error", "Debes entrenar el modelo primero.")
            return
        patron_str = self.pattern_entry.get()
        if not patron_str:
            messagebox.showwarning("Entrada Vacía", "Por favor, ingrese un patrón para probar.")
            return
        try:
            nuevo = np.array([list(map(float, patron_str.split(",")))])
            if nuevo.shape[1] != self.X.shape[1]:
                messagebox.showerror("Error de Dimensión", f"El patrón debe tener {self.X.shape[1]} entradas.")
                return
            pred = self.model.predict(nuevo)[0]
            self.result_label.config(text=f"🔮 El patrón [{patron_str}] se clasifica como: {pred}", foreground="#007BFF")
        except ValueError:
            messagebox.showerror("Error de Formato", "El patrón debe ser una serie de números separados por comas.")
        except Exception as e:
            messagebox.showerror("Error en Predicción", f"Ocurrió un error: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PerceptronApp(root)
    root.mainloop()
