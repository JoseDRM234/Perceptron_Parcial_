import tkinter as tk
from tkinter import filedialog, messagebox
from perceptron import Perceptron
from utils import cargar_dataset
import numpy as np
import os
import matplotlib.pyplot as plt

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "../results/")

class PerceptronApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Perceptrón Simple")
        self.dataset = None
        self.X = None
        self.y = None
        self.model = None

        # Botones principales
        tk.Button(root, text="📂 Cargar Dataset", command=self.cargar).pack(pady=5)
        tk.Button(root, text="⚙️ Configurar Parámetros", command=self.configurar).pack(pady=5)
        tk.Button(root, text="🚀 Entrenar", command=self.entrenar).pack(pady=5)
        tk.Button(root, text="📊 Ver Gráfica Error", command=self.mostrar_grafica).pack(pady=5)
        tk.Button(root, text="🔮 Probar Patrón Nuevo", command=self.probar).pack(pady=5)

    def cargar(self):
        file = filedialog.askopenfilename(initialdir="../datasets", filetypes=[("CSV files", "*.csv")])
        if file:
            nombre = os.path.basename(file)
            self.dataset = cargar_dataset(nombre)
            self.X = self.dataset.iloc[:, :-1].values
            self.y = self.dataset.iloc[:, -1].values
            messagebox.showinfo("Dataset", f"📂 Cargado: {nombre}\n"
                                           f"➡️ Patrones: {self.dataset.shape[0]}, "
                                           f"Entradas: {self.dataset.shape[1]-1}, Salidas: 1")

    def configurar(self):
        self.lr = float(tk.simpledialog.askstring("Parámetro", "Tasa de aprendizaje (ej: 0.1):"))
        self.max_iter = int(tk.simpledialog.askstring("Parámetro", "Máximo de iteraciones:"))
        self.tol = float(tk.simpledialog.askstring("Parámetro", "Error máximo permitido:"))
        messagebox.showinfo("Parámetros", f"Configurados:\nη={self.lr}, iter={self.max_iter}, ε={self.tol}")

    def entrenar(self):
        if self.X is None or self.y is None:
            messagebox.showerror("Error", "Debes cargar un dataset primero.")
            return
        self.model = Perceptron(n_inputs=self.X.shape[1], lr=self.lr, max_iter=self.max_iter, tol=self.tol)
        self.model.fit(self.X, self.y)
        messagebox.showinfo("Entrenamiento", "✅ Entrenamiento completado.")

    def mostrar_grafica(self):
        if self.model is None:
            messagebox.showerror("Error", "Debes entrenar primero.")
            return
        save_path = os.path.join(RESULTS_PATH, "graficas", "error.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.plot_errors(save_path)

    def probar(self):
        if self.model is None:
            messagebox.showerror("Error", "Debes entrenar primero.")
            return
        patron = tk.simpledialog.askstring("Nuevo patrón", f"Ingrese {self.X.shape[1]} valores separados por coma:")
        if patron:
            nuevo = np.array([list(map(float, patron.split(",")))])
            pred = self.model.predict(nuevo)[0]
            messagebox.showinfo("Predicción", f"🔮 El patrón {patron} se clasifica como: {pred}")


if __name__ == "__main__":
    root = tk.Tk()
    app = PerceptronApp(root)
    root.mainloop()
