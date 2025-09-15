import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from perceptron import Perceptron
from utils import cargar_dataset
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "../results/")
if not os.path.exists(os.path.join(RESULTS_PATH, "graficas")):
    os.makedirs(os.path.join(RESULTS_PATH, "graficas"))

class PerceptronApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 Perceptrón Simple - Interfaz Mejorada")
        self.root.state("zoomed")
        self.root.configure(bg="#f0f2f5")

        # === Variables principales ===
        self.dataset = None
        self.X = None
        self.y = None
        self.model = None
        self.canvas = None

        # === Configurar estilos ===
        self.setup_styles()

        # === Layout principal ===
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=2)

        # === Panel de control izquierdo ===
        self.create_control_panel()

        # === Panel de visualización derecho ===
        self.create_visualization_panel()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        
        # Configurar colores y fuentes
        style.configure("TLabel", font=("Segoe UI", 10), background="#f0f2f5")
        style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"), foreground="#2c3e50")
        style.configure("Info.TLabel", font=("Segoe UI", 9), foreground="#7f8c8d")
        style.configure("Result.TLabel", font=("Segoe UI", 11, "bold"), foreground="#27ae60")
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=(12, 8))
        style.configure("Primary.TButton", background="#3498db")
        style.configure("Success.TButton", background="#27ae60")
        style.configure("TEntry", font=("Segoe UI", 10), padding=5)
        style.configure("TCombobox", font=("Segoe UI", 10))

    def create_control_panel(self):
        # Frame principal izquierdo
        control_frame = ttk.Frame(self.root, padding=15)
        control_frame.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        
        # === Sección Dataset ===
        dataset_section = ttk.LabelFrame(control_frame, text="📂 Dataset", padding=15)
        dataset_section.pack(fill="x", pady=(0, 15))

        self.load_button = ttk.Button(
            dataset_section, 
            text="📁 Cargar Dataset", 
            command=self.cargar,
            style="Primary.TButton"
        )
        self.load_button.pack(pady=(0, 10))

        self.dataset_info_label = ttk.Label(
            dataset_section, 
            text="📄 No se ha cargado ningún dataset",
            style="Info.TLabel"
        )
        self.dataset_info_label.pack()

        # === Sección Configuración ===
        config_section = ttk.LabelFrame(control_frame, text="⚙️ Configuración del Modelo", padding=15)
        config_section.pack(fill="x", pady=(0, 15))

        # Grid para parámetros
        params_grid = ttk.Frame(config_section)
        params_grid.pack(fill="x")

        # Tasa de aprendizaje
        ttk.Label(params_grid, text="Tasa de aprendizaje (η):").grid(row=0, column=0, sticky="w", pady=5)
        self.lr_entry = ttk.Entry(params_grid, width=15)
        self.lr_entry.insert(0, "0.1")
        self.lr_entry.grid(row=0, column=1, sticky="w", padx=(10, 0), pady=5)

        # Máximo iteraciones
        ttk.Label(params_grid, text="Máximo iteraciones:").grid(row=1, column=0, sticky="w", pady=5)
        self.max_iter_entry = ttk.Entry(params_grid, width=15)
        self.max_iter_entry.insert(0, "100")
        self.max_iter_entry.grid(row=1, column=1, sticky="w", padx=(10, 0), pady=5)

        # Error máximo
        ttk.Label(params_grid, text="Error máximo (ε):").grid(row=2, column=0, sticky="w", pady=5)
        self.tol_entry = ttk.Entry(params_grid, width=15)
        self.tol_entry.insert(0, "0.01")
        self.tol_entry.grid(row=2, column=1, sticky="w", padx=(10, 0), pady=5)

        # === Sección Entrenamiento ===
        train_section = ttk.LabelFrame(control_frame, text="🚀 Entrenamiento", padding=15)
        train_section.pack(fill="x", pady=(0, 15))

        self.train_button = ttk.Button(
            train_section, 
            text="🎯 Entrenar Modelo", 
            command=self.entrenar,
            style="Success.TButton"
        )
        self.train_button.pack(pady=(0, 10))

        self.train_status_label = ttk.Label(
            train_section, 
            text="⏳ Modelo no entrenado",
            style="Info.TLabel"
        )
        self.train_status_label.pack()

        # === Sección Pruebas ===
        test_section = ttk.LabelFrame(control_frame, text="🧪 Pruebas del Modelo", padding=15)
        test_section.pack(fill="x", pady=(0, 15))

        # Prueba manual
        ttk.Label(test_section, text="Patrón manual:", style="Header.TLabel").pack(anchor="w", pady=(0, 5))
        ttk.Label(test_section, text="Ingresa valores separados por comas (ej: 1,0,1)", style="Info.TLabel").pack(anchor="w")
        
        self.pattern_entry = ttk.Entry(test_section, width=35)
        self.pattern_entry.pack(pady=(5, 10), fill="x")

        self.test_button = ttk.Button(
            test_section, 
            text="🔍 Probar Patrón Manual", 
            command=self.probar_manual
        )
        self.test_button.pack(pady=(0, 15))

        # Prueba desde dataset
        ttk.Label(test_section, text="Probar desde dataset:", style="Header.TLabel").pack(anchor="w", pady=(0, 5))
        ttk.Label(test_section, text="Selecciona una fila del dataset cargado", style="Info.TLabel").pack(anchor="w")
        
        self.combo_filas = ttk.Combobox(test_section, state="readonly", width=40)
        self.combo_filas.pack(pady=(5, 10), fill="x")

        self.test_dataset_button = ttk.Button(
            test_section, 
            text="🎲 Probar Fila del Dataset", 
            command=self.probar_dataset
        )
        self.test_dataset_button.pack(pady=(0, 15))

        # Resultado
        result_frame = ttk.Frame(test_section)
        result_frame.pack(fill="x", pady=(10, 0))
        
        ttk.Label(result_frame, text="Resultado:", style="Header.TLabel").pack(anchor="w")
        self.result_label = ttk.Label(
            result_frame, 
            text="🤖 Aquí aparecerá el resultado de la predicción",
            style="Result.TLabel",
            wraplength=300
        )
        self.result_label.pack(anchor="w", pady=(5, 0))

    def create_visualization_panel(self):
        # Frame principal derecho
        viz_frame = ttk.Frame(self.root, padding=15)
        viz_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        viz_frame.rowconfigure(0, weight=1)
        viz_frame.columnconfigure(0, weight=1)

        # === Sección Gráfica ===
        self.graph_frame = ttk.LabelFrame(viz_frame, text="📊 Evolución del Error durante el Entrenamiento", padding=15)
        self.graph_frame.grid(row=0, column=0, sticky="nsew")
        
        # Placeholder inicial
        self.create_placeholder_graph()

    def create_placeholder_graph(self):
        """Crea una gráfica placeholder cuando no hay datos"""
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        ax.text(0.5, 0.5, '📈 Gráfica aparecerá después del entrenamiento', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14, color='#7f8c8d')
        ax.set_title("Evolución del Error", fontsize=16, fontweight="bold", pad=20)
        ax.set_xlabel("Iteraciones", fontsize=12)
        ax.set_ylabel("Error", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_facecolor("#fafafa")
        fig.patch.set_facecolor("#f0f2f5")
        
        self.canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # ------------------------
    # Funciones principales
    # ------------------------
    def cargar(self):
        file = filedialog.askopenfilename(
            title="Seleccionar Dataset",
            initialdir=os.path.join(os.path.dirname(__file__), "../datasets"),
            filetypes=[("CSV files", "*.csv"), ("Todos los archivos", "*.*")]
        )
        if file:
            try:
                nombre = os.path.basename(file)
                self.dataset = cargar_dataset(nombre)
                self.X = self.dataset.iloc[:, :-1].values
                self.y = self.dataset.iloc[:, -1].values

                # Actualizar información del dataset
                info_text = f"✅ {nombre}\n📊 Patrones: {self.dataset.shape[0]} | Entradas: {self.X.shape[1]}"
                self.dataset_info_label.config(text=info_text, foreground="#27ae60")

                # Llenar combobox con las filas del dataset
                opciones = []
                for i, row in self.dataset.iterrows():
                    entradas = ', '.join([str(x) for x in row.values[:-1]])
                    salida = row.values[-1]
                    opciones.append(f"Fila {i+1}: [{entradas}] → {salida}")
                
                self.combo_filas["values"] = opciones
                if opciones:
                    self.combo_filas.current(0)  # Seleccionar la primera por defecto

                self.train_status_label.config(
                    text="✅ Dataset cargado correctamente. ¡Listo para entrenar!", 
                    foreground="#27ae60"
                )
                
            except Exception as e:
                messagebox.showerror("Error al cargar dataset", f"No se pudo cargar el archivo:\n{str(e)}")
                self.dataset_info_label.config(text="❌ Error al cargar dataset", foreground="#e74c3c")

    def entrenar(self):
        if self.X is None or self.y is None:
            messagebox.showerror("Error", "⚠️ Debes cargar un dataset primero.")
            return
        
        try:
            # Obtener parámetros
            lr = float(self.lr_entry.get())
            max_iter = int(self.max_iter_entry.get())
            tol = float(self.tol_entry.get())

            # Validar parámetros
            if lr <= 0:
                raise ValueError("La tasa de aprendizaje debe ser mayor que 0")
            if max_iter <= 0:
                raise ValueError("El máximo de iteraciones debe ser mayor que 0")
            if tol <= 0:
                raise ValueError("El error máximo debe ser mayor que 0")

            # Entrenar modelo
            self.model = Perceptron(n_inputs=self.X.shape[1], lr=lr, max_iter=max_iter, tol=tol)
            self.model.fit(self.X, self.y)
            
            # Actualizar estado
            iteraciones = len(self.model.errors)
            error_final = self.model.errors[-1] if self.model.errors else "N/A"
            status_text = f"🎉 Entrenamiento completado!\n🔄 Iteraciones: {iteraciones} | 📉 Error final: {error_final:.4f}"
            self.train_status_label.config(text=status_text, foreground="#27ae60")
            
            # Mostrar gráfica
            self.mostrar_grafica()
            
        except ValueError as e:
            messagebox.showerror("Error de parámetros", f"⚠️ {str(e)}")
        except Exception as e:
            messagebox.showerror("Error en entrenamiento", f"❌ Error durante el entrenamiento:\n{str(e)}")

    def mostrar_grafica(self):
        if self.model is None or not self.model.errors:
            return

        # Limpiar gráfica anterior
        for widget in self.graph_frame.winfo_children():
            widget.destroy()

        # Crear nueva gráfica
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        
        # Plotear datos
        iteraciones = range(1, len(self.model.errors) + 1)
        ax.plot(iteraciones, self.model.errors, 
                marker='o', linestyle='-', color='#3498db', 
                linewidth=2.5, markersize=6, markerfacecolor='#e74c3c')
        
        # Configurar apariencia
        ax.set_title("Evolución del Error durante el Entrenamiento", 
                    fontsize=16, fontweight="bold", pad=20)
        ax.set_xlabel("Iteraciones", fontsize=12)
        ax.set_ylabel("Error", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.7, color='#bdc3c7')
        ax.set_facecolor("#fafafa")
        fig.patch.set_facecolor("#f0f2f5")
        
        # Añadir información adicional
        if len(self.model.errors) > 1:
            ax.text(0.02, 0.98, f'Error inicial: {self.model.errors[0]:.4f}\nError final: {self.model.errors[-1]:.4f}', 
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        # Mostrar gráfica
        self.canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def probar_manual(self):
        if self.model is None:
            messagebox.showerror("Error", "🚫 Debes entrenar el modelo primero.")
            return
        
        try:
            patron_text = self.pattern_entry.get().strip()
            if not patron_text:
                messagebox.showwarning("Entrada vacía", "⚠️ Ingresa un patrón para probar.")
                return
                
            # Parsear patrón
            patron = np.array([list(map(float, patron_text.split(",")))])
            
            # Verificar dimensiones
            if patron.shape[1] != self.X.shape[1]:
                raise ValueError(f"El patrón debe tener {self.X.shape[1]} valores")
            
            # Predecir
            prediccion = self.model.predict(patron)[0]
            
            # Mostrar resultado
            resultado_text = f"🎯 Patrón: [{patron_text}]\n🤖 Predicción: {prediccion}"
            self.result_label.config(text=resultado_text, foreground="#2980b9")
            
        except ValueError as e:
            if "could not convert" in str(e):
                messagebox.showerror("Error de formato", 
                    "❌ Formato inválido. Usa números separados por comas.\nEjemplo: 1,0,1")
            else:
                messagebox.showerror("Error de entrada", f"❌ {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"❌ Error inesperado: {str(e)}")

    def probar_dataset(self):
        if self.model is None:
            messagebox.showerror("Error", "🚫 Debes entrenar el modelo primero.")
            return
            
        if self.combo_filas.current() == -1:
            messagebox.showwarning("Sin selección", "⚠️ Selecciona una fila del dataset.")
            return
        
        try:
            idx = self.combo_filas.current()
            
            # Obtener datos
            entradas = np.array([self.X[idx]], dtype=float)
            salida_esperada = self.y[idx]
            prediccion = self.model.predict(entradas)[0]
            
            # Determinar si la predicción es correcta
            es_correcto = "✅" if prediccion == salida_esperada else "❌"
            
            # Formatear entradas para mostrar
            entradas_texto = ', '.join([str(x) for x in self.X[idx]])
            
            # Mostrar resultado
            resultado_text = (f"📋 Fila {idx + 1}: [{entradas_texto}]\n"
                            f"🎯 Esperado: {salida_esperada} | 🤖 Predicho: {prediccion} {es_correcto}")
            
            color = "#27ae60" if prediccion == salida_esperada else "#e74c3c"
            self.result_label.config(text=resultado_text, foreground=color)
            
        except Exception as e:
            messagebox.showerror("Error", f"❌ Error al probar: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PerceptronApp(root)
    root.mainloop()