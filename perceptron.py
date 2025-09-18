
# n_inputs: cantidad de entradas.

# lr: tasa de aprendizaje.

# max_iter: número máximo de repeticiones.

# tol: tolerancia de error (si llega a un error muy pequeño, se detiene antes).

#w_init y b_init: inicialización de pesos y bias (al azar o con valores dados).

import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, n_inputs, lr=0.1, max_iter=100, tol=0.01,
                 w_init="random", b_init="random"):
        # Inicialización de pesos y umbral
        if w_init == "random":
            self.w = np.random.uniform(-1, 1, n_inputs)
        else:
            self.w = np.array(w_init, dtype=float)

        if b_init == "random":
            self.b = np.random.uniform(-1, 1)
        else:
            self.b = float(b_init)

        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.errors = []

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, X):
        linear = np.dot(X, self.w) + self.b
        return np.atleast_1d((linear >= 0).astype(int))

    def fit(self, X, y):
        for epoch in range(self.max_iter):
            total_error = 0
            for xi, target in zip(X, y):
                y_hat = self.activation(np.dot(xi, self.w) + self.b)
                error = target - y_hat
                # Regla Delta
                self.w += self.lr * error * xi
                self.b += self.lr * error
                total_error += abs(error)

            avg_error = total_error / len(y)
            self.errors.append(avg_error)

            if avg_error <= self.tol:
                print(f"✅ Entrenamiento detenido en la época {epoch+1}")
                break

    def plot_errors(self, save_path=None):
        plt.plot(self.errors, marker='o')
        plt.xlabel("Iteraciones")
        plt.ylabel("Error promedio")
        plt.title("Evolución del error en el entrenamiento")
        if save_path:
            plt.savefig(save_path)
        plt.show()