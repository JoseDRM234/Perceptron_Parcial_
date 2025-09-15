import numpy as np
from perceptron import Perceptron
from utils import cargar_dataset

def run_test():
    print("--- Running Test ---")
    try:
        # 1. Cargar dataset
        dataset = cargar_dataset("dataset1.csv")
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        print("Dataset loaded successfully.")

        # 2. Crear y entrenar el perceptrón
        n_inputs = X.shape[1]
        model = Perceptron(n_inputs=n_inputs, lr=0.1, max_iter=100)
        print("Training model...")
        model.fit(X, y)
        print("Model trained.")

        # 3. Probar con un nuevo patrón
        test_pattern = np.array([1, 1, 0])
        prediction = model.predict(test_pattern)
        print(f"Prediction for {test_pattern} is {prediction[0]}")

        # Prueba con otro patrón
        test_pattern_2 = np.array([0, 0, 1])
        prediction_2 = model.predict(test_pattern_2)
        print(f"Prediction for {test_pattern_2} is {prediction_2[0]}")

        print("--- Test Finished ---")

    except Exception as e:
        print(f"An error occurred during the test: {e}")

if __name__ == "__main__":
    run_test()