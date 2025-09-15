import pandas as pd
import os

DATASET_PATH = os.path.join(os.path.dirname(__file__), "../datasets/")

def cargar_dataset(nombre):
    ruta = os.path.join(DATASET_PATH, nombre)
    df = pd.read_csv(ruta)
    print(f"📂 Dataset cargado: {nombre}")
    print(f"➡️ Patrones: {df.shape[0]}, Entradas: {df.shape[1] - 1}, Salidas: 1")
    return df

def seleccionar_dataset():
    print("\n=== Selección de Dataset ===")
    print("1. Dataset 1 (binario, 3 entradas)")
    print("2. Dataset 2 (binario, 4 entradas)")
    print("3. Dataset 3 (académico)")
    opcion = input("Elige dataset (1, 2 o 3): ").strip()
    return f"dataset{opcion}.csv"
