import pandas as pd
import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier

muestras_entrenamiento = 50000  # Número de muestras de entrenamiento
muestras_prueba = 10000        # Número de muestras de prueba

# Inicializar variables para medir tiempos
inicio_tiempo_entrenamiento = 0
fin_tiempo_entrenamiento = 0
tiempo_entrenamiento = 0

inicio_tiempo_prueba = 0
fin_tiempo_prueba = 0
tiempo_prueba = 0

def cargar_dataset(nombre_archivo, muestras):
    """
    Carga un dataset CSV y devuelve los datos normalizados y las etiquetas.
    """
    x = []  # Datos de entrada
    y = []  # Etiquetas
    datos = pd.read_csv(nombre_archivo)
    y = np.array(datos.iloc[0:muestras, 0])  # Primera columna: etiquetas
    x = np.array(datos.iloc[0:muestras, 1:]) / 255  # Normalización
    return x, y



# Cargar los datos de entrenamiento y prueba
x_entrenamiento, y_entrenamiento = cargar_dataset(r"C:\Users\Kurt\Desktop\Universidad\IA\Tarea #1\mnist_train.csv", muestras_entrenamiento)
x_prueba, y_prueba = cargar_dataset(r"C:\Users\Kurt\Desktop\Universidad\IA\Tarea #1\mnist_test.csv", muestras_prueba)


# Crear y entrenar el modelo KNN
knn = KNeighborsClassifier(n_neighbors=3)
inicio_tiempo_entrenamiento = time.time()
knn.fit(x_entrenamiento, y_entrenamiento)
fin_tiempo_entrenamiento = time.time()
tiempo_entrenamiento = fin_tiempo_entrenamiento - inicio_tiempo_entrenamiento
print(f"Tiempo de entrenamiento: {tiempo_entrenamiento:.2f} s")



# Evaluación simple sobre todo el conjunto de prueba y cálculo del tiempo de inferencia
import time
inicio_inferencia = time.time()
precision = knn.score(x_prueba, y_prueba)
fin_inferencia = time.time()
tiempo_inferencia = fin_inferencia - inicio_inferencia
print(f"Precisión global (score): {precision:.4f}")
print(f"Tiempo de inferencia sobre todo el conjunto de prueba: {tiempo_inferencia:.4f} s")


print("-------------------------------")
print("Resultados")
print("-------------------------------")
print("Muestras de entrenamiento: ", muestras_entrenamiento)
print("Tiempo de entrenamiento: ", round(tiempo_entrenamiento, 2), " s")
print("Muestras de prueba: ", muestras_prueba)
print("Tiempo de prueba: ", round(tiempo_prueba, 2), " s")
print("Precisión en prueba: ", round(precision * 100, 2), "%")


indices_aleatorios = np.random.choice(len(x_prueba), 10, replace=False)

# Obtener los valores reales y predichos
numeros_reales = y_prueba[indices_aleatorios]
numeros_predichos = knn.predict(x_prueba[indices_aleatorios])

# Crear la tabla con pandas
tabla_resultados = pd.DataFrame({
    'Índice': indices_aleatorios,
    'Número Real': numeros_reales,
    'Número Predicho': numeros_predichos
})

print(tabla_resultados)


