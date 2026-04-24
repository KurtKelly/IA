import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# -------------------------------------------------
# Paso 1: Cargar y explorar dataset
# -------------------------------------------------

df = pd.read_csv("cleaned_dataset.csv")
print(df.head())
print(df.info())
print(df.describe())

# -------------------------------------------------
# Paso 2: Función de distancia euclidiana
# -------------------------------------------------

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

# Prueba con los ejemplos dados en la guía
x = np.array([1,106,70,28,135,34.2,0.142,22])
y = np.array([2,102,86,36,120,45.5,0.127,23])

dist = euclidean_distance(x, y)
print("Distancia euclidiana:", dist)

# -------------------------------------------------
# Paso 3: Implementación manual de KNN
# -------------------------------------------------

def knn_predict(X_train, y_train, x_test, k=3):
    distances = []

    for i in range(len(X_train)):
        dist = euclidean_distance(X_train[i], x_test)
        distances.append((dist, y_train[i]))

    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    labels = [label for _, label in neighbors]

    return Counter(labels).most_common(1)[0][0]

# -------------------------------------------------
# Paso 4: Separar variables y train/test split
# -------------------------------------------------

X = df.drop("Outcome", axis=1).values
y = df["Outcome"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# -------------------------------------------------
# Paso 5: Predicción individual explicada (KNN manual)
# -------------------------------------------------

x_test_example = X_test[0]
y_true = y_test[0]

y_pred_manual = knn_predict(X_train, y_train, x_test_example, k=3)
print("Real:", y_true, "Predicho (KNN manual):", y_pred_manual)

# -------------------------------------------------
# Paso 6: KNN con scikit-learn sin escalar
# -------------------------------------------------

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
acc_raw = accuracy_score(y_test, y_pred)
print("Accuracy sin escalar:", acc_raw)

# -------------------------------------------------
# Paso 7: Normalización Min-Max
# -------------------------------------------------

scaler = MinMaxScaler()
X_train_mm = scaler.fit_transform(X_train)
X_test_mm = scaler.transform(X_test)

knn.fit(X_train_mm, y_train)
pred_mm = knn.predict(X_test_mm)
acc_mm = accuracy_score(y_test, pred_mm)
print("Accuracy Min-Max:", acc_mm)

# -------------------------------------------------
# Paso 8: Estandarización Z-score
# -------------------------------------------------

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

knn.fit(X_train_std, y_train)
pred_std = knn.predict(X_test_std)
acc_std = accuracy_score(y_test, pred_std)
print("Accuracy Z-score:", acc_std)

# -------------------------------------------------
# Tabla comparativa (para Markdown / Word)
# -------------------------------------------------

print("\nResumen de accuracies:")
print("Sin escalar:", acc_raw)
print("Min-Max:", acc_mm)
print("Z-score:", acc_std)
