import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv("mnist_train.csv")
df.head()
df.shape
df.isnull().sum().head()

X = df.drop("label", axis=1)
y = df["label"]

X.shape, y.nunique()



X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train.shape, X_test.shape

depths = [5, 10, 20]



results = []

for d in depths:
    model = DecisionTreeClassifier(max_depth=d, random_state=42)
    model.fit(X_train, y_train)

    acc_train = accuracy_score(y_train, model.predict(X_train))
    acc_test = accuracy_score(y_test, model.predict(X_test))

    results.append({
        "Profundidad": d,
        "Accuracy Train": acc_train,
        "Accuracy Test": acc_test
    })

results_df = pd.DataFrame(results)
print(results_df)


plt.figure(figsize=(8,5))
plt.plot(results_df["Profundidad"], results_df["Accuracy Train"], label="Train")
plt.plot(results_df["Profundidad"], results_df["Accuracy Test"], label="Test")
plt.xlabel("Profundidad")
plt.ylabel("Accuracy")
plt.title("Train vs Test - Árboles de Decisión")
plt.legend()
plt.grid()
plt.show()
