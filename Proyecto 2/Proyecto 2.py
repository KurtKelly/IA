import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

df = pd.read_csv("trainingData.csv")
df.head()

print(df.shape)
df['FLOOR'].value_counts()

cols_to_drop = ['LONGITUDE','LATITUDE','SPACEID','RELATIVEPOSITION',
                'USERID','PHONEID','TIMESTAMP']

df = df.drop(columns=cols_to_drop)

X = df.filter(regex='^WAP')
y = df['FLOOR']

X[X == 100] = -100



X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)



param_knn = {
    'n_neighbors': [3,5,7],
    'weights': ['uniform','distance']
}

gs_knn = GridSearchCV(
    KNeighborsClassifier(),
    param_knn,
    cv=5,
    scoring='accuracy'
)

gs_knn.fit(X_train, y_train)
best_knn = gs_knn.best_estimator_

gnb = GaussianNB()
gnb.fit(X_train, y_train)


param_lr = {'C':[0.1,1,10]}
gs_lr = GridSearchCV(
    LogisticRegression(max_iter=2000),
    param_lr,
    cv=5
)
gs_lr.fit(X_train, y_train)
best_lr = gs_lr.best_estimator_



param_dt = {'max_depth':[5,10,20]}
gs_dt = GridSearchCV(
    DecisionTreeClassifier(),
    param_dt,
    cv=5
)
gs_dt.fit(X_train, y_train)
best_dt = gs_dt.best_estimator_


param_svm = {
    'C':[0.1,1,10],
    'kernel':['rbf','linear'],
    'gamma':['scale','auto']
}

rs_svm = RandomizedSearchCV(
    SVC(),
    param_svm,
    n_iter=6,
    cv=5
)
rs_svm.fit(X_train, y_train)
best_svm = rs_svm.best_estimator_


param_rf = {
    'n_estimators':[100,200],
    'max_depth':[10,20]
}

gs_rf = GridSearchCV(
    RandomForestClassifier(n_jobs=-1),
    param_rf,
    cv=5
)
gs_rf.fit(X_train, y_train)
best_rf = gs_rf.best_estimator_

def prepare_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train = train.drop(columns=cols_to_drop)
    test = test.drop(columns=cols_to_drop)

    X_train = train.filter(regex='^WAP')
    y_train = train['FLOOR']
    X_test = test.filter(regex='^WAP')
    y_test = test['FLOOR']

    X_train[X_train==100] = -100
    X_test[X_test==100] = -100

    return X_train, X_test, y_train, y_test

X_train_f, X_test_f, y_train_f, y_test_f = prepare_data(
    "trainingData.csv",
    "validationData.csv"
)



def evaluate_model(model, X_train, y_train, X_test, y_test):
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train

    start_test = time.time()
    y_pred = model.predict(X_test)
    test_time = time.time() - start_test

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro"),
        "recall": recall_score(y_test, y_pred, average="macro"),
        "f1": f1_score(y_test, y_pred, average="macro"),
        "train_time": train_time,
        "test_time": test_time
    }

models = {
    "KNN": best_knn,
    "GaussianNB": gnb,
    "Logistic Regression": best_lr,
    "Decision Tree": best_dt,
    "SVM": best_svm,
    "Random Forest": best_rf
}

results = {}

for name, model in models.items():
    print(f"\nEvaluando {name}...")
    results[name] = evaluate_model(
        model,
        X_train_f, y_train_f,
        X_test_f, y_test_f
    )

results_df = pd.DataFrame(results).T
print(results_df)

