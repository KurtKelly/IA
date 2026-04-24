import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# -------------------------------------------------
# Cargar y preparar dataset
# -------------------------------------------------

df = pd.read_csv("balanceSheetHistory_annually.csv")
df = df[['stock', 'endDate', 'cash']]

df['endDate'] = pd.to_datetime(df['endDate'])
df = df.sort_values(['stock', 'endDate'])

selected_stocks = ['AAPL', 'MSFT', 'GOOGL']
df_sel = df[df['stock'].isin(selected_stocks)]

# -------------------------------------------------
# Paso 4: Graficar cash vs tiempo
# -------------------------------------------------

for stock in selected_stocks:
    subset = df_sel[df_sel['stock'] == stock]
    plt.figure(figsize=(8,4))
    plt.plot(subset['endDate'], subset['cash'], label='Cash')
    plt.title(f'Cash vs Time - {stock}')
    plt.xlabel('Date')
    plt.ylabel('Cash')
    plt.grid()
    plt.legend()
    plt.show()

# -------------------------------------------------
# Paso 5–6: Split temporal + regresión
# -------------------------------------------------

models = {}
metrics = {}

for stock in selected_stocks:
    data = df_sel[df_sel['stock'] == stock].copy()

    # ✅ Crear rezago ANTES del split
    data['cash_lag1'] = data['cash'].shift(1)
    data.dropna(inplace=True)

    split = int(len(data) * 0.8)
    train = data.iloc[:split]
    test = data.iloc[split:]

    X_train = train[['cash_lag1']]
    y_train = train['cash']

    X_test = test[['cash_lag1']]
    y_test = test['cash']

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Guardar modelo y métricas
    models[stock] = (model, X_test, y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics[stock] = {'MSE': mse, 'RMSE': rmse, 'R2': r2}

# -------------------------------------------------
# Paso 7: Gráficas real vs predicho
# -------------------------------------------------

for stock, (_, X_test, y_test, y_pred) in models.items():
    plt.figure(figsize=(8, 4))
    plt.plot(y_test.index, y_test.values, 'o-', label='Real')
    plt.plot(y_test.index, y_pred, 'o--', label='Predicho')
    plt.title(f'Real vs Predicho - {stock}')
    plt.xlabel('Observación temporal')
    plt.ylabel('Cash')
    plt.legend()
    plt.grid()
    plt.show()

# -------------------------------------------------
# Paso 8: Mostrar métricas
# -------------------------------------------------

print("\nMétricas por empresa:")

for stock, m in metrics.items():
    print(f"\n{stock}")
    print("MSE :", m['MSE'])
    print("RMSE:", m['RMSE'])
    print("R2  :", m['R2'])