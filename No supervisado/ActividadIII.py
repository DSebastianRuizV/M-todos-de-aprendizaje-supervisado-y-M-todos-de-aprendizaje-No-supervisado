import pandas as pd
import numpy as np

# Crear un dataset de muestra con datos de estaciones y rutas
data = {
    'estacion_origen': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D'],
    'estacion_destino': ['B', 'C', 'A', 'D', 'A', 'D', 'A', 'C'],
    'distancia': [35, 10, 35, 20, 10, 40, 20, 40],
    'tiempo_viaje': [30, 15, 30, 25, 15, 45, 25, 45],
    'frecuencia_servicio': [10, 15, 10, 20, 15, 25, 20, 25]
}

df = pd.DataFrame(data)
print(df)

from sklearn.model_selection import train_test_split

# Variables independientes (features) y dependiente (target)
X = df[['distancia', 'frecuencia_servicio']]
y = df['tiempo_viaje']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Crear y entrenar el modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

import joblib

# Guardar el modelo entrenado
joblib.dump(model, 'modelo_transporte.pkl')