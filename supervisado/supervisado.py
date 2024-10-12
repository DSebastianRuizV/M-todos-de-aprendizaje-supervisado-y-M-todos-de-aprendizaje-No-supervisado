# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


data = {
    'hora': [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
    'numero_pasajeros': [45, 60, 30, 20, 50, 70, 80, 40, 35, 55, 90, 85, 65, 30, 25, 15],
    'tiempo_espera': [5, 7, 3, 2, 6, 8, 10, 4, 4, 7, 12, 10, 8, 3, 2, 1],
    'llega_a_tiempo': [1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1]  
}


df = pd.DataFrame(data)


X = df[['hora', 'numero_pasajeros', 'tiempo_espera']] 
y = df['llega_a_tiempo']  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print('Precisión del modelo: {:.2f}%'.format(accuracy * 100))


print("Coeficientes del modelo:", model.coef_)
import matplotlib.pyplot as plt

pandas
scikit-learn
matplotlib
