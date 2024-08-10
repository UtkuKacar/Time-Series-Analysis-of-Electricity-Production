import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Electric_Production.csv')
X = np.arange(len(data)).reshape(-1, 1)
y = data['Value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2) Score:", r2)

plt.figure(figsize=(12, 6))
plt.plot(data['DATE'], data['Value'], color='blue')
plt.title('Üretim Zaman Grafiği')
plt.xlabel('Tarih')
plt.ylabel('Electric Production')
plt.show()

plt.figure(figsize=(12, 6))
plt.scatter(X_test, y_test, color='blue', label='Gerçek Değerler')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Tahminler')
plt.title('Gerçek vs. Tahmin Edilen Değerler')
plt.xlabel('Index')
plt.ylabel('Electric Production')
plt.legend()
plt.show()

error = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.hist(error, bins=25, edgecolor='black')
plt.title('Hata Dağılımı')
plt.xlabel('Hata')
plt.ylabel('Frekans')
plt.show()
