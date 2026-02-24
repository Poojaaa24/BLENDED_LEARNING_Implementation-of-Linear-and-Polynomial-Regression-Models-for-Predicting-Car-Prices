# BLENDED_LEARNING
# Implementation-of-Linear-and-Polynomial-Regression-Models-for-Predicting-Car-Prices

## AIM:
To write a program to predict car prices using Linear Regression and Polynomial Regression models.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Libraries & Load Dataset
2. Divide the dataset into training and testing sets.
3. Select a suitable ML model, train it on the training data, and make predictions.
4. Assess model performance using metrics and interpret the results.

## Program:
```
/*
Program to implement Linear and Polynomial Regression models for predicting car prices.
Developed by: POOJA U
RegisterNumber:  212225230209
*/

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import matplotlib.pyplot as plt
#Load DatA
df = pd.read_csv('encoded_car_data (1).csv')
print(df.head())
#SELECT FEATURES AND Target
X=df[['enginesize','horsepower','citympg','highwaympg']]
Y=df['price']
#Split DatA
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
#1,LINEAR REGRESSION(WITH SCALING)
lr = Pipeline([
    ('scaler',StandardScaler()),
    ('model',LinearRegression())
])
lr.fit(X_train,Y_train)
y_pred_linear=lr.predict(X_test)
#2 POLYNOMIAL REGRE(D=2)
poly_model= Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])
poly_model.fit(X_train,Y_train)
y_pred_poly=poly_model.predict(X_test)
#EVALUATE MODELS
print('Name: POOJA U ')
print('Reg. No: 212225230209')
print("LinearRegression:")
# mse=mean_squared_error(Y_test,Y_pred_linear)
mean_absolute_error(Y_test,y_pred_linear)
print('MAE=',mean_absolute_error(Y_test,y_pred_linear))
print('MSE=',mean_squared_error(Y_test,y_pred_linear))
r2score=r2_score(Y_test,y_pred_linear)
print('R2 Score=',r2score)
print("\nPolynomial Regression: ")
print("MSE: {mean_squared_error(Y_test,y_pred_poly):.2f}")
print("MAE: {r2_score(Y-test, y_pred_poly):.2f}")

#Plot actual vs predicted
plt.figure(figsize=(10,5))
plt.scatter(Y_test,y_pred_linear,label='Linear',alpha=0.6)
plt.scatter(Y_test,y_pred_poly, label='Polynomial(degree=2)',alpha=0.6)
plt.plot([Y.min(),Y.max()],[Y.min(),Y.max()],'r--',label='Perfect Prediction')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.show


```

## Output:
<img width="1319" height="568" alt="Screenshot 2026-02-12 101247" src="https://github.com/user-attachments/assets/b4d1c738-a159-45ee-811c-7500fa3e4017" />
<img width="1337" height="587" alt="Screenshot 2026-02-12 101314" src="https://github.com/user-attachments/assets/d470a9b6-57d8-43e6-8223-278efa799668" />

<img width="1332" height="875" alt="Screenshot 2026-02-12 101328" src="https://github.com/user-attachments/assets/801d227d-fed4-4ab7-8b1a-15d7dd940c11" />

![simple linear regression model for predicting the marks scored](sam.png)


## Result:
Thus, the program to implement Linear and Polynomial Regression models for predicting car prices was written and verified using Python programming.
