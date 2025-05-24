# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries. 
2. Set variables for assigning dataset values. 
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict regression for marks by representing in a graph.
6. Compare graphs and hence linear regression is obtained for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: TARANIKKA A
RegisterNumber: 212223220115
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x=df.iloc[:,:-1].values
print(x)
y=df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train, regressor.predict(x_train),color='blue') 
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train, regressor.predict(x_train),color='red')
plt.title("Hours vs Scores (Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE=',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE=',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)


```

## Output:
HEAD VALUES:
![1ml](https://github.com/user-attachments/assets/7f816587-8aeb-438c-bee3-780f4339588f)

TAIL VALUES:
![2ml](https://github.com/user-attachments/assets/c7b5aff2-3d07-4da1-865c-a25500be8a94)

COMPARE DATASET:
![3ml](https://github.com/user-attachments/assets/0447e0f5-1bb0-45e8-945b-e7ec5f08968f)

PREDICTED VALUES:
![4ml](https://github.com/user-attachments/assets/a3ea07d0-227c-4d0e-8d26-42cc4dfdf572)

TRAINING SET:
![5ml](https://github.com/user-attachments/assets/9f4e520d-d1ec-42d5-87ea-a0e51cfca6d2)

TESTING SET:
![6ml](https://github.com/user-attachments/assets/e9af8022-0c25-4594-b23f-c91652913ee1)

ERROR:
![7ml](https://github.com/user-attachments/assets/8616702c-1b65-4659-b9f4-a93960907112)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
