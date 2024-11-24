# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: M.K.Suriya prakash (Iot)
RegisterNumber: 24901016 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv(r'C:\Users\Suriya\Documents\student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
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
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
*/
```

## Output:
![Screenshot 2024-11-24 091858](https://github.com/user-attachments/assets/7704eca8-4106-42f6-8c68-34bbc8a6069f)

![Screenshot 2024-11-24 091921](https://github.com/user-attachments/assets/ea4e15cd-74ab-408e-8ad6-f76be94cddcf)


![Screenshot 2024-11-24 091941](https://github.com/user-attachments/assets/8e1598ca-a8ce-453b-aa0f-4c45859c8f95)

![Screenshot 2024-11-24 092002](https://github.com/user-attachments/assets/711b1710-fb34-412a-889f-c4292e68eef3)



![graph 1 ex 2](https://github.com/user-attachments/assets/730e8529-4ebc-42a4-8743-be390f4986a8)

![download](https://github.com/user-attachments/assets/e8ad6b8d-88c9-4012-8247-0ef23a37d795)




MSE =  4.691397441397438
MAE =  4.691397441397438
RMSE=  2.1659633979819324



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
