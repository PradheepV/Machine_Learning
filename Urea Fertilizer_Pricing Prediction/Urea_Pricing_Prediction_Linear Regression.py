# -*- coding: utf-8 -*-
"""
Urea Price Prediction based on Consumption using Linear Regression Algorithm

Author: Pradheep Vepur
Created: 11/15/2019 11:05AM
"""
#Importing necessary modules
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Read the excel file and drop unnecessary columns
fert_use = pd.read_excel("Urea_Consumption_Price.xlsx")
fert_use.drop("Unnamed: 0",axis=1,inplace=True)
fert_use.info()

#Setting the Independent Variable X and dependent variable y
X = fert_use.iloc[:,:1].values.reshape(-1,1)
y = fert_use.iloc[:,1].values.reshape(-1,1)
x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.2,random_state=1)
lr = LinearRegression()
lr.fit(x_train,y_train)
y_predict = lr.predict(x_test)

#Storing Intercept in c, Coefficien in variable m (representing the linear equation: y=mx+c)
c = lr.intercept_
m = lr.coef_
Urea_act_pred = pd.DataFrame({"Actual":y_test.flatten(),"Predicted":y_predict.flatten()})

#Checking the fitted line in the training set
plt.scatter(x_train, y_train,color = "blue")
plt.plot(x_train,lr.predict(x_train),color="red")
plt.title("Training Data set Fitted Line")
plt.show()

#Checking the fitted line in the test set with prediction
plt.scatter(x_test, y_test,color = "blue")
plt.plot(x_test,y_predict,color="red")
plt.title("Predicted Data set Fitted Line")

#Bar plot of Actual vs Prediction
Urea_act_pred.plot(kind="bar")
plt.title("Bar plot of Actual vs Prediction")
plt.show()