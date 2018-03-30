# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 07:44:35 2018

@author: JEPRSDD
"""

#STEP - 1 -DATA PREPROCESSING

#data preprocessing template


#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
#here X is the matrix of features i.e., the matrix of independent variable = years of experiecnec
X = dataset.iloc[:, :-1].values
#y is the vector of the dependent variable
y = dataset.iloc[:, 1].values

#splitting the dataset into the training and test test
#since we have 30 observations a good split is about train set = 20 and test set = 10(remember 80-20 split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 1/3, random_state = 0)

#the model learns from the train set and we test the performance of that model on the test set

#q) these two will be corresponding or not ??? --- correspondingly
#q)how thesse are splitted into train_set and test_set randomly or any algo is behind this??



#feature scaling -- here we are not worried about the feature scaling because we have library for that
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""


#STEP - 2: FITTING THE SIMPLE LINEAR REGRESSION MODEL TO THE TRAINING SET

#we are fitting the model by using the library

#importing the linear regression class
from sklearn.linear_model import LinearRegression
#creating the object to the linearregressor class naming as regressor
regressor = LinearRegression()
#fitting the fit method to the regressor object to the training set
regressor.fit(X_train, y_train)

#note: here the machine is simple regression model and learning is achine is learning on the training dataset.

#STEP - 3
#predicting the test set results i.e.,by now the machine learned from the training set and we will amke predictions in the test dataset to estimate the performance of the prediction by comparing the actual 
#we will have a vector named y_pred which will contain all the predcted values of the test dataset
#here in this we will have the redicted values as salaries
y_pred =  regressor.predict(X_test)
#predict is a method in the class LinearRegression
#the input of the predict test set
#now compare the y_test and y_pred and see how much they are related

#STEP - 4
#visulaing the training dataset resultsa and test results.
# x axis = no of years of experience
#y axis = salaries
plt.scatter(X_train,  y_train, color = 'red' )
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
#observation point(actual) in red and regrssion points(predicted) in blue
plt.title('Years of Experience VS Salaries -- Training Set Results')
plt.xlabel('Years of Experience')
plt.ylabel('Salaries')
plt.show()

#the most important distinction is that we need to make distinction between the real values nad the predicted values -- predicted values are on the blue regression line.
#this have linear dependency since the salaries are increasing with the years of experince
#what we did here is that we took only training set and plotted a grpah where the machine learnt from the training dataset

#now, we will do same thing on the test dataset where the machine didnt learn but makes prodictions
plt.scatter(X_test,  y_test, color = 'red' )
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
#we will not change the X_train since regressor already trained on the training set or replaced by the test results we will get the same simple regression line
#observation point(actual) in red and regrssion points(predicted) in blue
plt.title('Years of Experience VS Salaries -- Test Set Results')
plt.xlabel('Years of Experience')
plt.ylabel('Salaries')
plt.show()

