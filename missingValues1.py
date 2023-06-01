import numpy as np
import pandas as pd
import math as mt
from sklearn.model_selection import train_test_split
melb_data = r'C:\Users\Niranjan\Desktop\Dev\melb_data.csv'
data = pd.read_csv(melb_data)

# select a target
y = data.Price

#Numerical Predictors
predicting_data = data.drop(['Price'], axis =1)
X = predicting_data.select_dtypes(exclude=['object'])

#Dividing data into train and test subsets

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=123)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

#Function for comparing different approaches

def score_dataset(X_train,X_test,y_train,y_test):
    model = RandomForestRegressor(n_estimators = 10, random_state = 0)
    model.fit(X_train,y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test,preds)

#Get names of columns with missing values

cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]

#Drop columns in training and testing data
reduced_X_train = X_train.drop(cols_with_missing, axis =1)
reduced_X_test = X_test.drop(cols_with_missing,axis =1)

print("MAE from approach 1 (DROP COLUMNS WITH MISSING VALUES) :")
print(score_dataset(reduced_X_train,reduced_X_test,y_train,y_test))
