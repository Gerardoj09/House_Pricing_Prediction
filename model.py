#!/usr/bin/env python
# coding: utf-8


import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


# Load the df_ETL.csv dataset.
file_path = "df_ETL.csv"
df_ETL = pd.read_csv(file_path )
df_ETL.head(10)

#using pandas get_dummies function to transform the categorical values into numerical 
df_dummies = pd.get_dummies(df_ETL,columns=["MSZoning","LotFrontage","Street","LotShape","LandContour","Utilities","LotConfig","LandSlope","Neighborhood","Condition1","Condition2","BldgType","HouseStyle","RoofStyle","RoofMatl","Exterior1st","Exterior2nd","MasVnrType","ExterQual","ExterCond","Foundation","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","Heating","HeatingQC","CentralAir","Electrical","KitchenQual","Functional","FireplaceQu","GarageType","GarageFinish","GarageQual","GarageCond","PavedDrive","Fence","SaleType","SaleCondition"])

df_dummies.head()

#get the SalesPrice column
df_dummies.columns.get_loc("SalePrice")

#make an array for the first ten columns and SalesPrice column
#selected_cols=np.r_[1:10, 33]

# df_train.iloc[rows:rows, cols:cols]
first_cols = df_dummies.iloc[:,1:6]
first_cols.head()

first_cols.index.name = "_id"

#export first columns to csv
#first_cols.to_csv('first_cols.csv')

# #### Define y target variable and Define X features variables

y_train = df_dummies["SalePrice"]
X_train = first_cols

# Load the df_ETL.csv dataset.
file_path = "test.csv"
test_df = pd.read_csv(file_path)
test_df.head(10)

#using pandas get_dummies function to transform the categorical values into numerical 
#get the SalesPrice column
X_test = test_df[["LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd"]]
X_test.head()

model = LinearRegression()

model.fit(X_train, y_train)

y_prediction = model.predict(X_test)
y_prediction


from sklearn.metrics import r2_score


df_test =pd.read_csv('sample_submission.csv')
y_test = df_test['SalePrice']


r2_score(y_test, y_prediction)