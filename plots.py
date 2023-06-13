import pandas as pd 
import numpy as np 
import math as mt
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
failure_data = r'C:\Users\Niranjan\Desktop\BISAG\machine failed\cleaned_data.csv'
dataset = pd.read_csv(failure_data)
df = dataset
df2 = pd.DataFrame(df)
df2["Date"] = df2['Date'].astype(str) + "  " + df2['Time'].astype(str)
df2['Date']= pd.to_datetime(df2['Date'], format='%d-%m-%Y %H:%M:%S')
df2 = df2.drop(['Time'],axis = 1)
df3=df2
df2 = df2.set_index("Date")
data = df2

import matplotlib.pyplot as plt

# Set the time ranges
range1_start = pd.to_datetime('2021-10-06 13:00:00')
range1_end = pd.to_datetime('2021-10-26 23:00:00')

range2_start = pd.to_datetime('2022-06-16 16:00:00')
range2_end = pd.to_datetime('2022-06-30 04:00:00')

# Filter the data based on the time ranges
range1_data = df2.loc[(df2.index >= range1_start) & (df2.index <= range1_end)]
range2_data = df2.loc[(df2.index >= range2_start) & (df2.index <= range2_end)]

# Loop through each column (parameter) in the DataFrame
for column in df2.columns:
    # Skip if the column is not numeric
    if not pd.api.types.is_numeric_dtype(df2[column]):
        continue
    
    # Create a scatter plot for the parameter in Range 1
    plt.figure(figsize=(10, 6))
    plt.scatter(range1_data.index, range1_data[column], color='blue')
    plt.xlabel('Date and Time')
    plt.ylabel(column)
    plt.title(f'{column} Data - Range 1')
    plt.show()
    
    # Create a scatter plot for the parameter in Range 2
    plt.figure(figsize=(10, 6))
    plt.scatter(range2_data.index, range2_data[column], color='red')
    plt.xlabel('Date and Time')
    plt.ylabel(column)
    plt.title(f'{column} Data - Range 2')
    plt.show()
