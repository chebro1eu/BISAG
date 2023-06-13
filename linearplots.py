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

# Loop through each column (parameter) in the DataFrame
for column in df2.columns:
    # Skip if the column is not numeric
    if not pd.api.types.is_numeric_dtype(df2[column]):
        continue
        
    # Calculate mean and standard deviation
    mean = df2[column].mean()
    std = df2[column].std()

    # Plot the line graph
    plt.figure(figsize=(10, 6))
    plt.plot(df2.index, df2[column], color='blue', label='Data')
    plt.axhline(mean, color='red', linestyle='--', label='Mean')
    plt.axhline(mean + 2 * std, color='green', linestyle='--', label='Mean + 2 Std Dev')
    plt.xlabel('Date and Time')
    plt.ylabel(column)
    plt.title(f'Data with Markers for Outliers - {column}')
    plt.legend()

    # Add markers for values higher than mean + 2 * standard deviation
    outliers = df2[df2[column] > (mean + 2 * std)]
    plt.scatter(outliers.index, outliers[column], color='red', marker='o', label='Outliers')

    # Print the individual DateTime indexes for each point
    for index, row in outliers.iterrows():
        plt.text(index, row[column], index.strftime('%Y-%m-%d %H:%M:%S'), ha='center', va='bottom')

    plt.show()
