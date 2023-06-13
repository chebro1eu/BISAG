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

# Set the time ranges
range1_start = pd.Timestamp('2021-10-06 13:00:00')
range1_end = pd.Timestamp('2021-10-26 23:00:00')

range2_start = pd.Timestamp('2022-06-16 16:00:00')
range2_end = pd.Timestamp('2022-06-30 04:00:00')

# Loop through each column (parameter) in the DataFrame
for column in df2.columns:
    # Skip if the column is not numeric
    if not pd.api.types.is_numeric_dtype(df2[column]):
        continue
    
    # Filter the data based on the time ranges
    range1_data = df2.loc[(df2.index >= range1_start) & (df2.index <= range1_end)]
    range2_data = df2.loc[(df2.index >= range2_start) & (df2.index <= range2_end)]
    
    # Calculate mean and standard deviation for each range
    range1_mean = df2[column].mean()
    range1_std = df2[column].std()
    range2_mean = df2[column].mean()
    range2_std = df2[column].std()

    # Plot the line graph for Range 1
    plt.figure(figsize=(10, 6))
    plt.plot(range1_data.index, range1_data[column], color='blue', label='Data')
    plt.axhline(range1_mean, color='red', linestyle='--', label='Mean')
    plt.axhline(range1_mean + 2 * range1_std, color='green', linestyle='--', label='Mean + 2 Std Dev')
    plt.axhline(range1_mean - 2 * range1_std, color='green', linestyle='--', label='Mean - 2 Std Dev')
    plt.xlabel('Date and Time')
    plt.ylabel(column)
    plt.title(f'Data with Markers for Outliers - Range 1 - {column}')
    plt.legend()

    # Add markers for values higher than mean + 2 * standard deviation
    range1_outliers = range1_data[range1_data[column] > (range1_mean + 2 * range1_std)]
    plt.scatter(range1_outliers.index, range1_outliers[column], color='red', marker='o', label='Outliers')

    # Print the individual DateTime indexes for each point in Range 1
    for index, row in range1_outliers.iterrows():
        plt.text(index, row[column], index.strftime('%Y-%m-%d %H:%M:%S'), ha='center', va='bottom')

    plt.show()

    # Plot the line graph for Range 2
    plt.figure(figsize=(10, 6))
    plt.plot(range2_data.index, range2_data[column], color='blue', label='Data')
    plt.axhline(range2_mean, color='red', linestyle='--', label='Mean')
    plt.axhline(range2_mean + 2 * range2_std, color='green', linestyle='--', label='Mean + 2 Std Dev')
    plt.axhline(range2_mean - 2 * range2_std, color='green', linestyle='--', label='Mean - 2 Std Dev')
    plt.xlabel('Date and Time')
    plt.ylabel(column)
    plt.title(f'Data with Markers for Outliers - Range 2 - {column}')
    plt.legend()

    # Add markers for values higher than mean + 2 * standard deviation
    range2_outliers = range2_data[range2_data[column] > (range2_mean + 2 * range2_std)]
    plt.scatter(range2_outliers.index, range2_outliers[column], color='red', marker='o', label='Outliers')

    # Print the individual DateTime indexes for each point in Range 2
    for index, row in range2_outliers.iterrows():
        plt.text(index, row[column], index.strftime('%Y-%m-%d %H:%M:%S'), ha='center', va='bottom')

    plt.show()
