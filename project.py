import pandas as pd
import numpy as np
import math as mt
from sklearn.model_selection import train_test_split

failure_data = r'C:\Users\Niranjan\Desktop\BISAG\machine failed\DATA1.csv'
dataset = pd.read_csv(failure_data)
df = dataset

df2 = df.iloc[:,:23]
print(df2.shape)
print(df2.isnull().sum())

#Hot deck imputation being run
df3 = pd.DataFrame(df2)

#Function to do Hot deck imputation
# Function to perform hot deck imputation
def hot_deck_imputation(df):
    for col in df.columns:
        missing_indices = df[col].isnull()  # Get indices of missing values
        
        for i in missing_indices[missing_indices].index:
            non_missing_vals = pd.to_numeric(df.loc[~missing_indices, col])  # Convert to numeric
            
            # Find the nearest observed value to the missing value
            nearest_val = non_missing_vals.iloc[(non_missing_vals - pd.to_numeric(df.loc[i, col])).abs().argsort()[0]]
            
            # Replace the missing value with the nearest observed value
            df.loc[i, col] = nearest_val
    
    return df

# Perform hot deck imputation

imputed_df = hot_deck_imputation(df3)
print(imputed_df)

imputed_df.to_csv(r'C:\Users\Niranjan\Desktop\BISAG\machine failed\cleaned_data.csv', index = False)
imputed_df.to_excel(r'C:\Users\Niranjan\Desktop\BISAG\machine failed\cleaned_data.xlsx', index = False)
