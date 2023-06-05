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

