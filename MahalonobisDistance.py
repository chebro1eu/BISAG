import pandas as pd 
import numpy as np 
import math as mt
from sklearn.model_selection import train_test_split

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

from scipy.spatial.distance import mahalanobis
# Calculate the Mahalanobis distance
cov_matrix = np.cov(data.T)  # Calculate the covariance matrix

# Regularization to handle singular matrix
reg_cov_matrix = cov_matrix + 1e-8 * np.eye(cov_matrix.shape[0])

# Calculate the inverse of the covariance matrix
inv_cov_matrix = np.linalg.inv(reg_cov_matrix)

# Calculate the Mahalanobis distance for each data point
mahalanobis_dist = []
mean_vector = data.mean().values  # Calculate the mean vector
for i in range(len(data)):
    data_point = data.iloc[i].values
    dist = mahalanobis(data_point, mean_vector, inv_cov_matrix)
    mahalanobis_dist.append(dist)

# Determine the threshold for anomaly detection
threshold = np.mean(mahalanobis_dist) +  0.73*np.std(mahalanobis_dist)  # Example: Threshold using mean + 3 standard deviations

# Classify data points as normal or anomalous
anomalies = np.array(mahalanobis_dist) > threshold

print("Anomalies found:")
for i, anomaly in enumerate(anomalies):
    if anomaly:
        original_data_point = data.iloc[i]  # Retrieve the original data point
        index_value = data.index[i]  # Retrieve the index value
        print(f"Data point {index_value} is an anomaly:")
