import pandas as pd 
import numpy as np 
import math as mt
from sklearn.model_selection import train_test_split

failure_data = r'C:\Users\Niranjan\Desktop\BISAG\machine failed\cleaned_data.csv'
dataset = pd.read_csv(failure_data)
df = dataset
df.shape
df2 = pd.DataFrame(df)
df2["Date"] = df2['Date'].astype(str) + "  " + df['Time'].astype(str)
df2['Date']= pd.to_datetime(df['Date'])
df3 = df2.set_index("Date")
df2 = df2.drop(['Time'],axis = 1)
df2 = df2.set_index("Date")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df2)
X_train, X_test = train_test_split(scaled_data,test_size=0.2)

input_dim = X_train.shape[1]
encoding_dim = 50
autoencoder = Sequential()
autoencoder.add(Dense(encoding_dim, activation='relu', input_shape=(input_dim,)))
autoencoder.add(Dense(input_dim, activation='sigmoid'))

# Compile and train the autoencoder
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_test, X_test))

# Calculate the reconstruction errors
reconstructions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)

# Determine the threshold for anomaly detection
threshold = np.mean(mse) + 3 * np.std(mse)

# Classify data points as normal or anomalous
anomalies = mse > threshold

# Display the anomalies using a scatter plot
plt.scatter(range(len(mse)), mse, c=anomalies, cmap='coolwarm')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.xlabel('Data Points')
plt.ylabel('Reconstruction Error')
plt.title('Anomaly Detection with Autoencoder')
plt.legend()
plt.show()
