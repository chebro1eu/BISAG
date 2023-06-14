import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam

# Load and preprocess the dataset
failure_data = r'C:\Users\Niranjan\Desktop\BISAG\machine failed\cleaned_data.csv'
dataset = pd.read_csv(failure_data)
df = dataset
df2 = pd.DataFrame(df)
df2["Date"] = df2['Date'].astype(str) + "  " + df2['Time'].astype(str)
df2['Date'] = pd.to_datetime(df2['Date'], format='%d-%m-%Y %H:%M:%S')
df2 = df2.drop(['Time'], axis=1)
df2 = df2.set_index("Date")
df_for_training = df2.astype(float)
target_variable = 'Failure'

# Split the data into train and test sets based on the specified time range
train_data = df_for_training.loc[:'2022-06-30 07:00:00']
test_data = df_for_training.loc['2022-06-30 07:00:00':'2022-08-20 19:00:00']

# Normalize the data
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

# Define the number of time steps and features
n_steps = 12
n_features = df_for_training.shape[1]

# Prepare the training data
train_X, train_y = [], []
for i in range(n_steps, len(train_data_scaled)):
    train_X.append(train_data_scaled[i - n_steps:i])
    train_y.append(train_data_scaled[i, df_for_training.columns.get_loc(target_variable)])
train_X, train_y = np.array(train_X), np.array(train_y)

# Prepare the test data
test_X, test_y = [], []
for i in range(n_steps, len(test_data_scaled)):
    test_X.append(test_data_scaled[i - n_steps:i])
    test_y.append(test_data_scaled[i, df_for_training.columns.get_loc(target_variable)])
test_X, test_y = np.array(test_X), np.array(test_y)

# Define and train the LSTM model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(n_features))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
model.fit(train_X, train_y, epochs=10, batch_size=16)

# Make predictions on the test data
test_predictions = model.predict(test_X)

# Reshape test_predictions to match the shape of the original scaled data
test_predictions = test_predictions.reshape(-1, n_features)

# Inverse scaling
test_predictions = scaler.inverse_transform(test_predictions)

# Convert the predictions and ground truth to pandas DataFrame for visualization
predicted_df = pd.DataFrame(test_predictions, index=test_data.index[n_steps:n_steps + len(test_predictions)], columns=df_for_training.columns)
actual_df = pd.DataFrame(test_y, index=test_data.index[n_steps:], columns=[target_variable])

# Plot the predicted and actual values
plt.figure(figsize=(10, 6))
plt.plot(predicted_df.index, predicted_df[target_variable], label='Predicted')
plt.plot(actual_df.index, actual_df[target_variable], label='Actual')
plt.xlabel('Date')
plt.ylabel(target_variable)
plt.title('Failure Prediction')
plt.legend()
plt.show()
