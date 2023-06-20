import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional, Concatenate, Dot, Activation
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

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

# Define the feature ranges
scaling_ranges = {
    'Eqp1 Load(%)': (10, 100),
    'Eqp1 Pressure1(bar)': (2.0, 3.0),
    'Eqp1 Pressure2(bar)': (7.0, 9.0),
    'Eqp1 Supply Of oil Pressure(bar) ': (7.0, 9.0),
    'Eqp1 oil filter variations Presssure(bar)': (None, 0.6),
    'Eqp1 Temp1(degree)': (5, 20),
    'Eqp1 Temp2(degree)': (50, 70),
    'Eqp1 oil temp sensor Oil level(mm)': (150, 200),
    'Eqp2 water flow(M/hr)': (90, 125),
    'Eqp2 Incoming Pressure(bar)': (-0.5, 1.5),
    'Eqp2 OutGoing Pressure(bar)': (2.0, 4.0),
    'Eqp2 Pressure(bar) Diff': (0.1, 0.1),
    'Eqp2 Incoming Temp(degree)': (10, 35),
    'Eqp2 Outgoiing Temp(degree)': (15, 40),
    'Eqp3 cold water flow (m/hr)': (75, 75),
    'Eqp3 Incoming Pressure(bar)': (-0.5, 1),
    'Eqp3 Outgoing Pressure(bar)': (4.0, 6.0),
    'Eqp3 Variation in Pressure(bar)': (0.3, None),
    'Eqp3 Incoming Temp(degree)': (12, 25),
    'Eqp3 Outgoing Temp(degree)': (6, 20),
    }

# Apply scaling ranges to the dataset
for column, (min_value, max_value) in scaling_ranges.items():
    if min_value is not None:
        df_for_training[column] = np.where(df_for_training[column] < min_value, min_value, df_for_training[column])
    if max_value is not None:
        df_for_training[column] = np.where(df_for_training[column] > max_value, max_value, df_for_training[column])

# Split the data into train and test sets based on the specified time range
train_data = df_for_training.loc[:'2021-10-14 17:00:00']
test_data = df_for_training.loc['2021-10-14 17:00:00':'2021-10-21 19:00:00']

# Normalize the data
scaler = MinMaxScaler()
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

# Define the number of time steps and features
n_steps = 26
n_features = df_for_training.shape[1]

# Prepare the training data
train_X, train_y = [], []
for i in range(n_steps, len(train_data_scaled)):
    train_X.append(train_data_scaled[i - n_steps:i])
    train_y.append(train_data_scaled[i, df_for_training.columns.get_loc('Failure')])
train_X, train_y = np.array(train_X), np.array(train_y)

# Prepare the test data
test_X, test_y = [], []
for i in range(n_steps, len(test_data_scaled)):
    test_X.append(test_data_scaled[i - n_steps:i])
    test_y.append(test_data_scaled[i, df_for_training.columns.get_loc('Failure')])
test_X, test_y = np.array(test_X), np.array(test_y)

# Define the architecture of the Bi-LSTM encoder-decoder model with attention
encoder_inputs = Input(shape=(n_steps, n_features))
encoder = Bidirectional(LSTM(64, return_sequences=True))(encoder_inputs)
decoder_inputs = Input(shape=(n_steps, n_features))
decoder = Bidirectional(LSTM(64, return_sequences=True))(decoder_inputs)
attention = Dot(axes=[2, 2])([decoder, encoder])
attention = Activation('softmax')(attention)
context = Dot(axes=[2, 1])([attention, encoder])
decoder_combined_context = Concatenate(axis=-1)([context, decoder])
decoder_outputs = Dense(1)(decoder_combined_context)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mae')

# Train the model
model.fit([train_X, train_X], train_y, epochs=10, batch_size=16)

# Make predictions on the test data
test_predictions = model.predict([test_X, test_X])
test_predictions.shape
test_predictions = test_predictions.reshape(34,26)
# Invert the scaling of the predicted values
test_predictions = scaler.inverse_transform(test_predictions)
# Enforce range constraint
test_predictions = np.clip(test_predictions, 0, 1)
test_predictions.shape
# Convert the predicted values to DataFrame
predicted_df = pd.DataFrame(test_predictions, columns=df_for_training.columns, index=test_data.index[n_steps:])

# Plot the predicted values
plt.figure(figsize=(10, 6))
plt.plot(predicted_df.index, predicted_df[target_variable], label='Predicted')

plt.xlabel('Date')
plt.ylabel(target_variable)
plt.title('Failure Prediction: Predicted')
plt.legend()
plt.show()
# Create a DataFrame for the actual values
actual_df = pd.DataFrame(test_y, columns=[target_variable], index=test_data.index[n_steps:])

# Plot the predicted and actual values
plt.figure(figsize=(10, 6))
plt.plot(predicted_df.index, predicted_df[target_variable], label='Predicted')
plt.plot(actual_df.index, actual_df[target_variable], label='Actual')

plt.xlabel('Date')
plt.ylabel(target_variable)
plt.title('Failure Prediction')
plt.legend()
plt.show()
from sklearn.metrics import accuracy_score

# Align the indices of test data and predicted values
test_data = test_data[n_steps:]
predicted_df = predicted_df[:len(test_data)]

# Convert the failure values to binary (0 or 1)
test_labels = test_data['Failure'].apply(lambda x: 1 if x >= 0.5 else 0)
predicted_labels = predicted_df['Failure'].apply(lambda x: 1 if x >= 0.5 else 0)

accuracy = accuracy_score(test_labels, predicted_labels)
accuracy
