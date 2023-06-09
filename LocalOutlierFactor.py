from sklearn.neighbors import LocalOutlierFactor
failure_data = r'C:\Users\Niranjan\Desktop\BISAG\machine failed\cleaned_data.csv'
dataset = pd.read_csv(failure_data)
df = dataset
df2 = pd.DataFrame(df)
df2["Date"] = df2['Date'].astype(str) + "  " + df['Time'].astype(str)
df2['Date']= pd.to_datetime(df['Date'])
df2 = df2.drop(['Time'],axis = 1)
df2 = df2.set_index("Date")
data = df2

# Train the Local Outlier Factor
lof = LocalOutlierFactor(novelty=True)
lof.fit(data)

# Predict anomaly scores for the entire dataset
scores = -lof.decision_function(data)

# Determine the threshold for anomaly detection
threshold = scores.mean() + 0.0005 * scores.std()  # Example: Threshold using mean + 3 standard deviations

# Classify data points as normal or anomalous
anomalies = scores > threshold

# Print the anomalies found
print("Anomalies found:")
for i, anomaly in enumerate(anomalies):
    if anomaly:
        original_data_point = data.iloc[i]  # Retrieve the original data point
        print(f"Data point {i} is an anomaly:")
