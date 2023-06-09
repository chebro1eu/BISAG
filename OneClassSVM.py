from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
failure_data = r'C:\Users\Niranjan\Desktop\BISAG\machine failed\cleaned_data.csv'
dataset = pd.read_csv(failure_data)
df = dataset
df2 = pd.DataFrame(df)
df2["Date"] = df2['Date'].astype(str) + "  " + df['Time'].astype(str)
df2['Date']= pd.to_datetime(df['Date'])
df2 = df2.drop(['Time'],axis = 1)
df2 = df2.set_index("Date")

# Train the One-Class SVM
svm = OneClassSVM()
svm.fit(df2)

# Predict anomaly scores for the entire dataset
scores = svm.score_samples(df2)

# Determine the threshold for anomaly detection
threshold = scores.mean() - 5* scores.std()  # Example: Threshold using mean - 3 standard deviations

# Classify data points as normal or anomalous
anomalies = scores < threshold

# Print the anomalies found
print("Anomalies found:")
for i, anomaly in enumerate(anomalies):
    if anomaly:
        original_data_point = df2.iloc[i]  # Retrieve the original data point
        print(f"Data point {i} is an anomaly:")
        
