import pandas as pd
import os
import sklearn.metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATASET_PATH = "."
if os.path.exists("/data/mlproject22"):
    DATASET_PATH = "/data/mlproject22"

powerpredict_df = pd.read_csv(os.path.join(DATASET_PATH, "powerpredict.csv.zip"))

# preprocessing
powerpredict_df = powerpredict_df.dropna()
categorical_cols = powerpredict_df.select_dtypes(include=['object']).columns
encoded_df = pd.get_dummies(powerpredict_df, columns=categorical_cols) # converts categorical variables into binary columns
X = encoded_df.drop(columns=["power_consumption"])
y = encoded_df[["power_consumption"]]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features for better model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# linear regression
regr = LinearRegression()
regr.fit(X_train_scaled, y_train)

# Make predictions on the training and testing data
y_train_predicted = regr.predict(X_train_scaled)
y_test_predicted = regr.predict(X_test_scaled)

# Calculate the mean absolute error
train_mae = mean_absolute_error(y_train, y_train_predicted)
test_mae = mean_absolute_error(y_test, y_test_predicted)

print("Train Dataset Score:", train_mae)
print("Test Dataset Score:", test_mae)
