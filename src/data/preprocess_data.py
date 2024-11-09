import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
project_root = os.getenv("PROJECT_ROOT")

# Define paths
data_dir = os.path.join(project_root, 'data')
processed_data_dir = os.path.join(data_dir, 'processed')
os.makedirs(processed_data_dir, exist_ok=True)

# Load the Iris dataset
raw_dir = os.path.join(data_dir, 'raw')
iris = pd.read_csv(os.path.join(raw_dir, 'iris.csv'))
X = iris.drop('class', axis=1)
y = iris['class']

# Initialize and fit the scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler for future use
scaler_path = os.path.join(processed_data_dir, 'scaler.pkl')
joblib.dump(scaler, scaler_path)

# Combine scaled features and target
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
processed_df = pd.concat([X_scaled_df, y], axis=1)

# Save the processed data
processed_data_path = os.path.join(processed_data_dir, 'processed_iris.csv')
processed_df.to_csv(processed_data_path, index=False)

print(f"Processed data saved at {processed_data_path}")
print(f"Scaler saved at {scaler_path}")
