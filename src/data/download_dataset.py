import os
from dotenv import load_dotenv
import pandas as pd
from sklearn.datasets import load_iris

# Get project root
load_dotenv()
root_dir = os.getenv("PROJECT_ROOT")
os.makedirs(os.path.join(root_dir, "data", "raw"), exist_ok=True)

# Load Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(iris.data)
iris_df['class'] = iris.target
iris_df.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
iris_df.dropna(how="all", inplace=True)
target_dir = os.path.join(root_dir, "data", "raw", "iris.csv")
iris_df.to_csv(target_dir, index=False)

print(f"Iris dataset saved to {target_dir}")
