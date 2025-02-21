import pandas as pd
from sklearn.model_selection import train_test_split

# Sample dataset loading
df = pd.read_csv("dataset.csv")  # Ensure you have a dataset
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Dataset loaded and split successfully!")
