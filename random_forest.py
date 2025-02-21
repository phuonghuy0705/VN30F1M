import argparse
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Sử dụng GPU 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Khởi tạo argparse
parser = argparse.ArgumentParser(description="Train a Random Forest model!")
parser.add_argument("--input", type=str, required=True, help="Path to input CSV file")
parser.add_argument("--output", type=str, required=True, help="Path to save trained model")
parser.add_argument("--seq_length", type=int, default=30, help="Sequence length for time series data")

args = parser.parse_args()

# Load dataset
df = pd.read_csv(args.input)

feature_columns = ["Open", "High", "Low", "Close", "Volume"]
target_column = "Label"

# Chia train/test
test_days = 30
test_size = test_days * 24 * 60
train_test_split = len(df) - test_size
train_df = df.iloc[:train_test_split]
test_df = df.iloc[train_test_split:]

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
train_df.loc[:, feature_columns] = scaler.fit_transform(train_df[feature_columns])
test_df.loc[:, feature_columns] = scaler.transform(test_df[feature_columns])

train_df["Label"] = train_df["Label"].fillna(0).astype(int)
test_df["Label"] = test_df["Label"].fillna(0).astype(int)

# Tạo Sliding Window
def create_sequences(data, labels, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(labels[i + seq_length])
    return np.array(X), np.array(y)

sequence_length = args.seq_length
X_train, y_train = create_sequences(train_df[feature_columns].values, train_df[target_column].values, sequence_length)
X_test, y_test = create_sequences(test_df[feature_columns].values, test_df[target_column].values, sequence_length)

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Vẽ iểu đồ phân phối nhãn
unique, counts = np.unique(y_train, return_counts=True)
plt.bar(unique, counts)
plt.xlabel("Labels")
plt.ylabel("Count")
plt.title("Label Distribution in Training Set")
plt.show()

# Huấn luyện mô hình Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Dự đoán trên tập test
y_pred_test = rf_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"Test Accuracy: {test_accuracy:.4f}")
print("Classification Report (Test):")
print(classification_report(y_test, y_pred_test))

# Vẽ Confusion matrix
cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Lưu mô hình
joblib.dump(rf_model, args.output)
print(f"Model saved as {args.output}")
