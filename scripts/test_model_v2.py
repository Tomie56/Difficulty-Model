import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size=512, output_size=1):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x


df_test = pd.read_csv('../datas/test_lmsys_chat_with_rewards_cleaned.csv')
df_test = df_test[['conversation', 'reward']] 
df_test = df_test.dropna(subset=['reward'])
print(f"Test data loaded. Shape: {df_test.shape}")


embeddings_test = np.load('../datas/embeddings_test.npy')
embeddings_test = torch.tensor(embeddings_test, dtype=torch.float32)
y_test = torch.tensor(df_test['reward'].values, dtype=torch.float32) 

print(f"Embeddings shape: {embeddings_test.shape}, y_test shape: {y_test.shape}")


test_dataset = TensorDataset(embeddings_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

print("Test data loader created.")


print("Loading the trained model...")
input_size = embeddings_test.shape[1] 
model_mlp = MLPModel(input_size)
model_mlp.load_state_dict(torch.load('../models/mlp_reward_predictor_v2.pth'))
model_mlp.eval() 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_mlp.to(device)


print("Evaluating the model on the test set...")
predictions = []
true_values = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model_mlp(inputs)
        predictions.append(outputs.cpu().numpy())
        true_values.append(labels.cpu().numpy())

predictions = np.concatenate(predictions, axis=0)
true_values = np.concatenate(true_values, axis=0)

print("Model evaluation complete. Predictions and true values collected.")


print("Calculating evaluation metrics...")
# 均方误差 (MSE)
mse = mean_squared_error(true_values, predictions)
print(f"Test MSE: {mse}")

# 均方根误差 (RMSE)
rmse = np.sqrt(mse)
print(f"Test RMSE: {rmse}")

# 平均绝对误差 (MAE)
mae = mean_absolute_error(true_values, predictions)
print(f"Test MAE: {mae}")

# R2决定系数
r2 = r2_score(true_values, predictions)
print(f"Test R²: {r2}")


print("Printing sample predictions vs true values...")
for i in range(10):  
    print(f"Predicted: {predictions[i]}, True: {true_values[i]}")
