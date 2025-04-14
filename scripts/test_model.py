import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import torch.nn as nn

# Step 1: 定义 MLP 模型
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=1):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

print("Step 1: Model definition complete.")

# Step 2: 加载测试集数据
print("Step 2: Loading test data...")
df_test = pd.read_csv('../datas/test_lmsys_chat_with_rewards_cleaned.csv')  # 加载测试集
df_test = df_test[['conversation', 'reward']]  # 仅使用需要的列
df_test = df_test.dropna(subset=['reward'])  # 丢弃 reward 为 NaN 的行
print(f"Test data loaded. Shape: {df_test.shape}")

print("Loading saved embeddings for test set...")
embeddings_test = np.load('../datas/embeddings_test.npy')  # 直接加载保存的测试集隐藏状态
embeddings_test = torch.tensor(embeddings_test, dtype=torch.float32)  # 转换为 torch 张量
y_test = torch.tensor(df_test['reward'].values, dtype=torch.float32)  # 转换为张量

print(f"Embeddings shape: {embeddings_test.shape}, y_test shape: {y_test.shape}")

# 创建 DataLoader
test_dataset = TensorDataset(embeddings_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

print("Test data loader created.")

# Step 3: 加载训练好的模型
print("Step 3: Loading the trained model...")
input_size = embeddings_test.shape[1]  # 输入特征的大小
model_mlp = MLPModel(input_size)
model_mlp.load_state_dict(torch.load('../models/mlp_reward_predictor.pth'))  # 加载模型权重
model_mlp.eval()  # 设置为评估模式

# 将模型移动到 GPU（如果有的话）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_mlp.to(device)

print("Model loaded and moved to device.")

# Step 4: 评估模型
print("Step 4: Evaluating the model on the test set...")
predictions = []
true_values = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model_mlp(inputs)
        predictions.append(outputs.cpu().numpy())
        true_values.append(labels.cpu().numpy())

# 将预测结果和真实标签转换为 numpy 数组
predictions = np.concatenate(predictions, axis=0)
true_values = np.concatenate(true_values, axis=0)

print("Model evaluation complete. Predictions and true values collected.")

# Step 5: 计算并输出评估指标
print("Step 5: Calculating evaluation metrics...")
# 均方误差 (MSE)
mse = mean_squared_error(true_values, predictions)
print(f"Test MSE: {mse}")

# 均方根误差 (RMSE)
rmse = np.sqrt(mse)
print(f"Test RMSE: {rmse}")

# 平均绝对误差 (MAE)
mae = mean_absolute_error(true_values, predictions)
print(f"Test MAE: {mae}")

# R²决定系数
r2 = r2_score(true_values, predictions)
print(f"Test R²: {r2}")

# Step 6: 打印一部分预测值和真实值进行对比
print("Step 6: Printing sample predictions vs true values...")
for i in range(10):  # 打印前10个预测和真实值
    print(f"Predicted: {predictions[i]}, True: {true_values[i]}")
