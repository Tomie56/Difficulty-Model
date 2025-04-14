import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
import torch.nn as nn
from tqdm import tqdm

df = pd.read_csv('../datas/train_lmsys_chat_with_rewards_cleaned.csv')
df = df[['conversation', 'reward']]
df = df.dropna(subset=['reward'])

embeddings = np.load('../datas/embeddings.npy') 
embeddings = torch.tensor(embeddings, dtype=torch.float32) 

y = torch.tensor(df['reward'].values, dtype=torch.float32)

train_dataset = TensorDataset(embeddings, y)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)

print("定义MLP模型...")
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

input_size = embeddings.shape[1]
model_mlp = MLPModel(input_size)
print("MLP模型定义完成。")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_mlp.to(device)

print("定义损失函数和优化器...")
criterion = nn.MSELoss()  # MSELoss
optimizer = torch.optim.Adam(model_mlp.parameters(), lr=1e-4)
print("损失函数和优化器定义完成。")

print("开始训练模型...")
epochs = 50
for epoch in range(epochs):
    model_mlp.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', ncols=100)):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model_mlp(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader)}")

torch.save(model_mlp.state_dict(), '../models/mlp_reward_predictor.pth')
print("模型已保存！")
