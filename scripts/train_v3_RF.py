import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# -------------------------------
# 配置路径与参数
# -------------------------------
data_path = '../datas/train_lmsys_chat_with_rewards_cleaned.csv'
embedding_path = '../datas/embeddings.npy'
model_path = '../models/rf_reward_predictor.pkl'
n_estimators = 200

# -------------------------------
# 加载数据与嵌入
# -------------------------------
print("Loading data and embeddings...")

if not os.path.exists(data_path) or not os.path.exists(embedding_path):
    raise FileNotFoundError("Input CSV or embedding file not found.")

df = pd.read_csv(data_path)
df = df[['conversation', 'reward']].dropna(subset=['reward'])

embeddings = np.load(embedding_path)
X_train = embeddings
y_train = df['reward'].values

print(f"Data loaded. Training samples: {len(y_train)}")

# -------------------------------
# 训练随机森林模型
# -------------------------------
print("Training Random Forest Regressor...")
rf_model = RandomForestRegressor(
    n_estimators=n_estimators,
    random_state=42,
    n_jobs=-1,  # 使用所有可用CPU核心
    verbose=1
)
rf_model.fit(X_train, y_train)

# -------------------------------
# 保存模型
# -------------------------------
joblib.dump(rf_model, model_path)
print(f"Model saved to: {model_path}")
