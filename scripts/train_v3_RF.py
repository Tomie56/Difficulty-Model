import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.read_csv('../datas/train_lmsys_chat_with_rewards_cleaned.csv')
df = df[['conversation', 'reward']].dropna(subset=['reward'])
embeddings = np.load('../datas/embeddings.npy')


X_train = embeddings
y_train = df['reward'].values

print("开始训练随机森林模型...")
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

joblib.dump(rf_model, '../models/rf_reward_predictor.pkl')

print("Random Forest model trained on the training set.")
