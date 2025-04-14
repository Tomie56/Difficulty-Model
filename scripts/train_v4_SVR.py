import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import joblib

df_train = pd.read_csv('../datas/train_lmsys_chat_with_rewards_cleaned.csv')
df_train = df_train[['conversation', 'reward']].dropna(subset=['reward'])
train_embeddings = np.load('../datas/embeddings.npy')

X_train = train_embeddings 
y_train = df_train['reward'].values 


svr_model = SVR()

# 使用 GridSearchCV 来调优超参数
param_grid = {
    'C': [1, 10, 100],  # 惩罚参数
    'epsilon': [0.01, 0.1, 0.5],  # 控制支持向量的精度
    'kernel': ['rbf'],  # 核函数
}

grid_search = GridSearchCV(svr_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
print("Best parameters found by GridSearchCV:", grid_search.best_params_)


best_svr_model = grid_search.best_estimator_


joblib.dump(best_svr_model, '../models/svr_reward_predictor.pkl')
print("Model saved to '../models/svr_reward_predictor.pkl'")
