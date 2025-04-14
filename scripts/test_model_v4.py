import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

df_test = pd.read_csv('../datas/test_lmsys_chat_with_rewards_cleaned.csv')
df_test = df_test[['conversation', 'reward']].dropna(subset=['reward'])
test_embeddings = np.load('../datas/embeddings_test.npy')


X_test = test_embeddings 
y_test = df_test['reward'].values 


svr_model = joblib.load('../models/svr_reward_predictor_with_tuning.pkl')
y_pred = svr_model.predict(X_test)


mse_test = mean_squared_error(y_test, y_pred)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test, y_pred)
r2_test = r2_score(y_test, y_pred)

print(f"SVR Test MSE: {mse_test}")
print(f"SVR Test RMSE: {rmse_test}")
print(f"SVR Test MAE: {mae_test}")
print(f"SVR Test R²: {r2_test}")


print("Printing sample predictions vs true values...")
for i in range(10):
    print(f"Predicted: {y_pred[i]}, True: {y_test[i]}")