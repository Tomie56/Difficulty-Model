import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# 创建结果保存目录
results_dir = '../results/resource_allocation'
os.makedirs(results_dir, exist_ok=True)

y_true = ['Hard', 'Hard', 'Easy', 'Easy', 'Hard', 'Easy', 'Hard', 'Easy', 'Hard', 'Easy',
          'Hard', 'Hard', 'Easy', 'Easy', 'Easy', 'Hard', 'Easy', 'Hard', 'Easy', 'Hard',
          'Easy', 'Easy', 'Hard', 'Hard', 'Easy', 'Hard', 'Hard', 'Easy', 'Easy', 'Easy',
          'Hard', 'Hard', 'Easy', 'Hard', 'Easy', 'Easy', 'Hard', 'Easy', 'Hard', 'Easy',
          'Easy', 'Easy', 'Hard', 'Hard', 'Easy', 'Hard', 'Hard', 'Easy', 'Easy', 'Hard']

y_pred = ['Hard', 'Easy', 'Hard', 'Hard', 'Hard', 'Easy', 'Hard', 'Easy', 'Hard', 'Easy',
          'Hard', 'Hard', 'Easy', 'Easy', 'Easy', 'Hard', 'Easy', 'Hard', 'Easy', 'Easy',
          'Hard', 'Easy', 'Hard', 'Hard', 'Hard', 'Hard', 'Easy', 'Easy', 'Easy', 'Easy',
          'Hard', 'Hard', 'Easy', 'Hard', 'Hard', 'Easy', 'Hard', 'Hard', 'Easy', 'Easy',
          'Easy', 'Easy', 'Easy', 'Easy', 'Easy', 'Hard', 'Easy', 'Easy', 'Easy', 'Easy']

labels = ['Easy', 'Hard']  # 顺序统一

# ===========================
# 混淆矩阵可视化
# ===========================
cm = confusion_matrix(y_true, y_pred, labels=labels)
df_cm = pd.DataFrame(cm, index=labels, columns=labels)

plt.figure(figsize=(6, 5))
sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Human Judged')
plt.title('Confusion Matrix: Model Prediction vs Human Judgement')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'confusion_matrix_human_validation.png'))
plt.close()

# ===========================
# 分类评估指标计算并保存
# ===========================
report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(os.path.join(results_dir, 'human_validation_metrics.csv'))
