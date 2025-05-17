import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

ds = load_dataset("lmsys/lmsys-chat-1m")

df = ds['train'].to_pandas()

filtered_df = df[(df['turn'] <= 10) & (df['language'] == 'English') & (df['redacted'] == False)]

filtered_df = filtered_df.head(55000)

train_df = filtered_df.head(50000)
test_df = filtered_df.tail(5000)

train_df.to_csv('train_lmsys_chat.csv', index=False)
test_df.to_csv('test_lmsys_chat.csv', index=False)
