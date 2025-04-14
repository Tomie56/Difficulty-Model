import pandas as pd
from transformers import AutoTokenizer, pipeline
import torch
import ast

rm_tokenizer = AutoTokenizer.from_pretrained("sfairXC/FsfairX-LLaMA3-RM-v0.1")
device = 0 

rm_pipe = pipeline(
    "sentiment-analysis",
    model="sfairXC/FsfairX-LLaMA3-RM-v0.1",
    device=device,
    tokenizer=rm_tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16}
)

pipe_kwargs = {
    "top_k": 1,
    "function_to_apply": "none",
    "batch_size": 1
}

df = pd.read_csv('../datas/train_lmsys_chat_processed_new.csv')

def ensure_conversation_format(conversation):
    try:
        if isinstance(conversation, str):
            conversation = ast.literal_eval(conversation)
        return conversation
    except Exception as e:
        print(f"Error ensuring conversation format: {e}")
        return None


def get_reward_for_conversation(conversation, idx):
    conversation = ensure_conversation_format(conversation)
    if conversation is None:
        return None

    try:
        test_texts = [rm_tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False).replace(rm_tokenizer.bos_token, "")]
        pipe_outputs = rm_pipe(test_texts, **pipe_kwargs)
        
        rewards = [output[0]["score"] for output in pipe_outputs]
        
        print(f"Processing conversation {idx + 1}... Reward: {rewards[0] if rewards else 'None'}")
        
        return rewards[0] if rewards else None
    except Exception as e:
        print(f"Error processing conversation {idx + 1}: {e}")
        return None


df['reward'] = df['conversation'].apply(lambda x, idx: get_reward_for_conversation(x, idx), idx=df.index)
df.to_csv('../datas/train_lmsys_chat_with_rewards_new.csv', index=False)

print("处理完成，数据已保存至 'test_lmsys_chat_with_rewards_new.csv'.")
