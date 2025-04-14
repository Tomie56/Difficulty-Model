from transformers import AutoTokenizer, pipeline
import torch

rm_tokenizer = AutoTokenizer.from_pretrained("sfairXC/FsfairX-LLaMA3-RM-v0.1")
device = 0 # accelerator.device

rm_pipe = pipeline(
      "sentiment-analysis",
      model="sfairXC/FsfairX-LLaMA3-RM-v0.1",
      #device="auto",
      device=device,
      tokenizer=rm_tokenizer,
      model_kwargs={"torch_dtype": torch.bfloat16}
  )

pipe_kwargs = {
    #   "return_all_scores": True,
      "top_k": 1,
      
      "function_to_apply": "none",
      "batch_size": 1
  }

chats = [
#    ------------------------------------- Bad
    [{"role": "user", "content": "Hello, what's your name?"}, {"role": "assistant", "content": "Hi"}], ##过短与不完整
    [{"role": "user", "content": "I'm feeling a bit down today."}, {"role": "assistant", "content": "It's got nothing to do with me."}], ##负面情绪
    [{"role": "user", "content": "Can you help me with my homework?"}, {"role": "assistant", "content": "sufiageufgauefgafgafadasdasdad?"}], ##乱码
    
    
    [{"role": "user", "content": "I'm a bit stressed with work lately."}, {"role": "assistant", "content": "It's got nothing to do with me. But do you want any tips for managing stress?"}],
    [{"role": "user", "content": "I just finished a great book, and it was amazing!"}, {"role": "assistant", "content": "That's great! What book was it?sufiageufgauefgafgafadasdasdad?"}],
    [{"role": "user", "content": "I don't know what to do with my life."}, {"role": "assistant", "content": "I'm really sorry yousufiageufgauefgafgafadasdasdad?'re feeling like that. Do you want to explore some ideas together?"}],

#    -------------------------------------- Good
   
    [{"role": "user", "content": "I'm a bit stressed with work lately."}, {"role": "assistant", "content": "It sounds tough. Do you want any tips for managing stress?"}],
    [{"role": "user", "content": "I just finished a great book, and it was amazing!"}, {"role": "assistant", "content": "That's great! What book was it?"}],
    [{"role": "user", "content": "I don't know what to do with my life."}, {"role": "assistant", "content": "I'm really sorry you're feeling like that. Do you want to explore some ideas together?"}],

]

for idx, chat in enumerate(chats):
    test_texts = [rm_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False).replace(rm_tokenizer.bos_token, "")]
    pipe_outputs = rm_pipe(test_texts, **pipe_kwargs)
    rewards = [output[0]["score"] for output in pipe_outputs]
    print(f"Q {idx+1} reward:{rewards}")
