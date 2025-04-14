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
    # "return_all_scores": True,
    "top_k": None,
      "function_to_apply": "none",
      "batch_size": 1
  }

chat = [
   {"role": "user", "content": "Hello, how are you?"},
   {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
   {"role": "user", "content": "I'd like to show off how chat templating works!"},
  ]

test_texts = [rm_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False).replace(rm_tokenizer.bos_token, "")]
pipe_outputs = rm_pipe(test_texts, **pipe_kwargs)
rewards = [output[0]["score"] for output in pipe_outputs]