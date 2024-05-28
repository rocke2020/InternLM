import time
from icecream import ic
ic.configureOutput(includeContext=True, argToStringFunction=str)
ic.lineWrapWidth = 120

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_dir = "internlm/internlm2-chat-7b"
tokenizer = AutoTokenizer.from_pretrained(
    "internlm/internlm2-chat-7b", trust_remote_code=True
)
# Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and might cause OOM Error.
t0 = time.time()

full_load = 0
ic(full_load)
if full_load:
    model = AutoModelForCausalLM.from_pretrained(
        "internlm/internlm2-chat-7b",
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
else:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization_config,
    )
model = model.eval()
print(f"Model loaded in {time.time()-t0:.2f} seconds.")
response, history = model.chat(tokenizer, "hello", history=[])
print(response)
# Output: Hello? How can I help you today?
response, history = model.chat(
    tokenizer,
    "please provide three suggestions about time management and reply in English",
    history=history,
)
print(response)
