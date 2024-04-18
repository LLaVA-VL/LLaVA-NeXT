from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda"  # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B-Chat", torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B-Chat")

prompt = "Give me a short introduction to large language model."
messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)
