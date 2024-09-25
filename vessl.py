import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from llava.conversation import Conversation, SeparatorStyle
from llava.model.language_model.llava_qwen import LlavaQwenConfig, LlavaQwenForCausalLM
from inference import load_model, run_inference
from contextlib import asynccontextmanager

class Message(BaseModel):
    role: str
    content: str

class GenerateRequest(BaseModel):
    embeddings: List[str]
    conversation: List[Message]

model_path = os.environ['MODEL_PATH']

config = None
tokenizer = None
model = None
mm_projector = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global config, tokenizer, model, mm_projector 
    config = LlavaQwenConfig.from_pretrained(model_path)
    tokenizer, model, mm_projector = load_model(model_path, config)
    print("Model loaded successfully!")
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/generate")
async def generate(request: GenerateRequest):
    # Access embeddings and conversation from the request
    embeddings = request.embeddings
    conversation = request.conversation

    embedding_tensors = []
    for embedding in embeddings:
        byte_array = bytes.fromhex(embedding)
        import struct
        num_floats = len(byte_array) // 4
        embedding_tensors.append(torch.tensor(struct.unpack('<' + 'f' * num_floats, byte_array), dtype=torch.float16).unsqueeze(0))

    embedding_tensors = torch.stack(embedding_tensors).cuda()

    conv = Conversation(
        system="""<|im_start|>system
You are a helpful assistant.""",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        version="qwen",
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
    )

    for message in conversation:
        if 'user' in message.role:
            conv.append_message(conv.roles[0], message.content)
        elif 'assistant' in message.role:
            conv.append_message(conv.roles[1], message.content)
        
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    prompt, outputs = run_inference(embedding_tensors, prompt, config, tokenizer, model, mm_projector)

    return {
        "prompt": prompt, 
        "outputs": outputs
    }

