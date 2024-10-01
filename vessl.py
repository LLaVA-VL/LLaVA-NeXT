# Basic
import os
from pydantic import BaseModel
from typing import List
# FastAPI
from fastapi import FastAPI, HTTPException, Depends, Header
from contextlib import asynccontextmanager
# ML
import torch
from llava.conversation import Conversation, SeparatorStyle
from llava.model.language_model.llava_qwen import LlavaQwenConfig, LlavaQwenForCausalLM
from inference import load_model, run_inference

class Message(BaseModel):
    role: str
    content: str

class GenerateRequest(BaseModel):
    embeddings: List[str]
    conversation: List[Message]

model_path = os.getenv(['MODEL_PATH'])
if not model_path:
    raise RuntimeError("MODEL_PATH environment variable not set. Set it by running `export MODEL_PATH='...'`")
api_key = os.getenv('API_KEY')
if not api_key:
    raise RuntimeError("API_KEY environment variable not set. Set it by running `export API_KEY='...'`")

config = None
tokenizer = None
model = None
mm_projector = None

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != api_key:
        raise HTTPException(status_code=401, detail="Invalid API Key")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global config, tokenizer, model, mm_projector 
    config = LlavaQwenConfig.from_pretrained(model_path)
    tokenizer, model, mm_projector = load_model(model_path, config)
    print("Model loaded successfully!")
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/generate")
async def generate(
    request: GenerateRequest,
    _: str = Depends(verify_api_key)):

    embeddings = request.embeddings # TODO: support response_type = hexadecimal | vector
    conversation = request.conversation
    return await run_inference(embeddings, conversation)

async def run_inference(embeddings, conversation):
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

