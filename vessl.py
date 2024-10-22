# Basic
import os
from pydantic import BaseModel
from typing import List
# FastAPI
from fastapi import FastAPI, HTTPException, Depends, Header, Query
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
    embeddings: List[str] = None
    vectors: List[List[float]] = None
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

"""
Request:
    Headers:
        `x-api-key` (str)
    Query Parameters:
        `request_type` (str, optional): The type of request expected. Either `hexadecimal` (default) or `vectors`.
    Body:
        `embeddings` (List[str], optional): A list of hexadecimal embeddings.
        `vectors` (List[List[float]], optional): A list of vectors.
        `conversation` (List[Message]): A list of messages with role and content.
Returns:
    The result of the inference based on the provided embeddings or vectors and conversation.

Examples:

1. request_type == `embeddings`:
```
curl --location 'http://127.0.0.1:8000/generate?request_type=hexadecimal' \
--header 'x-api-key;' \
--data '{
    "embeddings": [
        "85 84 84 3F F1 F0 70 3F 91 90 10 3E D9 D8 D8 3E CD CC CC 3E C1 C0 C0 3C 8B",
        "91 90 10 3F C1 C0 40 3F 91 90 10 40 C1 C0 C0 3C 91 90 10 3E 00 00 00 00 00"
    ],
    "conversation":[
        {"role":"user", "content":"<image>\nPlease provide a detailed description of the video, focusing on the main subjects, their actions, the background scenes."}
    ]
}'
```

2. request_type == `vectors`:
```
curl --location 'http://127.0.0.1:8000/generate?request_type=vectors' \
--header 'x-api-key: YOUR_API_KEY' \
--data '{
    "vectors": [
        [ 1.0, 0.9, 0.1 ],
        [ 0.5, 0.2, 0.0 ]
    ],
    "conversation":[
        {"role":"user", "content":"<image>\nPlease provide a detailed description of the video, focusing on the main subjects, their actions, the background scenes."}
    ]
}'
```
"""
@app.post("/generate")
async def generate(
    request: GenerateRequest,
    request_type: str = Query(default="hexadecimal", enum=["hexadecimal", "vectors"]),
    _: str = Depends(verify_api_key)):

    embedding_tensors = []
    if request_type == "hexadecimal":
        if "embeddings" not in request.model_fields:
            raise HTTPException(status_code=400, detail="request_type is `hexadecimal` but `embeddings` not found in request")
        for embedding in request.embeddings:
            byte_array = bytes.fromhex(embedding)
            import struct
            num_floats = len(byte_array) // 4
            embedding_tensors.append(torch.tensor(struct.unpack('<' + 'f' * num_floats, byte_array), dtype=torch.float16).unsqueeze(0))
    elif request_type == "vectors":
        if "vectors" not in request.model_fields:
            raise HTTPException(status_code=400, detail="request_type is `vectors` but `vectors` not found in request")
        embedding_tensors = torch.tensor(request.vectors, dtype=torch.float16)
    else:
        raise HTTPException(status_code=400, detail="request_type needs to be hexadecimal or vectors")
        
    conversation = request.conversation
    return await run_inference(embedding_tensors, conversation)

async def run_inference(embedding_tensors, conversation):
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

