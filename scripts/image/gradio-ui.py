#!/usr/bin/env python

import gradio as gr
import argparse
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from PIL import Image
import torch
import copy
from transformers import TextStreamer
from transformers import BitsAndBytesConfig

parser = argparse.ArgumentParser(description="LLaVA-NeXT Demo")
parser.add_argument("--load-8bit", action="store_true", help="Load model in 8-bit mode")
parser.add_argument("--load-4bit", action="store_true", help="Load model in 4-bit mode")
parser.add_argument("--initial-model", type=str, default="lmms-lab/llama3-llava-next-8b", help="Initial model to load")
args = parser.parse_args()

model = None
tokenizer = None
image_processor = None

device = "cuda"
device_map = "auto"

def load_model(pretrained, load_8bit=False, load_4bit=False):
    global model, tokenizer, image_processor, conv_template
    model_name = "llava_llama3" if "llama" in pretrained else "llava_qwen"
    conv_template = "llava_llama_3" if "llama" in pretrained else "qwen_1_5"

    quantization_config = None
    if load_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif load_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

    with torch.inference_mode():
        tokenizer, model, image_processor, max_length = load_pretrained_model(
            pretrained, None, model_name, device_map=device_map, quantization_config=quantization_config,
        )
    model.eval()
    model.tie_weights()

    torch.cuda.empty_cache()

def llava_chat(image, user_input, chat_history, temperature, do_sample, max_new_tokens, repetition_penalty):
    global device, conv_template
    if image is not None:
        image = Image.fromarray(image)
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

    # Create a conversation template
    conv = copy.deepcopy(conv_templates[conv_template])

    # Append previous chat history
    for message in chat_history:
        conv.append_message(conv.roles[0], message[0])
        conv.append_message(conv.roles[1], message[1])

    if image is not None:
        question = DEFAULT_IMAGE_TOKEN + "\n" + user_input
    else:
        question = user_input

    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    # Use inference mode to reduce memory usage for inference
    with torch.inference_mode():
        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        image_tensor = [img.to(device) for img in image_tensor] if image is not None else None

    try:
        with torch.inference_mode():
            cont = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image.size] if image is not None else None,
                do_sample=do_sample,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
            )

            text_output = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
            chat_history.append((user_input, text_output))
    except Exception as e:
        print(f"Error during model.generate: {e}")
        chat_history.append((user_input, f"Error: {e}"))
        raise e

    yield chat_history, gr.update(value=None)

def clear_history():
    return [], gr.update(value=[])

def change_model(selected_model):
    if ' (' in selected_model:
        model_name, bit_mode = selected_model.split(' (')
        bit_mode = bit_mode.rstrip(')')
        load_8bit = bit_mode == '8 bit'
        load_4bit = bit_mode == '4 bit'
    else:
        model_name = selected_model
        load_8bit = False
        load_4bit = False
    load_model(model_name, load_8bit, load_4bit)
    return f"Model {selected_model} loaded successfully."

# Determine the initial model selection based on command line arguments
bit_mode = ''
if args.load_8bit:
    bit_mode = '8 bit'
elif args.load_4bit:
    bit_mode = '4 bit'

initial_model_name = args.initial_model
initial_model = f"{initial_model_name} ({bit_mode})" if bit_mode else initial_model_name
load_model(initial_model_name, args.load_8bit, args.load_4bit)

# JavaScript to submit input on Enter key press only
js = """
() => {
    document.querySelector('textarea').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            e.preventDefault(); // Prevent default action to avoid new line
            document.getElementById('submit-btn').click();
        }
    });
}
"""

# Define the Gradio interface
with gr.Blocks(title="LLaVA-NeXT Demo", js=js) as llava_demo:
    gr.Markdown("# LLaVA-NeXT Demo")

    with gr.Row():
        with gr.Column():
            model_selector = gr.Dropdown(
                choices=[
                    "lmms-lab/llama3-llava-next-8b",
                    "lmms-lab/llama3-llava-next-8b (8 bit)",
                    "lmms-lab/llama3-llava-next-8b (4 bit)",
                    "lmms-lab/llava-next-72b",
                    "lmms-lab/llava-next-72b (8 bit)",
                    "lmms-lab/llava-next-72b (4 bit)",
                    "lmms-lab/llava-next-110b",
                    "lmms-lab/llava-next-110b (8 bit)",
                    "lmms-lab/llava-next-110b (4 bit)",
                    "lmms-lab/llava-next-vicuna-v1.5-7b-s2",
                    "lmms-lab/llava-next-vicuna-v1.5-7b-s2 (8 bit)",
                    "lmms-lab/llava-next-vicuna-v1.5-7b-s2 (4 bit)"
                ],
                value=initial_model,
                label="Select Model"
            )
            load_model_btn = gr.Button("Load Model")
            image_input = gr.Image(label="Upload Image (optional)", height=300)
            user_input = gr.Textbox(label="Your Input", lines=3)
            temperature = gr.Slider(minimum=0, maximum=1, value=0.2, label="Temperature")
            do_sample = gr.Checkbox(value=True, label="Do Sample")
            max_new_tokens = gr.Slider(minimum=1, maximum=32768, value=2048, step=1, label="Max New Tokens")
            repetition_penalty = gr.Slider(minimum=1.0, maximum=2.0, value=1.1, step=0.1, label="Repetition Penalty")
            submit_btn = gr.Button("Submit", elem_id="submit-btn")
            clear_btn = gr.Button("Clear Chat History")

        with gr.Column():
            chat_history = gr.Chatbot(label="Chat History")
            model_status = gr.Textbox(label="Model Status", value=f"Model {initial_model} loaded successfully.", interactive=False)

    submit_btn.click(llava_chat, [image_input, user_input, chat_history, temperature, do_sample, max_new_tokens, repetition_penalty], [chat_history, image_input])
    clear_btn.click(clear_history, [], [chat_history, image_input])
    load_model_btn.click(change_model, [model_selector], [model_status])

llava_demo.launch(server_name="0.0.0.0", server_port=7860)
