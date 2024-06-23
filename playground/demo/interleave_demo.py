
# from .demo_modelpart import InferenceDemo
import gradio as gr
import os
# import time
import cv2


# import copy
import torch
# import random
import numpy as np

from llava import conversation as conversation_lib
from llava.constants import DEFAULT_IMAGE_TOKEN


from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

class InferenceDemo(object):
    def __init__(self,args,model_path) -> None:
        disable_torch_init()

        model_name = get_model_name_from_path(args.model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit)

        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        elif 'qwen' in model_name.lower():
            conv_mode = "qwen_1_5"
        else:
            conv_mode = "llava_v0"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print("[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(conv_mode, args.conv_mode, args.conv_mode))
        else:
            args.conv_mode = conv_mode
            pass
        self.conv_mode=conv_mode
        self.conversation = conv_templates[args.conv_mode].copy()
        self.num_frames = args.num_frames
        pass



def is_valid_video_filename(name):
    # 定义支持的视频文件后缀列表
    video_extensions = ['avi', 'mp4', 'mov', 'mkv', 'flv', 'wmv', 'mjpeg']
    
    # 获取文件的后缀名，并转换为小写
    ext = name.split('.')[-1].lower()
    
    # 检查文件后缀是否在支持的列表中
    if ext in video_extensions:
        return True
    else:
        return False

def sample_frames(video_file, num_frames) :
    # 打开视频文件
    video = cv2.VideoCapture(video_file)
    #获取视频的总帧数
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    #计算采样间隔
    interval = total_frames // num_frames
    #初始化结果列表
    frames = []
    #遍历视频的每一帧
    for i in range(total_frames):
        #读取当前帧
        ret, frame = video.read()
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        #如果帧为空，则跳过
        if not ret:
            continue
        #如果是采样帧，则添加到结果列表中
        if i % interval == 0:
            frames.append(pil_img)
    # import pdb;pdb.set_trace()
    # 关闭视频文件
    video.release()
    return frames

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            print('failed to load the image')
    else:
        print('Load image from local file')
        print(image_file)
        image = Image.open(image_file).convert("RGB")
        
    return image


def clear_history(history):

    our_chatbot.conversation = conv_templates[our_chatbot.conv_mode].copy()

    return None
def clear_response(history):
    for index_conv in range(1, len(history)):
        # loop until get a text response from our model.
        conv = history[-index_conv]
        if not (conv[0] is None):
            break
    question = history[-index_conv][0]
    history = history[:-index_conv]
    return history, question

# def print_like_dislike(x: gr.LikeData):
#     print(x.index, x.value, x.liked)


def add_video(history, video):
    # if len(history)>1:
    #     history=[]
    print("LOG. Add Video Function is called.")
    # path_input_vid = get_tmp_file_name(type="video",prefix="tmp_input_vid_")
    # print("LOG. Video is saved to: ", path_input_vid)
    # import pdb;pdb.set_trace()
    history = history + [((video, ), "A New video is added.")]
    return history

def add_message(history, message):
    # history=[]
    for x in message["files"]:
        history.append(((x,), None))
    if message["text"] is not None:
        history.append((message["text"], None))
    return history, gr.MultimodalTextbox(value=None, interactive=False)

def add_text(history, text, temporature_slider=0.2):
    def response2stream(response, question):
        return [[question, response]]
    images_this_term=[]
    text_this_term=''
    # import pdb;pdb.set_trace()
    num_new_images = 0
    for i,message in enumerate(history):
        if type(message[0]) is tuple:
            images_this_term.append(message[0][0])
            if is_valid_video_filename(message[0][0]):
                num_new_images+=our_chatbot.num_frames
            else:
                num_new_images+=1
        else:
            num_new_images=0
            
    # for message in history[-i-1:]:
    #     images_this_term.append(message[0][0])

    assert len(images_this_term)>0, "must have an image"
    # image_files = (args.image_file).split(',')
    # image = [load_image(f) for f in images_this_term if f]
    image_list=[]
    for f in images_this_term:
        if is_valid_video_filename(f):
            image_list+=sample_frames(f, our_chatbot.num_frames)
        else:
            image_list.append(load_image(f))
    image_tensor = [our_chatbot.image_processor.preprocess(f, return_tensors="pt")["pixel_values"][0].half().to(our_chatbot.model.device) for f in image_list]

    image_tensor = torch.stack(image_tensor)
    image_token = DEFAULT_IMAGE_TOKEN*num_new_images
    # if our_chatbot.model.config.mm_use_im_start_end:
    #     inp = DEFAULT_IM_START_TOKEN + image_token + DEFAULT_IM_END_TOKEN + "\n" + inp
    # else:
    inp=text
    inp = image_token + "\n" + inp
    our_chatbot.conversation.append_message(our_chatbot.conversation.roles[0], inp)
    our_chatbot.conversation.append_message(our_chatbot.conversation.roles[1], None)
    prompt = our_chatbot.conversation.get_prompt()

    input_ids = tokenizer_image_token(prompt, our_chatbot.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(our_chatbot.model.device)
    stop_str = our_chatbot.conversation.sep if our_chatbot.conversation.sep_style != SeparatorStyle.TWO else our_chatbot.conversation.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, our_chatbot.tokenizer, input_ids)
    streamer = TextStreamer(our_chatbot.tokenizer, skip_prompt=True, skip_special_tokens=True)
    # import pdb;pdb.set_trace()
    with torch.inference_mode():
        output_ids = our_chatbot.model.generate(input_ids, images=image_tensor, do_sample=True, temperature=temporature_slider, max_new_tokens=1024, streamer=streamer, use_cache=True, stopping_criteria=[stopping_criteria])

    outputs = our_chatbot.tokenizer.decode(output_ids[0]).strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    our_chatbot.conversation.messages[-1][-1] = outputs
   
    history += response2stream(outputs, text)

    return history, None
 

def bot(history):
    text=history[-1][0]

    images_this_term=[]
    text_this_term=''
    # import pdb;pdb.set_trace()
    num_new_images = 0
    for i,message in enumerate(history[:-1]):
        if type(message[0]) is tuple:
            images_this_term.append(message[0][0])
            if is_valid_video_filename(message[0][0]):
                num_new_images+=our_chatbot.num_frames
            else:
                num_new_images+=1
        else:
            num_new_images=0
            
    # for message in history[-i-1:]:
    #     images_this_term.append(message[0][0])

    assert len(images_this_term)>0, "must have an image"
    # image_files = (args.image_file).split(',')
    # image = [load_image(f) for f in images_this_term if f]
    image_list=[]
    for f in images_this_term:
        if is_valid_video_filename(f):
            image_list+=sample_frames(f, our_chatbot.num_frames)
        else:
            image_list.append(load_image(f))
    image_tensor = [our_chatbot.image_processor.preprocess(f, return_tensors="pt")["pixel_values"][0].half().to(our_chatbot.model.device) for f in image_list]

    image_tensor = torch.stack(image_tensor)
    image_token = DEFAULT_IMAGE_TOKEN*num_new_images
    # if our_chatbot.model.config.mm_use_im_start_end:
    #     inp = DEFAULT_IM_START_TOKEN + image_token + DEFAULT_IM_END_TOKEN + "\n" + inp
    # else:
    inp=text
    inp = image_token+ "\n" + inp
    our_chatbot.conversation.append_message(our_chatbot.conversation.roles[0], inp)
    # image = None
    our_chatbot.conversation.append_message(our_chatbot.conversation.roles[1], None)
    prompt = our_chatbot.conversation.get_prompt()

    input_ids = tokenizer_image_token(prompt, our_chatbot.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(our_chatbot.model.device)
    stop_str = our_chatbot.conversation.sep if our_chatbot.conversation.sep_style != SeparatorStyle.TWO else our_chatbot.conversation.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, our_chatbot.tokenizer, input_ids)
    streamer = TextStreamer(our_chatbot.tokenizer, skip_prompt=True, skip_special_tokens=True)
    # import pdb;pdb.set_trace()
    with torch.inference_mode():
        output_ids = our_chatbot.model.generate(input_ids, images=image_tensor, do_sample=True, temperature=0.2, max_new_tokens=1024, streamer=streamer, use_cache=True, stopping_criteria=[stopping_criteria])

    outputs = our_chatbot.tokenizer.decode(output_ids[0]).strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    our_chatbot.conversation.messages[-1][-1] = outputs
   
    history[-1]=[text,outputs]
    
    return history
txt = gr.Textbox(
    scale=4,
    show_label=False,
    placeholder="Enter text and press enter.",
    container=False,
)
with gr.Blocks() as demo:
    # Informations
    title_markdown = ("""
        # LLaVA-NeXT Interleave
        [[Blog]](https://llava-vl.github.io/blog/2024-06-16-llava-next-interleave/)  [[Code]](https://github.com/LLaVA-VL/LLaVA-NeXT) [[Model]](https://huggingface.co/lmms-lab/llava-next-interleave-7b)
    """)
    tos_markdown = ("""
    ### TODO!. Terms of use
    By using this service, users are required to agree to the following terms:
    The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research.
    Please click the "Flag" button if you get any inappropriate answer! We will collect those to keep improving our moderator.
    For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.
    """)
    learn_more_markdown = ("""
    ### TODO!. License
    The service is a research preview intended for non-commercial use only, subject to the model [License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA, [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI, and [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT. Please contact us if you find any potential violation.
    """)
    models = [
        "LLaVA-Interleave-7B",
    ]
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    gr.Markdown(title_markdown)

    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False
    )
    chat_input = gr.MultimodalTextbox(interactive=True, file_types=["image","video"], placeholder="Enter message or upload file...", show_label=False)
    
    

    with gr.Row():
        upvote_btn = gr.Button(value="👍  Upvote", interactive=True)
        downvote_btn = gr.Button(value="👎  Downvote", interactive=True)
        flag_btn = gr.Button(value="⚠️  Flag", interactive=True)
        #stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=True)
        regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=True)
        clear_btn = gr.Button(value="🗑️  Clear history", interactive=True)
    chat_msg = chat_input.submit(add_message, [chatbot, chat_input], [chatbot, chat_input])
    bot_msg = chat_msg.then(bot, chatbot, chatbot, api_name="bot_response")
    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

    # chatbot.like(print_like_dislike, None, None)
    clear_btn.click(fn=clear_history, inputs=[chatbot], outputs=[chatbot], api_name="clear_all")
    with gr.Column():
        gr.Examples(examples=[
            [{"files": [f"{cur_dir}/examples/iron.jpg",f"{cur_dir}/examples/iron_man_cartoon.jpg"], "text": "Please describe the similarity of the images."}],
            [{"files": [f"{cur_dir}/examples/image-00007.jpeg",f"{cur_dir}/examples/image-00053.jpeg"], "text": "What is the relation between the images?"}],
            [{"files": [f"{cur_dir}/examples/strawberry.png",f"{cur_dir}/examples/orange.png"], "text": "How to edit image1 to make it look like image2?"}],
            [{"files": [f"{cur_dir}/examples/dog_to_monkey1.png",f"{cur_dir}/examples/dog_to_monkey2.png"], "text": "How to edit image1 to make it look like image2?"}],
            [{"files": [f"{cur_dir}/examples/original_bench.jpeg",f"{cur_dir}/examples/changed_bench.jpeg"], "text": "How to edit image1 to make it look like image2?"}],
            [{"files": [f"{cur_dir}/examples/startup.png",f"{cur_dir}/examples/bigcompany.png"], "text": "What if fun about the images?"}],
            [{"files": [f"{cur_dir}/examples/resumea.png",f"{cur_dir}/examples/resumeb.jpg"], "text": "what do you think about this candidate?"}],
            [{"files": [f"{cur_dir}/examples/twitter1.jpeg",f"{cur_dir}/examples/twitter2.jpeg",f"{cur_dir}/examples/twitter3.jpeg",f"{cur_dir}/examples/twitter4.jpeg"], "text": "Please write a blog post for the images."}],

            
        ], inputs=[chat_input], label="Compare images: ")

demo.queue()
if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--server_name", default="0.0.0.0", type=str)
    argparser.add_argument("--port", default="6123", type=str)
    argparser.add_argument("--model_path", default="", type=str)
    # argparser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    argparser.add_argument("--model-base", type=str, default=None)
    argparser.add_argument("--num-gpus", type=int, default=1)
    argparser.add_argument("--conv-mode", type=str, default=None)
    argparser.add_argument("--temperature", type=float, default=0.2)
    argparser.add_argument("--max-new-tokens", type=int, default=512)
    argparser.add_argument("--num_frames", type=int, default=16)
    argparser.add_argument("--load-8bit", action="store_true")
    argparser.add_argument("--load-4bit", action="store_true")
    argparser.add_argument("--debug", action="store_true")
    
    args = argparser.parse_args()
    model_path = args.model_path
    filt_invalid="cut"
    our_chatbot = InferenceDemo(args,model_path)
    # import pdb;pdb.set_trace()
    try:
        demo.launch(server_name=args.server_name, server_port=int(args.port),share=True)
    except Exception as e:
        args.port=int(args.port)+1
        print(f"Port {args.port} is occupied, try port {args.port}")
        demo.launch(server_name=args.server_name, server_port=int(args.port),share=True)
