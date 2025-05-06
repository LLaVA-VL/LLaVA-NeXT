import dataclasses
from enum import auto, Enum
from typing import List, Any, Dict, Union, Tuple
import re
import base64
from io import BytesIO
from PIL import Image
from transformers import AutoTokenizer


class SeparatorStyle(Enum):
    """Different separator style."""

    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    CHATML = auto()
    LLAMA_2 = auto()
    LLAMA_3 = auto()
    QWEN = auto()
    GEMMA = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    tokenizer_id: str = ""
    tokenizer: Any = None
    # Stop criteria (the default one is EOS token)
    stop_str: Union[str, List[str]] = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    skip_next: bool = False

    def get_prompt(self):
        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0]
            if "mmtag" in self.version:
                init_msg = init_msg.replace("<image>", "").strip()
                messages[0] = (init_role, init_msg)
                messages.insert(0, (self.roles[0], "<Image><image></Image>"))
                messages.insert(1, (self.roles[1], "Received."))
            elif not init_msg.startswith("<image>"):
                init_msg = init_msg.replace("<image>", "").strip()
                messages[0] = (init_role, "<image>\n" + init_msg)
            else:
                messages[0] = (init_role, init_msg)

        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"

        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"

        elif self.sep_style == SeparatorStyle.CHATML:
            ret = "" if self.system == "" else self.system + self.sep + "\n"
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, images, _ = message
                        message = "<image>" * len(images) + message
                    ret += role + "\n" + message + self.sep + "\n"
                else:
                    ret += role + "\n"
            return ret

        elif self.sep_style == SeparatorStyle.LLAMA_3:
            if self.tokenizer is None:
                raise ValueError("Llama 3 tokenizer is not available. Make sure you have the necessary permissions.")
            chat_template_messages = [{"role": "system", "content": self.system}]
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, images = message
                        message = "<image>" * len(images) + message
                    chat_template_messages.append({"role": role, "content": message})

            # print(chat_template_messages)
            return self.tokenizer.apply_chat_template(chat_template_messages, tokenize=False, add_generation_prompt=True)
            # ret = "" if self.system == "" else self.system + self.sep + "\n"
            # for role, message in messages:
            #     if message:
            #         if type(message) is tuple:
            #             message, images = message
            #             message = "<image>" * len(images) + message
            #         ret += role + "\n" + message + self.sep + "\n"
            #     else:
            #         ret += role + "\n"
            # return ret

        elif self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role

        elif self.sep_style == SeparatorStyle.GEMMA:
            ret = ""
            for i, (role, message) in enumerate(messages):
                assert role == self.roles[i % 2], "Conversation should alternate user/assistant/user/assistant/..."
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role

        elif self.sep_style == SeparatorStyle.LLAMA_2:
            wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n" if len(msg) > 0 else msg
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = ""

            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i == 0:
                        message = wrap_sys(self.system) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += " " + message + " " + self.sep2
                else:
                    ret += ""
            ret = ret.lstrip(self.sep)

        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ""
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

    def process_image(self, image, image_process_mode, return_pil=False, image_format="PNG"):
        if image_process_mode == "Pad":

            def expand2square(pil_img, background_color=(122, 116, 104)):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(image)
        elif image_process_mode in ["Default", "Crop"]:
            pass
        elif image_process_mode == "Resize":
            image = image.resize((336, 336))
        else:
            raise ValueError(f"Invalid image_process_mode: {image_process_mode}")

        if type(image) is not Image.Image:
            image = Image.open(image).convert("RGB")

        max_hw, min_hw = max(image.size), min(image.size)
        aspect_ratio = max_hw / min_hw
        max_len, min_len = 672, 448
        shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
        longest_edge = int(shortest_edge * aspect_ratio)
        W, H = image.size
        if H > W:
            H, W = longest_edge, shortest_edge
        else:
            H, W = shortest_edge, longest_edge
        image = image.resize((W, H))
        if return_pil:
            return image
        else:
            buffered = BytesIO()
            image.save(buffered, format=image_format)
            img_b64_str = base64.b64encode(buffered.getvalue()).decode()
            return img_b64_str

    def get_images(self, return_pil=False, return_path=False):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image, image_process_mode = msg
                    if type(image) != list:
                        image = [image]
                    for img in image:
                        if not return_path and self.is_image_file(img):
                            img = self.process_image(img, image_process_mode, return_pil=return_pil)
                        else:
                            images.append(img)
        return images

    def is_image_file(self, filename):
        image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"]
        return any(filename.lower().endswith(ext) for ext in image_extensions)

    def is_video_file(self, filename):
        video_extensions = [".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".mpeg", ".mpg"]
        return any(filename.lower().endswith(ext) for ext in video_extensions)

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image, image_process_mode = msg
                    if type(image) != list:
                        image = [image]
                    if len(image) == 1:
                        msg = "<image>\n" + msg.replace("<image>", "").strip()
                    else:
                        msg = re.sub(r"(<image>)\n(?=<image>)", r"\1 ", msg)

                    img_str_list = []                         
                    for img in image:
                        if self.is_image_file(img):
                            img_b64_str = self.process_image(img, "Default", return_pil=False, image_format="JPEG")
                            img_str = f'<img src="data:image/jpeg;base64,{img_b64_str}" style="max-width: 256px; max-height: 256px; width: auto; height: auto; object-fit: contain;"/>'
                            img_str_list.append(img_str)
                        elif self.is_video_file(img):
                            ret.append(((img,), None))

                    msg = msg.strip()
                    img_place_holder = ""
                    for img_str in img_str_list:
                        img_place_holder += f"{img_str}\n\n"

                    if len(img_str_list) > 0:
                        msg = f"{img_place_holder}\n\n{msg}"

                    if len(msg) > 0:
                        ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(system=self.system, roles=self.roles, messages=[[x, y] for x, y in self.messages], offset=self.offset, sep_style=self.sep_style, sep=self.sep, sep2=self.sep2, version=self.version)

    def dict(self):
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


conv_vicuna_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. " "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=[
        ["Human", "What are the key differences between renewable and non-renewable energy sources?"],
        [
            "Assistant",
            "Renewable energy sources are those that can be replenished naturally in a relatively "
            "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
            "Non-renewable energy sources, on the other hand, are finite and will eventually be "
            "depleted, such as coal, oil, and natural gas. Here are some key differences between "
            "renewable and non-renewable energy sources:\n"
            "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
            "energy sources are finite and will eventually run out.\n"
            "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
            "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
            "and other negative effects.\n"
            "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
            "have lower operational costs than non-renewable sources.\n"
            "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
            "locations than non-renewable sources.\n"
            "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
            "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
            "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
            "non-renewable sources are not, and their depletion can lead to economic and social instability.\n",
        ],
    ],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_vicuna_v1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. " "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llama_2 = Conversation(
    system="""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_llava_llama_2 = Conversation(
    system="You are a helpful language and vision assistant. " "You are able to understand the visual content that the user provides, " "and assist the user with a variety of tasks using natural language.",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

def safe_load_tokenizer(tokenizer_id):
    try:
        return AutoTokenizer.from_pretrained(tokenizer_id)
    except Exception:
        return None

conv_llava_llama_3 = Conversation(
    system="You are a helpful language and vision assistant. " "You are able to understand the visual content that the user provides, " "and assist the user with a variety of tasks using natural language.",
    roles=("user", "assistant"),
    version="llama_v3",
    messages=[],
    offset=0,
    sep="<|eot_id|>",
    sep_style=SeparatorStyle.LLAMA_3,
    tokenizer_id="meta-llama/Meta-Llama-3-8B-Instruct",
    tokenizer=safe_load_tokenizer("meta-llama/Meta-Llama-3-8B-Instruct"),
    stop_token_ids=[128009],
)

conv_mistral_instruct = Conversation(
    system="",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="",
    sep2="</s>",
)

conv_llava_llama_2_simple = Conversation(
    system="Answer the questions about the visual content that the user provides.",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_llava_llama_2_mmtag = Conversation(
    system="Answer the questions about the visual content that the user provides." "The visual content will be provided with the following format: <Image>visual content</Image>.",
    roles=("USER", "ASSISTANT"),
    version="llama_v2_mmtag",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_mpt = Conversation(
    system="""<|im_start|>system
A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

conv_qwen = Conversation(
    system="""<|im_start|>system
You are a helpful assistant.""",
    roles=("<|im_start|>user", "<|im_start|>assistant"),
    version="qwen",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.CHATML,
    sep="<|im_end|>",
)

conv_qwen_2_5 = Conversation(
    system="""<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.""",
    roles=("<|im_start|>user", "<|im_start|>assistant"),
    version="qwen",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.CHATML,
    sep="<|im_end|>",
)

conv_gemma_instruct = Conversation(system="", roles=("<start_of_turn>user\n", "<start_of_turn>model\n"), version="gemma", messages=[], offset=0, sep_style=SeparatorStyle.GEMMA, sep="<end_of_turn>\n")

conv_llava_plain = Conversation(
    system="",
    roles=("", ""),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.PLAIN,
    sep="\n",
)

conv_llava_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. " "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_llava_v0_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
    "The visual content will be provided with the following format: <Image>visual content</Image>.",
    roles=("Human", "Assistant"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
    version="v0_mmtag",
)

conv_llava_v1 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. " "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llava_v1_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
    "The visual content will be provided with the following format: <Image>visual content</Image>.",
    roles=("USER", "ASSISTANT"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
    version="v1_mmtag",
)

conv_mistral_orca = Conversation(
    system="""<|im_start|>system
You are MistralOrca, a large language model trained by Alignment Lab AI. Write out your reasoning step-by-step to be sure you get the right answers!""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

conv_mistral_zephyr = Conversation(
    system="""<|system|>
You are a helpful AI assistant.""",
    roles=("<|user|>\n", "<|assistant|>\n"),
    version="mpt",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="</s>",
)

conv_mistral_direct = Conversation(
    system="""<|im_start|>system
Answer the questions.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

conv_chatml_direct = Conversation(
    system="""<|im_start|>system
Answer the questions.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

default_conversation = conv_vicuna_v0
conv_templates = {
    "default": conv_vicuna_v0,
    "v0": conv_vicuna_v0,
    "v1": conv_vicuna_v1,
    "vicuna_v1": conv_vicuna_v1,
    "llama_2": conv_llama_2,
    "mistral_instruct": conv_mistral_instruct,
    "mistral_orca": conv_mistral_orca,
    "mistral_zephyr": conv_mistral_zephyr,
    "mistral_direct": conv_mistral_direct,
    "plain": conv_llava_plain,
    "v0_plain": conv_llava_plain,
    "chatml_direct": conv_chatml_direct,
    "llava_v0": conv_llava_v0,
    "llava_v0_mmtag": conv_llava_v0_mmtag,
    "llava_v1": conv_llava_v1,
    "llava_v1_mmtag": conv_llava_v1_mmtag,
    "llava_llama_2": conv_llava_llama_2,
    "llava_llama_3": conv_llava_llama_3,
    "llava_llama_2_simple": conv_llava_llama_2_simple,
    "llava_llama_2_mmtag": conv_llava_llama_2_mmtag,
    "llava_mistral_instruct": conv_mistral_instruct,
    "mpt": conv_mpt,
    "qwen_1_5": conv_qwen,
    "qwen_2": conv_qwen,
    "qwen_2_5": conv_qwen_2_5,
    "gemma_instruct": conv_gemma_instruct,
}


if __name__ == "__main__":
    print(default_conversation.get_prompt())
