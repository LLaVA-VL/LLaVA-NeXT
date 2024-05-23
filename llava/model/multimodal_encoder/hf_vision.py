import torch
import torch.nn as nn

from transformers import AutoModel, AutoImageProcessor, AutoConfig, CLIPImageProcessor
from llava.utils import rank0_print


class HFVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower.replace("hf:", "", 1)
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = AutoConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        try:
            self.image_processor = AutoImageProcessor.from_pretrained(self.vision_tower_name)
        except Exception as e:
            if "448" in self.vision_tower_name:
                image_size = 448
                # use image processor with conig
                self.image_processor = CLIPImageProcessor(size={"shortest_edge": image_size}, do_center_crop=True, crop_size=image_size)
            else:
                self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        rank0_print(f"Loaded image processor: {self.image_processor}")
        self.vision_tower = AutoModel.from_pretrained(self.vision_tower_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
        self.device = self.vision_tower.device
        self.dtype = self.vision_tower.dtype
        self.config = self.vision_tower.config

        if hasattr(self.vision_tower, "vision_model"):
            self.vision_tower = self.vision_tower.vision_model
        self.vision_tower.requires_grad_(False)
        # self.vision_tower.eval()
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        select_feature_type = self.select_feature

        if self.select_feature in ["slicefour_patch", "slicefour_cls_patch"]:
            select_every_k_layer = len(image_forward_outs.hidden_states) // 4
            image_features = torch.cat([image_forward_outs.hidden_states[i] for i in range(select_every_k_layer + self.select_layer, len(image_forward_outs.hidden_states), select_every_k_layer)], dim=-1)
            select_feature_type = select_feature_type.replace("slicefour_", "")
        else:
            image_features = image_forward_outs.hidden_states[self.select_layer]

        if select_feature_type == "patch":
            image_features = image_features[:, 1:]
        elif select_feature_type == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {select_feature_type}")
        return image_features

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    # @property
    # def dtype(self):
    #     return self.vision_tower.dtype

    # @property
    # def device(self):
    #     return self.vision_tower.device

    @property
    def hidden_size(self):
        try:
            _hidden_size = self.config.hidden_size
        except:
            _hidden_size = self.config.vision_config.hidden_size
        if "slicefour" in self.select_feature:
            _hidden_size *= 4
        return _hidden_size

    @property
    def num_patches(self):
        _num_patches = (self.config.image_size // self.config.patch_size) ** 2
        if "cls_patch" in self.select_feature:
            _num_patches += 1
        return _num_patches

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def image_size(self):
        return self.config.image_size
