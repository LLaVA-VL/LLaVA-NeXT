import torch
import torch.nn as nn

from .eva_clip_processors import EvaClipImageTrainProcessor
from .eva_vit import EVAEncoderWrapper
from .factory import list_models, add_model_config, get_model_config

from llava.utils import rank0_print


class EvaClipVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.vision_tower_pretrained = args.vision_tower_pretrained
        self.config = get_model_config(vision_tower)

        if not delay_load:
            rank0_print(f"Loading EVA ViT: {self.vision_tower_name}")
            self.load_model()
        elif getattr(args, "unfreeze_mm_vision_tower", False):
            # TODO: better detector is needed.
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
            self.load_model()
        elif hasattr(args, "mm_tunable_parts") and "mm_vision_tower" in args.mm_tunable_parts:
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`.")
            self.load_model()
        else:
            self.cfg_only = self.config

    def load_model(self, device_map=None):
        rank0_print(f"Pretrained: {self.vision_tower_pretrained}")
        self.image_processor = EvaClipImageTrainProcessor(self.config["vision_cfg"]["image_size"])
        self.vision_tower = EVAEncoderWrapper(self.vision_tower_pretrained, self.config)
        rank0_print(f"Loaded image processor: {self.image_processor}")
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0)).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_features = self.vision_tower(images.to(device=self.device, dtype=self.dtype)).to(images.dtype)

        return image_features

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def hidden_size(self):
        return self.config["vision_cfg"]["width"]

    @property
    def num_patches(self):
        return (self.config["vision_cfg"]["image_size"] // self.config["vision_cfg"]["patch_size"]) ** 2

    @property
    def num_patches_per_side(self):
        return self.config["vision_cfg"]["image_size"] // self.config["vision_cfg"]["patch_size"]

    @property
    def image_size(self):
        return self.config["vision_cfg"]["image_size"]
