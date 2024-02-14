import os
from .clip_encoder import CLIPVisionTower
from .imagebind import ImageBindWrapper
from .open_clip_encoder import OpenCLIPVisionTower
from .eva_vit import EvaViTWrapper
from .hf_vision import HFVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, "mm_vision_tower", getattr(vision_tower_cfg, "vision_tower", None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif vision_tower in ["imagebind_huge"]:
        return ImageBindWrapper(vision_tower, args=vision_tower_cfg, **kwargs)
    elif vision_tower in ["eva_vit_g", "Internal-EVA02-CLIP-10B-14", "Internal-EVA02-CLIP-10B-14-448"]:
        return EvaViTWrapper(vision_tower, args=vision_tower_cfg, **kwargs)
    elif vision_tower.startswith("hf:"):
        return HFVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif vision_tower.startswith("open_clip_hub"):
        return OpenCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f"Unknown vision tower: {vision_tower}")
