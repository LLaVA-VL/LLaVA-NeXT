import torch
import torch.nn as nn
from transformers import CLIPImageProcessor
from llava.utils import rank0_print

try:
    import open_clip
    import torchvision
    from open_clip.transformer import _expand_token
except ImportError:
    print("OpenCLIP not installed")
    open_clip = None

HIDDEN_SIZE_DICT = {
    "ViT-H-14-378-quickgelu": 1280,
}


class OpenCLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.model_name = vision_tower.replace("open_clip_hub:", "")
        self.pretrained = args.vision_tower_pretrained
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

        if not delay_load:
            rank0_print(f"Loading vision tower: {vision_tower}")
            self.load_model()
        elif getattr(args, "unfreeze_mm_vision_tower", False):
            # TODO: better detector is needed.
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
            self.load_model()
        elif hasattr(args, "mm_tunable_parts") and "mm_vision_tower" in args.mm_tunable_parts:
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`.")
            self.load_model()

    def load_model(self, device_map="auto"):
        rank0_print(f"Loading OpenCLIP model: {self.model_name}")
        rank0_print(f"Pretrained: {self.pretrained}")
        vision_tower, _, image_processor = open_clip.create_model_and_transforms(model_name=self.model_name, pretrained=self.pretrained, precision="fp32", device="cuda")

        resize_transform = [t for t in image_processor.transforms if isinstance(t, torchvision.transforms.Resize)][0]
        normalize_transform = [t for t in image_processor.transforms if isinstance(t, torchvision.transforms.Normalize)][0]
        self.resize_transform_size = resize_transform.size  # 224 or 384
        self.patch_size = vision_tower.visual.conv1.kernel_size[0]  # 14 or 16

        self.image_processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14",
            crop_size=resize_transform.size,
            size={"shortest_edge": resize_transform.size},
            image_mean=list(normalize_transform.mean),
            image_std=list(normalize_transform.std),
        )
        rank0_print(f"Loaded image processor: {self.image_processor}")
        self.vision_tower = vision_tower.visual
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs[self.select_layer]
        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            image_features = image_features
        elif self.select_feature == "conv_flatten":
            image_features = image_features.flatten(2).transpose(1, 2)
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    def forward_visual(self, x, output_hidden_states=False):
        if hasattr(self.vision_tower, "trunk") and hasattr(self.vision_tower.trunk, "_intermediate_layers"):
            return self.vision_tower.trunk._intermediate_layers(x, abs(self.select_layer))
        else:

            def forward_openclip(self, x: torch.Tensor):
                features = []
                x = self.conv1(x)  # shape = [*, width, grid, grid]
                x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
                x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

                # class embeddings and positional embeddings
                x = torch.cat(
                    [_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x],
                    dim=1,
                )
                # shape = [*, grid ** 2 + 1, width]
                x = x + self.positional_embedding.to(x.dtype)

                x = self.patch_dropout(x)
                x = self.ln_pre(x)

                x = x.permute(1, 0, 2)  # NLD -> LND
                for r in self.transformer.resblocks:
                    x = r(x, attn_mask=None)
                    features.append(x)
                return features

            return forward_openclip(self.vision_tower, x)

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.forward_visual(image.to(self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.forward_visual(images.to(self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        if hasattr(self.vision_tower, "conv1"):
            return self.vision_tower.conv1.weight.dtype
        if hasattr(self.vision_tower, "trunk"):
            return self.vision_tower.trunk.patch_embed.proj.weight.dtype
        raise NotImplementedError

    @property
    def device(self):
        if hasattr(self.vision_tower, "conv1"):
            return self.vision_tower.conv1.weight.device
        if hasattr(self.vision_tower, "trunk"):
            return self.vision_tower.trunk.patch_embed.proj.weight.device
        raise NotImplementedError

    @property
    def config(self):
        return None

    @property
    def hidden_size(self):
        if self.model_name in HIDDEN_SIZE_DICT:
            return HIDDEN_SIZE_DICT[self.model_name]
        else:
            raise NotImplementedError

    @property
    def num_patches(self):
        image_size = self.resize_transform_size if isinstance(self.resize_transform_size, int) else self.resize_transform_size[0]
        _num_patches = (image_size // self.patch_size) ** 2
        if "cls_patch" in self.select_feature:
            _num_patches += 1
        return _num_patches

    @property
    def image_size(self):
        return self.resize_transform_size

    @property
    def num_patches_per_side(self):
        return self.resize_transform_size // self.patch_size
