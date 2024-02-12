import torch
import torch.nn as nn
import torchvision

from transformers import CLIPImageProcessor


try:
    import open_clip
    from open_clip.transformer import _expand_token
except ImportError:
    open_clip = None


class OpenCLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.hub_name = vision_tower
        self.pretrained = args.vision_tower_pretrained
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()

    def load_model(self):
        vision_tower, _, image_processor = open_clip.create_model_and_transforms(model_name=self.hub_name, pretrained=self.pretrained)
        resize_transform = [t for t in image_processor.transforms if isinstance(t, torchvision.transforms.Resize)][0]
        normalize_transform = [t for t in image_processor.transforms if isinstance(t, torchvision.transforms.Normalize)][0]
        self.resize_transform_size = resize_transform.size
        self.image_processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14",
            crop_size=resize_transform.size,
            size={"shortest_edge": resize_transform.size},
            image_mean=list(normalize_transform.mean),
            image_std=list(normalize_transform.std),
        )
        self.vision_tower = vision_tower.visual
        self.vision_tower.requires_grad_(False)
        self.vision_tower.eval()
        self.is_loaded = True

    def train(self, mode = True):
        self.training = mode

        if self.is_loaded:
            self.vision_tower.eval()

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        elif self.select_feature == 'conv_flatten':
            image_features = image_features.flatten(2).transpose(1, 2)
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def forward_visual(self, x, output_hidden_states=False):
        if hasattr(self.vision_tower, 'trunk'):
            return self.vision_tower.trunk._intermediate_layers(x, abs(self.select_layer))
        else:
            def forward_openclip(self, x: torch.Tensor):
                features = []
                x = self.conv1(x)  # shape = [*, width, grid, grid]
                x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
                x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

                # class embeddings and positional embeddings
                x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
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

    @torch.no_grad()
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
        if hasattr(self.vision_tower, 'conv1'):
            return self.vision_tower.conv1.weight.dtype
        if hasattr(self.vision_tower, 'trunk'):
            return self.vision_tower.trunk.patch_embed.proj.weight.dtype
        raise NotImplementedError

    @property
    def device(self):
        if hasattr(self.vision_tower, 'conv1'):
            return self.vision_tower.conv1.weight.device
        if hasattr(self.vision_tower, 'trunk'):
            return self.vision_tower.trunk.patch_embed.proj.weight.device
        raise NotImplementedError

    @property
    def config(self):
        return None

    @property
    def hidden_size(self):
        if hasattr(self.vision_tower, 'ln_post'):
            return self.vision_tower.ln_post.weight.shape[0]
        if hasattr(self.vision_tower, 'trunk'):
            return self.vision_tower.trunk.num_features
        raise NotImplementedError

    @property
    def num_patches(self):
        raise NotImplementedError
