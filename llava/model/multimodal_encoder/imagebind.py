import torch
import torch.nn as nn

from transformers import CLIPImageProcessor

try:
    from imagebind.models import imagebind_model
    from imagebind.models.imagebind_model import ModalityType
    from imagebind.data import load_and_transform_audio_data
except ImportError:
    pass


class ImageBindWrapper(nn.Module):
    def __init__(self, vision_tower, select_layer, select_feature="patch", delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = select_layer
        self.select_feature = select_feature

        if not delay_load:
            self.load_model()

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.vision_tower = imagebind_model.imagebind_huge(pretrained=True)
        for p in self.vision_tower.parameters():
            p.requires_grad = False
        self.vision_tower.eval()
        self.is_loaded = True

    def train(self, mode=True):
        self.training = mode

        if self.is_loaded:
            self.vision_tower.eval()

    @torch.no_grad()
    def forward(self, x):
        if type(x) == dict:
            if x["audios"] is not None:
                inputs = {ModalityType.AUDIO: load_and_transform_audio_data(x["audios"], device=self.device).half()}
                embeddings = self.vision_tower(inputs)
                audio_embedding = embeddings[ModalityType.AUDIO]
                return audio_embedding.unsqueeze(1)
        else:
            inputs = {ModalityType.VISION: x.to(dtype=self.dtype)}
            embeddings = self.vision_tower(inputs)
            vision_embedding = embeddings[ModalityType.VISION]
            if vision_embedding.ndim == 2:
                return vision_embedding.unsqueeze(1)
            if vision_embedding.shape[1] == 257:
                return vision_embedding[:, 1:]
            raise ValueError(f"Unexpected shape: {vision_embedding.shape}")

    @property
    def dummy_feature(self):
        return torch.zeros(1, 1024, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.modality_preprocessors.vision.cls_token.dtype

    @property
    def device(self):
        return self.vision_tower.modality_preprocessors.vision.cls_token.device

    @property
    def hidden_size(self):
        return 1024
