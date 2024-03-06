"""
# Adapted from https://github.com/baaivision/EVA/tree/master/EVA-CLIP
"""

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers.image_processing_utils import BatchFeature
from PIL import Image
from transformers.image_transforms import convert_to_rgb


class BaseProcessor:
    def __init__(self):
        self.transform = lambda x: x
        return

    def __call__(self, item):
        return self.transform(item)


class EvaClipImageBaseProcessor(BaseProcessor):
    def __init__(self, mean=None, std=None):
        self.mean = (0.48145466, 0.4578275, 0.40821073) if mean is None else mean
        self.std = (0.26862954, 0.26130258, 0.27577711) if std is None else std

        self.normalize = transforms.Normalize(self.mean, self.std)

    @property
    def image_mean(self):
        return self.mean


class EvaClipImageTrainProcessor(EvaClipImageBaseProcessor):
    def __init__(self, image_size=224, mean=None, std=None, min_scale=0.5, max_scale=1.0):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                convert_to_rgb,
                transforms.Resize(
                    image_size,
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

        self.image_size = image_size

    def preprocess(self, images, return_tensors):
        if isinstance(images, Image.Image):
            images = [images]
        else:
            assert isinstance(images, list)

        transformed_images = [self.transform(image).numpy() for image in images]
        data = {"pixel_values": transformed_images}

        return BatchFeature(data=data, tensor_type=return_tensors)

    def __call__(self, item):
        return self.transform(item)

    @property
    def crop_size(self):
        return {"height": self.image_size, "width": self.image_size}

    @property
    def size(self):
        return {"shortest_edge": self.image_size}
