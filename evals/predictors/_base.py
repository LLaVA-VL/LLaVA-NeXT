import torch
from abc import ABC, abstractmethod
import numpy as np
from PIL import Image

import warnings
warnings.filterwarnings("ignore", category=UserWarning)



class Predict(ABC):
    def __init__(self, device="cuda:0"):
        self.device = torch.device(device)
        self.history = None
        self.IMAGE_TOKEN = None
        
    @abstractmethod
    def predict(self, question, images):
        pass
    
    @abstractmethod
    def process_image(self, img_path):
        pass
    