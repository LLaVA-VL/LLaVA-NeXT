#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import math
import re
import time
import torch
import torch.nn as nn
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_resampler.builder import build_vision_resampler
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
from llava.utils import rank0_print, rank_print
import random


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower
        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
                self.vision_resampler = [vision_resampler]
            else:
                self.vision_tower = vision_tower
                self.vision_resampler = vision_resampler
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_resampler = self.vision_resampler[0]
                vision_tower = self.vision_tower[0]
            else:
                vision_resampler = self.vision_resampler
                vision_tower = self.vision_tower
            vision_tower.load_model()

            # In case it is frozen by LoRA
            for p in self.vision_resampler.parameters():
                p.requires_grad = True

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_hidden_size = getattr(vision_resampler, "hidden_size", vision_tower.hidden_size)
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        
        if not hasattr(self.config, 'add_faster_video'):
            if model_args.add_faster_video:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.faster_token = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)

            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            rank0_print(f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
            incompatible_keys = self.vision_resampler.load_state_dict(get_w(mm_projector_weights, "vision_resampler"), strict=False)
            rank0_print(f"Loaded vision resampler weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_2dPool(self, image_feature, stride=2):
        height = width = self.get_vision_tower().num_patches_per_side
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        # image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        if self.config.mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "bilinear":
            height, width = image_feature.shape[2:]
            scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
            image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')

        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        # image_features = self.get_model().vision_resampler(image_features, images=images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features
    
    def encode_multimodals(self, videos_or_images, video_idx_in_batch, split_sizes=None):
        rank0_print(f"DEBUG_LOG: LlavaMetaForCausalLM.encode_multimodals entered. Type of videos_or_images: {type(videos_or_images)}")
        if isinstance(videos_or_images, list) and videos_or_images:
            rank0_print(f"DEBUG_LOG: LlavaMetaForCausalLM.encode_multimodals - videos_or_images is list. First element type: {type(videos_or_images[0])}, shape: {videos_or_images[0].shape if hasattr(videos_or_images[0], 'shape') else 'N/A'}")
        elif torch.is_tensor(videos_or_images):
            rank0_print(f"DEBUG_LOG: LlavaMetaForCausalLM.encode_multimodals - videos_or_images is tensor. Shape: {videos_or_images.shape}, Dtype: {videos_or_images.dtype}, Device: {videos_or_images.device}")
        
        videos_or_images_for_tower = videos_or_images # Keep original for other uses if any
        if torch.is_tensor(videos_or_images_for_tower) and videos_or_images_for_tower.dtype == torch.bfloat16:
            rank0_print(f"DEBUG_LOG: LlavaMetaForCausalLM.encode_multimodals - >>>>> Casting videos_or_images_for_tower from bfloat16 to float32 before vision tower call. Shape: {videos_or_images_for_tower.shape} <<<<<")
            videos_or_images_for_tower = videos_or_images_for_tower.to(torch.float32)
            rank0_print(f"DEBUG_LOG: LlavaMetaForCausalLM.encode_multimodals - After casting to float32. Dtype: {videos_or_images_for_tower.dtype}")
        elif isinstance(videos_or_images_for_tower, list) and videos_or_images_for_tower and torch.is_tensor(videos_or_images_for_tower[0]) and videos_or_images_for_tower[0].dtype == torch.bfloat16:
            rank0_print(f"DEBUG_LOG: LlavaMetaForCausalLM.encode_multimodals - >>>>> Casting list of videos_or_images_for_tower from bfloat16 to float32 before vision tower call. <<<<<")
            videos_or_images_for_tower = [tensor.to(torch.float32) for tensor in videos_or_images_for_tower]
            rank0_print(f"DEBUG_LOG: LlavaMetaForCausalLM.encode_multimodals - After casting list to float32. First tensor Dtype: {videos_or_images_for_tower[0].dtype if videos_or_images_for_tower else 'N/A'}")

        videos_or_images_features = None
        try:
            rank0_print(f"DEBUG_LOG: LlavaMetaForCausalLM.encode_multimodals - Before vision_tower call with potentially dtype-casted input.")
            vision_tower_module = self.get_model().get_vision_tower()
            rank0_print(f"DEBUG_LOG: LlavaMetaForCausalLM.encode_multimodals - Vision tower module: {type(vision_tower_module)}")
            videos_or_images_features = vision_tower_module(videos_or_images_for_tower) # Use the potentially casted tensor
            rank0_print(f"DEBUG_LOG: LlavaMetaForCausalLM.encode_multimodals - After vision_tower call. Features type: {type(videos_or_images_features)}, shape: {videos_or_images_features.shape if hasattr(videos_or_images_features, 'shape') else 'N/A'}")
        except Exception as e:
            rank0_print(f"DEBUG_LOG: CRITICAL_ERROR in LlavaMetaForCausalLM.encode_multimodals during vision_tower call: {e}")
            # Potentially re-raise or handle, but for debugging, logging is key.
            # Depending on the error, videos_or_images_features might be None or partially complete.
            # If it's None and the code below expects a tensor, it will fail later.
            # For now, let it proceed to see if a None features tensor causes issues or if it's caught.
            # If the process hangs, this log might not even be reached if the error is too low-level.
            if videos_or_images_features is None and torch.is_tensor(videos_or_images):
                # Create a dummy tensor to prevent immediate crash if rest of the code expects a tensor
                # This is a strong indication of failure in vision tower
                rank0_print(f"DEBUG_LOG: LlavaMetaForCausalLM.encode_multimodals - Vision tower call failed. Creating dummy features.")
                # Dummy features should match expected output dimensions if possible, or be identifiable as dummy.
                # Expected output: (batch_size_images, num_patches, hidden_size_vision_tower)
                # This is hard to get right without knowing exact shapes. For now, a simple small tensor.
                # This part is risky, as an incorrect dummy tensor can cause other errors.
                # A better approach if an error is caught might be to return None and handle it upstream.
                # For now, just logging and letting it potentially fail later if features are None.
                pass # Let videos_or_images_features remain None or as is from the exception context
            # raise # Re-raise the exception to halt execution and get a full traceback if possible

        if videos_or_images_features is None:
            rank0_print(f"DEBUG_LOG: ERROR - videos_or_images_features is None after vision tower call (possibly due to an exception). Cannot proceed with splitting.")
            # Returning empty lists or suitable error indicators for features
            # This might require the calling function to handle this gracefully.
            return [], [] # Or raise an exception

        per_videos_or_images_features = torch.split(videos_or_images_features, split_sizes, dim=0) if split_sizes is not None and sum(split_sizes) == videos_or_images_features.shape[0] else [videos_or_images_features]
        rank0_print(f"DEBUG_LOG: LlavaMetaForCausalLM.encode_multimodals - Split features. Num splits: {len(per_videos_or_images_features)}")
        
        all_videos_or_images_features = []
        all_faster_video_features = []
        cur_mm_spatial_pool_stride = self.config.mm_spatial_pool_stride

        for idx, feat in enumerate(per_videos_or_images_features):
            rank0_print(f"DEBUG_LOG: LlavaMetaForCausalLM.encode_multimodals - Processing feature split {idx}. Shape: {feat.shape if hasattr(feat, 'shape') else 'N/A'}")
            rank0_print(f"DEBUG_LOG: LlavaMetaForCausalLM.encode_multimodals - Before mm_projector for split {idx}.")
            feat = self.get_model().mm_projector(feat)
            rank0_print(f"DEBUG_LOG: LlavaMetaForCausalLM.encode_multimodals - After mm_projector for split {idx}. Shape: {feat.shape if hasattr(feat, 'shape') else 'N/A'}")
            faster_video_feature = 0
            slower_img_feat = 0
            if idx in video_idx_in_batch and cur_mm_spatial_pool_stride > 1:
                rank0_print(f"DEBUG_LOG: LlavaMetaForCausalLM.encode_multimodals - Applying 2D pooling for video split {idx}.")
                slower_img_feat = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
                if self.config.add_faster_video:
                    cur_mm_spatial_pool_stride_faster = cur_mm_spatial_pool_stride * 2 # Corrected variable name
                    faster_video_feature = self.get_2dPool(feat,cur_mm_spatial_pool_stride_faster)
            if slower_img_feat is not 0:
                all_videos_or_images_features.append(slower_img_feat)
            else:
                all_videos_or_images_features.append(feat)
            all_faster_video_features.append(faster_video_feature)
        rank0_print(f"DEBUG_LOG: LlavaMetaForCausalLM.encode_multimodals returning. Num features: {len(all_videos_or_images_features)}")
        return all_videos_or_images_features,all_faster_video_features

    def add_token_per_grid(self, image_feature):
        resize_h = int(math.sqrt(image_feature.shape[1]))
        num_frames = image_feature.shape[0]
        feature_dim = image_feature.shape[-1]

        image_feature = image_feature.view(num_frames, 1, resize_h, resize_h, -1)
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
        image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        if getattr(self.config, "add_faster_video", False):
            # import pdb; pdb.set_trace()
            # (3584, 832, 14) -> (3584, 64, 13, 14)
            image_feature = image_feature.view(feature_dim, num_frames,resize_h, -1)
            #  (3584, 64, 13, 14) -> (64, 13, 14, 3584)
            image_feature = image_feature.permute(1, 2, 3, 0).contiguous()
            # (64, 13, 14, 3584) -> (64, 13*14, 3584)
            image_feature = image_feature.flatten(1, 2)
            # import pdb; pdb.set_trace()
            return image_feature
        # import pdb; pdb.set_trace()
        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
        return image_feature

    def add_token_per_frame(self, image_feature):
        image_feature = image_feature.permute(2, 0, 1).contiguous()
        image_feature =  torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        image_feature = image_feature.permute(1, 2, 0).contiguous()
        return image_feature

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        modalities,
        image_sizes=None
    ):
        rank0_print(f"DEBUG_LOG: LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal entered. input_ids: {input_ids.shape if input_ids is not None else 'None'}, images type: {type(images)}, num images: {len(images) if isinstance(images, list) else (images.shape[0] if torch.is_tensor(images) else 'N/A')}")
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or (isinstance(images, list) and not any(x is not None for x in images)) or (torch.is_tensor(images) and images.nelement() == 0) :
            rank0_print(f"DEBUG_LOG: LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal - No vision tower or no images/all images are None. Returning original inputs.")
            if past_key_values is not None and vision_tower is not None and images is not None and len(images) > 0:
                pass 
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if isinstance(images, list):
            if all(x is None for x in images):
                rank0_print(f"DEBUG_LOG: LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal - All images in list are None. Returning original inputs.")
                return input_ids, position_ids, attention_mask, past_key_values, None, labels
        
        rank0_print(f"DEBUG_LOG: LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal - Processing images. Number of image tensors: {len(images) if isinstance(images, list) else 1}")

        # INFO: image_idx_in_batch is the index of image in batch, not the index of image in all images
        # E.g., if batch size is 2, and the first sample has 1 image, the second sample has 2 images, then image_idx_in_batch is [0, 1, 1]
        image_idx_in_batch = [] 
        video_idx_in_batch = []
        image_idx_counter = -1 
        batch_size = input_ids.shape[0]

        for i in range(batch_size):
            num_images_in_sample = (input_ids[i] == IMAGE_TOKEN_INDEX).sum().item()
            image_idx_counter += num_images_in_sample
            if modalities is not None and len(modalities) > i and modalities[i] == "video":
                video_idx_in_batch.append(image_idx_counter)
            image_idx_in_batch.extend([i] * num_images_in_sample)

        if isinstance(images, list):
            images_tensor_list = [x.to(self.device) for x in images if x is not None] 
            images_tensor_shapes = [x.shape for x in images_tensor_list]
            rank0_print(f"DEBUG_LOG: LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal - 'images' is a list. Num non-None tensors: {len(images_tensor_list)}, Shapes: {images_tensor_shapes}")
            if not images_tensor_list:
                rank0_print(f"DEBUG_LOG: LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal - All images in list were None. Returning original inputs.")
                return input_ids, position_ids, attention_mask, past_key_values, None, labels
            
            # Attempt to concatenate if all shapes are identical and ndim is appropriate
            can_concat = all(x.shape == images_tensor_list[0].shape for x in images_tensor_list) and images_tensor_list[0].ndim == 4 # Expect (C, H, W) or (T, C, H, W) -> after to(device) usually (C,H,W) for image, (T,C,H,W) for video frame list
            if can_concat and images_tensor_list[0].ndim == 3: # list of (C,H,W) images, stack to (N,C,H,W)
                 images_for_encoding = torch.stack(images_tensor_list, dim=0)
                 rank0_print(f"DEBUG_LOG: LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal - Stacked list of image tensors. New shape: {images_for_encoding.shape}")
            elif can_concat and images_tensor_list[0].ndim == 4: # list of (T,C,H,W) video frame sets, concat to (N*T,C,H,W) or handle as list if mixed video/image
                 # This case is more complex: are they all videos? are Ts the same? For now, let encode_multimodals handle list if complex.
                 # Assuming for now they are all same T and should be concatenated or handled by encode_multimodals as a list.
                 images_for_encoding = images_tensor_list # Pass as list to encode_multimodals if complex
                 rank0_print(f"DEBUG_LOG: LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal - 'images' is list of 4D tensors. Passing as list to encode_multimodals.")
            else: # Different shapes or not 3D/4D, pass as list
                images_for_encoding = images_tensor_list
                rank0_print(f"DEBUG_LOG: LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal - Image tensors in list have different shapes or unexpected ndim. Passing as list to encode_multimodals.")

        elif torch.is_tensor(images):
            images_for_encoding = images.to(self.device)
            rank0_print(f"DEBUG_LOG: LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal - 'images' is a tensor. Shape: {images_for_encoding.shape}")
        else:
            rank0_print(f"DEBUG_LOG: ERROR - 'images' is of unexpected type: {type(images)}. Returning original inputs.")
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        
        rank0_print(f"DEBUG_LOG: LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal - Before encode_multimodals call.")
        image_features_tuple = self.encode_multimodals(images_for_encoding, video_idx_in_batch, torch.bincount(torch.tensor(image_idx_in_batch)).tolist() if image_idx_in_batch else None)
        image_features = image_features_tuple[0] # Assuming first element is the primary features
        # faster_video_features = image_features_tuple[1]
        rank0_print(f"DEBUG_LOG: LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal - After encode_multimodals. Number of feature tensors from encode_multimodals: {len(image_features) if isinstance(image_features, list) else 'N/A (not a list)'}")
        if image_features and isinstance(image_features, list) and image_features[0] is not None:
            rank0_print(f"DEBUG_LOG: LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal - First image_feature shape: {image_features[0].shape}")
        elif torch.is_tensor(image_features):
             rank0_print(f"DEBUG_LOG: LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal - image_features is tensor, shape: {image_features.shape}")

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        rank0_print(f"DEBUG_LOG: LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal - Starting to iterate through batch for embedding construction. Batch size: {input_ids.shape[0]}")

        for batch_idx, cur_input_ids in enumerate(input_ids):
            # ... (inner loop logic) ...
            # Add a log inside the loop for each sample
            rank0_print(f"DEBUG_LOG: LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal - Processing batch_idx {batch_idx} for multimodal embedding construction.")
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds_segments = []
            cur_new_labels_segments = [] if labels is not None else None
            text_segment_start_idx = 0

            if not image_features and len(image_token_indices) > 0:
                rank0_print(f"DEBUG_LOG: ERROR - batch_idx {batch_idx} has image tokens but image_features is empty or None. This should not happen.")
                # Fallback or raise error
                # For now, just embed text and skip image tokens to avoid crashing, but this is an error state.
                text_features_all = self.get_model().embed_tokens(cur_input_ids)
                cur_new_input_embeds_segments.append(text_features_all)
                if labels is not None:
                    cur_new_labels_segments.append(labels[batch_idx])
            elif len(image_token_indices) == 0:
                text_features_all = self.get_model().embed_tokens(cur_input_ids)
                cur_new_input_embeds_segments.append(text_features_all)
                if labels is not None:
                    cur_new_labels_segments.append(labels[batch_idx])
            else:
                text_features_all = self.get_model().embed_tokens(cur_input_ids)
                for k, image_token_idx in enumerate(image_token_indices):
                    if image_token_idx > text_segment_start_idx:
                        cur_new_input_embeds_segments.append(text_features_all[text_segment_start_idx:image_token_idx])
                        if labels is not None:
                            cur_new_labels_segments.append(labels[batch_idx][text_segment_start_idx:image_token_idx])
                    
                    if cur_image_idx < len(image_features):
                        image_feature_current = image_features[cur_image_idx]
                        cur_new_input_embeds_segments.append(image_feature_current)
                        if labels is not None:
                            num_image_feature_tokens = image_feature_current.shape[0]
                            cur_new_labels_segments.append(torch.full((num_image_feature_tokens,), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                    else:
                        rank0_print(f"DEBUG_LOG: ERROR - batch_idx {batch_idx}, k={k}. cur_image_idx {cur_image_idx} out of bounds for image_features (len {len(image_features)}). Skipping image insert.")
                        # Optionally append a placeholder or handle error, for now, it effectively skips this image token's features.
                    
                    text_segment_start_idx = image_token_idx + 1
                    cur_image_idx += 1

                if text_segment_start_idx < len(cur_input_ids):
                    cur_new_input_embeds_segments.append(text_features_all[text_segment_start_idx:])
                    if labels is not None:
                        cur_new_labels_segments.append(labels[batch_idx][text_segment_start_idx:])
            
            if cur_new_input_embeds_segments:
                 new_input_embeds.append(torch.cat(cur_new_input_embeds_segments, dim=0))
                 if labels is not None and cur_new_labels_segments:
                     new_labels.append(torch.cat(cur_new_labels_segments, dim=0))
                 elif labels is not None: # No label segments but labels are expected
                     rank0_print(f"DEBUG_LOG: WARNING - batch_idx {batch_idx} had embed segments but no label segments created.")
                     # This might happen if all were image tokens and no text, or an error. Add a dummy if needed by padding logic.
                     # For now, let padding handle it or error later if lengths mismatch.
            else:
                rank0_print(f"DEBUG_LOG: WARNING - batch_idx {batch_idx} resulted in no embedding segments.")
                # Add a dummy zero embedding if necessary to prevent padding issues, matching hidden_size
                # This is a fallback and indicates an issue.
                # new_input_embeds.append(torch.zeros((0, self.config.hidden_size), dtype=self.dtype, device=self.device))
                # if labels is not None:
                #    new_labels.append(torch.zeros((0,), dtype=torch.long, device=self.device))


        rank0_print(f"DEBUG_LOG: LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal - Finished iterating for embedding construction. Num new_input_embeds: {len(new_input_embeds)}")
        
        if not new_input_embeds: # If all samples were problematic and resulted in no embeds
            rank0_print(f"DEBUG_LOG: ERROR - No valid input embeddings could be constructed for any sample in the batch. Returning Nones.")
            return None, None, attention_mask, past_key_values, None, None # Or handle error more gracefully

        # ... (padding logic - assume correct for now, but add start/end logs)
        rank0_print(f"DEBUG_LOG: LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal - Starting padding logic.")
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', 2048)
        new_input_embeds = [embed[:tokenizer_model_max_length] for embed in new_input_embeds]
        if labels is not None and new_labels:
            new_labels = [label[:tokenizer_model_max_length] for label in new_labels]
        elif labels is not None and not new_labels: # if labels were expected but not generated (e.g. all image batch)
             rank0_print(f"DEBUG_LOG: WARNING - Labels were expected but new_labels list is empty before padding.")
             # new_labels will remain empty, padding logic for labels might need to handle this or error

        max_len = max(x.shape[0] for x in new_input_embeds) if new_input_embeds else 0
        padded_input_embeds = torch.zeros((len(new_input_embeds), max_len, new_input_embeds[0].shape[1]), dtype=new_input_embeds[0].dtype, device=self.device)
        padded_labels = None
        if labels is not None and new_labels: # Only pad labels if they were generated
            padded_labels = torch.full((len(new_labels), max_len), IGNORE_INDEX, dtype=torch.long, device=self.device)
            for i, label_seq in enumerate(new_labels):
                padded_labels[i, :label_seq.shape[0]] = label_seq
        elif labels is not None: # Labels expected but not generated, create IGNORE_INDEX labels
            rank0_print(f"DEBUG_LOG: LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal - Labels expected but not generated, creating full IGNORE_INDEX labels of shape ({len(new_input_embeds)}, {max_len})")
            padded_labels = torch.full((len(new_input_embeds), max_len), IGNORE_INDEX, dtype=torch.long, device=self.device)

        new_attention_mask = torch.zeros((len(new_input_embeds), max_len), dtype=attention_mask.dtype, device=attention_mask.device) if attention_mask is not None else None
        for i, embed_seq in enumerate(new_input_embeds):
            padded_input_embeds[i, :embed_seq.shape[0]] = embed_seq
            if new_attention_mask is not None:
                new_attention_mask[i, :embed_seq.shape[0]] = 1
        inputs_embeds = padded_input_embeds
        labels = padded_labels
        attention_mask = new_attention_mask
        input_ids = None 
        rank0_print(f"DEBUG_LOG: LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal - Padding done. Final inputs_embeds: {inputs_embeds.shape if inputs_embeds is not None else 'None'}")

        rank0_print(f"DEBUG_LOG: LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal - Returning. inputs_embeds shape: {inputs_embeds.shape if inputs_embeds is not None else 'None'}, labels shape: {labels.shape if labels is not None else 'None'}, attention_mask shape: {attention_mask.shape if attention_mask is not None else 'None'}")
        return input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
