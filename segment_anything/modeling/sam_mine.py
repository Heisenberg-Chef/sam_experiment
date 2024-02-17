# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from segment_anything.modeling.common import LayerNorm2d
from .image_encoder import ImageEncoderViT
from .mask_decoder_mine import MaskDecoderMine
from .prompt_encoder import PromptEncoder


class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
            self,
            image_encoder: ImageEncoderViT,
            prompt_encoder: PromptEncoder,
            mask_decoder: MaskDecoderMine,
            pixel_mean: List[float] = [123.675, 116.28, 103.53],
            pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        ##############################
        # mine
        ##############################
        dim = self.image_encoder.dim
        self.cascade_vit_1 = nn.Sequential(
            nn.ConvTranspose2d(dim, 256, 1),
            nn.GELU(),
            LayerNorm2d(256)
        )

        self.cascade_vit_2 = nn.Sequential(
            nn.ConvTranspose2d(dim, 256, 1),
            nn.GELU(),
            LayerNorm2d(256)
        )

        self.cascade_vit_3 = nn.Sequential(
            nn.ConvTranspose2d(dim, 256, 1),
            nn.GELU(),
            LayerNorm2d(256)
        )

        self.cascade_vit_4 = nn.Sequential(
            nn.ConvTranspose2d(dim, 256, 1),
            nn.GELU(),
            LayerNorm2d(256)
        )

        self.vit_layer_norm = LayerNorm2d(256)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(
            self,
            batched_input: List[Dict[str, Any]],
            multimask_output: bool,
            hq_token_only: bool = False,
    ) -> List[Dict[str, torch.Tensor]]:
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings, interm_embeddings = self.image_encoder(input_images)
        vit_features_1 = interm_embeddings[0].permute(0, 3, 1, 2)
        vit_features_2 = interm_embeddings[1].permute(0, 3, 1, 2)
        vit_features_3 = interm_embeddings[2].permute(0, 3, 1, 2)
        vit_features_4 = interm_embeddings[3].permute(0, 3, 1, 2)

        l_1 = self.cascade_vit_1(vit_features_1)
        l_2 = self.cascade_vit_2(vit_features_2)
        l_3 = self.cascade_vit_3(vit_features_3)
        l_4 = self.cascade_vit_4(vit_features_4)
        interm_embeddings = self.vit_layer_norm(image_embeddings + l_1 + l_2 + l_3 + l_4)
        outputs = []
        for image_record, curr_embedding, curr_interm in zip(batched_input, image_embeddings, interm_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
                hq_token_only=hq_token_only,
                interm_embeddings=curr_interm.unsqueeze(0),
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            # Boolean couldn't be tracked
            # masks = masks > self.mask_threshold
            masks = torch.clamp(masks, min=0, max=1)
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

    def postprocess_masks(
            self,
            masks: torch.Tensor,
            input_size: Tuple[int, ...],
            original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
