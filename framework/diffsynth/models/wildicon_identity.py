import math
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _log_once(module: nn.Module, key: str, message: str):
    logged = getattr(module, "_wildicon_logged_messages", None)
    if logged is None:
        logged = set()
        setattr(module, "_wildicon_logged_messages", logged)
    if key not in logged:
        print(message)
        logged.add(key)


def _shape_str(tensor: Optional[torch.Tensor]) -> str:
    if tensor is None:
        return "None"
    return str(tuple(tensor.shape))


class CenterPriorSegmentor(nn.Module):
    def __init__(self, radius_x: float = 0.9, radius_y: float = 0.9, sharpness: float = 8.0):
        super().__init__()
        self.radius_x = radius_x
        self.radius_y = radius_y
        self.sharpness = sharpness

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        _, _, height, width = image.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=image.device, dtype=image.dtype),
            torch.linspace(-1.0, 1.0, width, device=image.device, dtype=image.dtype),
            indexing="ij",
        )
        distance = torch.sqrt((xx / self.radius_x) ** 2 + (yy / self.radius_y) ** 2)
        mask = torch.sigmoid((1.0 - distance) * self.sharpness).unsqueeze(0).unsqueeze(0)
        return image * mask


class TorchvisionDeepLabSegmentor(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50

        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        self.model = deeplabv3_resnet50(weights=weights)
        self.model.eval()
        self.model.requires_grad_(False)
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1), persistent=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        pixel_values = (image + 1.0) / 2.0
        pixel_values = (pixel_values - self.mean) / self.std
        logits = self.model(pixel_values)["out"]
        foreground = 1.0 - logits.softmax(dim=1)[:, :1]
        foreground = F.interpolate(foreground, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return image * foreground


def build_segmentor(segmentor_type: str) -> nn.Module:
    if segmentor_type == "external":
        return None
    if segmentor_type == "center_prior":
        return CenterPriorSegmentor()
    if segmentor_type == "deeplabv3_resnet50":
        return TorchvisionDeepLabSegmentor()
    raise ValueError(f"Unsupported WildIcon segmentor: {segmentor_type}")


class GaussianHighPass(nn.Module):
    def __init__(self, sigma: float = 3.0):
        super().__init__()
        self.sigma = sigma
        if sigma <= 0:
            self.register_buffer("kernel", torch.ones(1, 1, 1, 1), persistent=False)
            self.padding = 0
            return
        radius = max(1, int(math.ceil(3 * sigma)))
        coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        self.register_buffer("kernel", kernel_2d.unsqueeze(0).unsqueeze(0), persistent=False)
        self.padding = radius

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self.sigma <= 0:
            return torch.zeros_like(image)
        kernel = self.kernel.to(dtype=image.dtype, device=image.device).expand(image.shape[1], 1, -1, -1)
        blurred = F.conv2d(image, kernel, padding=self.padding, groups=image.shape[1])
        return image - blurred


class IdentityEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        hidden_dim: int = 128,
        token_dim: int = 256,
        token_grid_size: Tuple[int, int] = (4, 4),
    ):
        super().__init__()
        self.token_grid_size = token_grid_size
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim * 2),
            nn.SiLU(),
            nn.Conv2d(hidden_dim * 2, token_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, token_dim),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(token_grid_size),
        )
        self.output_norm = nn.LayerNorm(token_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        tokens = features.flatten(2).transpose(1, 2)
        return self.output_norm(tokens)


class DINOv3BackboneEncoder(nn.Module):
    def __init__(
        self,
        model_name_or_path: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        token_grid_size: Tuple[int, int] = (4, 4),
        train_mode: str = "frozen",
        trainable_layers: int = 1,
        local_files_only: bool = True,
    ):
        super().__init__()
        from transformers import DINOv3ViTModel

        self.model_name_or_path = model_name_or_path
        self.token_grid_size = token_grid_size
        self.train_mode = train_mode
        self.trainable_layers = max(1, int(trainable_layers))
        self.model = DINOv3ViTModel.from_pretrained(model_name_or_path, local_files_only=local_files_only)
        image_size = self.model.config.image_size
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = tuple(image_size)
        self.hidden_size = self.model.config.hidden_size
        self.num_register_tokens = getattr(self.model.config, "num_register_tokens", 0)
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1), persistent=False)
        self.apply_training_policy()

    def apply_training_policy(self):
        self.model.requires_grad_(False)
        if self.train_mode == "full":
            self.model.requires_grad_(True)
        elif self.train_mode == "last_n":
            for layer in self.model.layer[-self.trainable_layers:]:
                layer.requires_grad_(True)
            self.model.norm.requires_grad_(True)
        elif self.train_mode != "frozen":
            raise ValueError(f"Unsupported DINOv3 train mode: {self.train_mode}")
        self._refresh_module_modes()

    def _refresh_module_modes(self):
        if self.train_mode == "frozen":
            self.model.eval()
        else:
            self.model.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        self._refresh_module_modes()
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        pixel_values = F.interpolate((image.float() + 1.0) / 2.0, size=self.image_size, mode="bicubic", align_corners=False)
        pixel_values = (pixel_values - self.mean) / self.std
        sequence_output = self.model(pixel_values=pixel_values, return_dict=True).last_hidden_state
        patch_tokens = sequence_output[:, 1 + self.num_register_tokens :, :]
        batch_size, num_tokens, channels = patch_tokens.shape
        spatial_size = int(math.isqrt(num_tokens))
        if spatial_size * spatial_size != num_tokens:
            raise ValueError(f"DINOv3 patch token count {num_tokens} does not form a square grid.")
        feature_map = patch_tokens.transpose(1, 2).reshape(batch_size, channels, spatial_size, spatial_size)
        pooled = F.adaptive_avg_pool2d(feature_map, self.token_grid_size)
        return pooled.flatten(2).transpose(1, 2)


class IdentityFusionHead(nn.Module):
    def __init__(self, visual_dim: int, highfreq_dim: int, token_dim: int):
        super().__init__()
        fused_dim = visual_dim + highfreq_dim
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
            nn.LayerNorm(token_dim),
        )

    def forward(self, visual_tokens: torch.Tensor, highfreq_tokens: torch.Tensor) -> torch.Tensor:
        return self.fusion(torch.cat([visual_tokens, highfreq_tokens], dim=-1))


class IdentityProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.projector(tokens)


class WildIconIdentityAdapter(nn.Module):
    def __init__(
        self,
        context_dim: int,
        selected_block_ids: Iterable[int],
        segmentor_type: str = "center_prior",
        highpass_sigma: float = 3.0,
        encoder_hidden_dim: int = 128,
        encoder_token_dim: int = 256,
        token_grid_size: Tuple[int, int] = (4, 4),
        encoder_type: str = "light_cnn",
        backbone_model_name_or_path: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        backbone_train_mode: str = "frozen",
        backbone_trainable_layers: int = 1,
        backbone_local_files_only: bool = True,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.segmentor_type = segmentor_type
        self.segmentor = build_segmentor(segmentor_type)
        self.highpass = GaussianHighPass(highpass_sigma)
        self.identity_texture_encoder = None
        self.identity_head = None
        if encoder_type == "light_cnn":
            self.identity_encoder = IdentityEncoder(
                in_channels=6,
                hidden_dim=encoder_hidden_dim,
                token_dim=encoder_token_dim,
                token_grid_size=token_grid_size,
            )
        elif encoder_type == "dinov3":
            self.identity_encoder = DINOv3BackboneEncoder(
                model_name_or_path=backbone_model_name_or_path,
                token_grid_size=token_grid_size,
                train_mode=backbone_train_mode,
                trainable_layers=backbone_trainable_layers,
                local_files_only=backbone_local_files_only,
            )
            self.identity_texture_encoder = IdentityEncoder(
                in_channels=3,
                hidden_dim=encoder_hidden_dim,
                token_dim=encoder_token_dim,
                token_grid_size=token_grid_size,
            )
            self.identity_head = IdentityFusionHead(
                visual_dim=self.identity_encoder.hidden_size,
                highfreq_dim=encoder_token_dim,
                token_dim=encoder_token_dim,
            )
        else:
            raise ValueError(f"Unsupported WildIcon encoder type: {encoder_type}")
        self.identity_projector = IdentityProjector(encoder_token_dim, context_dim)
        self.selected_block_ids = tuple(sorted(set(int(block_id) for block_id in selected_block_ids)))
        self.apply_training_policy()

    def freeze_segmentor(self):
        if self.segmentor is None:
            return
        self.segmentor.eval()
        self.segmentor.requires_grad_(False)

    def apply_training_policy(self):
        self.freeze_segmentor()
        if hasattr(self.identity_encoder, "apply_training_policy"):
            self.identity_encoder.apply_training_policy()

    def train(self, mode: bool = True):
        super().train(mode)
        self.apply_training_policy()
        return self

    def build_foreground(
        self,
        image: Optional[torch.Tensor] = None,
        segmented_image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if segmented_image is not None:
            _log_once(self, "starting_foreground_external", "starting wildicon foreground stage: using pre-segmented reference image")
            segmented = segmented_image.float()
            if segmented.shape[1] == 1:
                if image is None:
                    segmented = segmented.repeat(1, 3, 1, 1)
                else:
                    segmented = image.float() * segmented
            elif segmented.shape[1] == 4:
                alpha = segmented[:, 3:4]
                segmented = segmented[:, :3] * alpha
            return segmented
        if image is None:
            raise ValueError("WildIconIdentityAdapter requires either `image` or `segmented_image`.")
        if self.segmentor is None:
            raise ValueError("WildIconIdentityAdapter with `segmentor_type=\"external\"` requires `segmented_image`.")
        self.freeze_segmentor()
        _log_once(self, "starting_foreground_segmentor", f"starting wildicon foreground stage: segmentor_type={self.segmentor_type}")
        return self.segmentor(image.float())

    def extract_identity_features(
        self,
        image: Optional[torch.Tensor] = None,
        segmented_image: Optional[torch.Tensor] = None,
        return_debug: bool = False,
    ):
        segmented = self.build_foreground(image=image, segmented_image=segmented_image)
        _log_once(self, "starting_high_frequency", f"starting wildicon high-frequency stage: gaussian_highpass_sigma={self.highpass.sigma}")
        high_frequency = self.highpass(segmented)
        debug = {
            "segmented": segmented,
            "high_frequency": high_frequency,
        }
        if self.encoder_type == "light_cnn":
            _log_once(self, "starting_identity_encoder_light_cnn", "starting wildicon identity encoder stage: light_cnn on concat(I_seg, I_hf)")
            identity_input = torch.cat([segmented, high_frequency], dim=1)
            tokens = self.identity_encoder(identity_input)
            _log_once(
                self,
                "wildicon_light_cnn_shapes",
                f"wildicon debug shapes: segmented={_shape_str(segmented)}, high_frequency={_shape_str(high_frequency)}, identity_input={_shape_str(identity_input)}, identity_tokens={_shape_str(tokens)}",
            )
            debug["identity_input"] = identity_input
        else:
            _log_once(self, "starting_identity_encoder_dinov3", "starting wildicon identity encoder stage: I_seg -> DINOv3, I_hf -> CNN texture encoder, then token fusion")
            backbone_tokens = self.identity_encoder(segmented)
            highfreq_tokens = self.identity_texture_encoder(high_frequency)
            _log_once(self, "starting_identity_fusion", "starting wildicon identity fusion stage: concat visual tokens and high-frequency tokens, then fuse")
            tokens = self.identity_head(backbone_tokens, highfreq_tokens)
            _log_once(
                self,
                "wildicon_dinov3_shapes",
                f"wildicon debug shapes: segmented={_shape_str(segmented)}, high_frequency={_shape_str(high_frequency)}, backbone_tokens={_shape_str(backbone_tokens)}, high_frequency_tokens={_shape_str(highfreq_tokens)}, identity_tokens={_shape_str(tokens)}",
            )
            debug["backbone_tokens"] = backbone_tokens
            debug["high_frequency_tokens"] = highfreq_tokens
        _log_once(self, "starting_identity_projector", "starting wildicon identity projector stage: projecting identity tokens into Wan text-conditioning space")
        projected_tokens = self.identity_projector(tokens)
        _log_once(
            self,
            "wildicon_projected_shapes",
            f"wildicon debug shapes: projected_identity_context={_shape_str(projected_tokens)}, selected_block_ids={self.selected_block_ids}",
        )
        if return_debug:
            debug["identity_tokens"] = tokens
            debug["projected_tokens"] = projected_tokens
            return debug
        return projected_tokens

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        segmented_image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.extract_identity_features(image=image, segmented_image=segmented_image, return_debug=False)
