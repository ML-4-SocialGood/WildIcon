import torch, os, argparse, accelerate, warnings
import torch.nn.functional as F
from diffsynth.core import UnifiedDataset, load_state_dict
from diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
from diffsynth.models.wildicon_identity import WildIconIdentityAdapter
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def log_main_process(message: str, accelerator=None):
    if accelerator is None or accelerator.is_main_process:
        print(message)


def parse_int_list(value):
    if value is None or value == "":
        return None
    return tuple(int(item.strip()) for item in value.split(",") if item.strip() != "")


def parse_grid_size(value, default=(4, 4)):
    if value is None or value == "":
        return default
    parts = [int(item.strip()) for item in value.split(",") if item.strip() != ""]
    if len(parts) != 2:
        raise ValueError(f"Invalid WildIcon token grid size: {value}. Expected 'H,W'.")
    return parts[0], parts[1]


class FrozenIdentityFeatureLoss(torch.nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        local_files_only: bool = True,
        image_size: int | None = None,
    ):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        self.config = AutoConfig.from_pretrained(model_name_or_path, local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(model_name_or_path, local_files_only=local_files_only)
        self.model.eval()
        self.model.requires_grad_(False)
        configured_size = getattr(self.config, "image_size", 224)
        if isinstance(configured_size, (tuple, list)):
            configured_size = configured_size[0]
        self.image_size = int(image_size or configured_size or 224)
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1), persistent=False)

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        pixel_values = (image.float() + 1.0) / 2.0
        pixel_values = F.interpolate(pixel_values, size=(self.image_size, self.image_size), mode="bicubic", align_corners=False)
        pixel_values = (pixel_values - self.mean) / self.std
        outputs = self.model(pixel_values=pixel_values, return_dict=True)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state[:, 0]
        return F.normalize(features.float(), dim=-1)


def _scheduler_sigma(pipe, timestep):
    timestep_id = torch.argmin((pipe.scheduler.timesteps.to(timestep.device) - timestep).abs())
    return pipe.scheduler.sigmas.to(device=timestep.device, dtype=timestep.dtype)[timestep_id]


def _sample_video_frames(video_tensor: torch.Tensor, num_frames: int) -> torch.Tensor:
    frame_count = video_tensor.shape[2]
    if frame_count <= num_frames:
        indices = torch.arange(frame_count, device=video_tensor.device)
    else:
        indices = torch.linspace(0, frame_count - 1, steps=num_frames, device=video_tensor.device).round().long()
    sampled = video_tensor[:, :, indices]
    return sampled.permute(0, 2, 1, 3, 4).reshape(-1, video_tensor.shape[1], video_tensor.shape[3], video_tensor.shape[4])


def _reference_to_tensor(pipe, reference_image, height: int, width: int):
    if isinstance(reference_image, list):
        reference_image = reference_image[0]
    if isinstance(reference_image, torch.Tensor):
        reference = reference_image
        if reference.ndim == 3:
            reference = reference.unsqueeze(0)
        return reference.to(dtype=pipe.torch_dtype, device=pipe.device)
    return pipe.preprocess_image(
        reference_image.resize((width, height)),
        torch_dtype=pipe.torch_dtype,
        device=pipe.device,
    )


def WildIconFlowMatchSFTLoss(
    pipe,
    identity_loss_fn=None,
    identity_loss_weight: float = 0.0,
    identity_loss_num_frames: int = 4,
    **inputs,
):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)

    if "first_frame_latents" in inputs:
        inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)

    identity_loss = None
    if identity_loss_fn is not None and identity_loss_weight > 0 and inputs.get("identity_reference_image") is not None:
        sigma = _scheduler_sigma(pipe, timestep).to(dtype=noise_pred.dtype, device=noise_pred.device)
        clean_latents = inputs["latents"] - sigma * noise_pred
        decoded = pipe.vae.decode(
            clean_latents,
            device=pipe.device,
            tiled=inputs.get("tiled", False),
            tile_size=inputs.get("tile_size", 128),
            tile_stride=inputs.get("tile_stride", 64),
        )
        generated_frames = _sample_video_frames(decoded, identity_loss_num_frames)
        reference = _reference_to_tensor(
            pipe,
            inputs["identity_reference_image"],
            height=decoded.shape[-2],
            width=decoded.shape[-1],
        )
        generated_features = identity_loss_fn(generated_frames)
        reference_features = identity_loss_fn(reference).repeat_interleave(generated_frames.shape[0] // reference.shape[0], dim=0)
        identity_loss = 1.0 - (generated_features * reference_features).sum(dim=-1).mean()

    if "first_frame_latents" in inputs:
        noise_pred = noise_pred[:, :, 1:]
        training_target = training_target[:, :, 1:]

    flow_loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    flow_loss = flow_loss * pipe.scheduler.training_weight(timestep)
    if identity_loss is not None:
        return flow_loss + float(identity_loss_weight) * identity_loss
    return flow_loss


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        redirect_common_files=True,
        wildicon_enabled=False,
        wildicon_selected_block_ids=None,
        wildicon_segmentor_type="center_prior",
        wildicon_highpass_sigma=3.0,
        wildicon_encoder_hidden_dim=128,
        wildicon_encoder_token_dim=256,
        wildicon_token_grid_size=(4, 4),
        wildicon_encoder_type="light_cnn",
        wildicon_backbone_model_name_or_path="facebook/dinov3-vitl16-pretrain-lvd1689m",
        wildicon_backbone_train_mode="frozen",
        wildicon_backbone_trainable_layers=1,
        wildicon_backbone_local_files_only=True,
        wildicon_identity_loss_weight=0.0,
        wildicon_identity_loss_model_name_or_path=None,
        wildicon_identity_loss_num_frames=4,
        wildicon_identity_loss_local_files_only=True,
        resume_trainable_checkpoint=None,
    ):
        super().__init__()
        print("starting wan training module initialization")
        # Warning
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True

        if wildicon_enabled:
            trainable_model_names = [] if trainable_models is None else [name for name in trainable_models.split(",") if name != ""]
            if "wildicon_adapter" not in trainable_model_names:
                trainable_model_names.append("wildicon_adapter")
            trainable_models = ",".join(trainable_model_names)
        
        # Load models
        print("starting wan pipeline loading")
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        audio_processor_config = self.parse_path_or_model_id(audio_processor_path)
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            audio_processor_config=audio_processor_config,
            redirect_common_files=redirect_common_files,
        )
        if wildicon_enabled:
            print("starting wildicon adapter attachment")
            self.attach_wildicon_adapter(
                device=device,
                selected_block_ids=wildicon_selected_block_ids,
                segmentor_type=wildicon_segmentor_type,
                highpass_sigma=wildicon_highpass_sigma,
                encoder_hidden_dim=wildicon_encoder_hidden_dim,
                encoder_token_dim=wildicon_encoder_token_dim,
                token_grid_size=wildicon_token_grid_size,
                encoder_type=wildicon_encoder_type,
                backbone_model_name_or_path=wildicon_backbone_model_name_or_path,
                backbone_train_mode=wildicon_backbone_train_mode,
                backbone_trainable_layers=wildicon_backbone_trainable_layers,
                backbone_local_files_only=wildicon_backbone_local_files_only,
            )
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        
        # Training mode
        print("starting training mode configuration")
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        if self.pipe.wildicon_adapter is not None:
            self.pipe.wildicon_adapter.apply_training_policy()
        if resume_trainable_checkpoint is not None:
            self.resume_trainable_checkpoint(resume_trainable_checkpoint, device=device)

        self.wildicon_identity_loss_weight = float(wildicon_identity_loss_weight)
        self.wildicon_identity_loss_num_frames = int(wildicon_identity_loss_num_frames)
        self.wildicon_identity_loss = None
        if wildicon_enabled and self.wildicon_identity_loss_weight > 0:
            identity_loss_model = wildicon_identity_loss_model_name_or_path or wildicon_backbone_model_name_or_path
            print(f"starting wildicon identity loss encoder loading: {identity_loss_model}")
            self.wildicon_identity_loss = FrozenIdentityFeatureLoss(
                identity_loss_model,
                local_files_only=wildicon_identity_loss_local_files_only,
            ).to(device=device)
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: WildIconFlowMatchSFTLoss(
                pipe,
                identity_loss_fn=self.wildicon_identity_loss,
                identity_loss_weight=self.wildicon_identity_loss_weight,
                identity_loss_num_frames=self.wildicon_identity_loss_num_frames,
                **inputs_shared,
                **inputs_posi,
            ),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: WildIconFlowMatchSFTLoss(
                pipe,
                identity_loss_fn=self.wildicon_identity_loss,
                identity_loss_weight=self.wildicon_identity_loss_weight,
                identity_loss_num_frames=self.wildicon_identity_loss_num_frames,
                **inputs_shared,
                **inputs_posi,
            ),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

    def resume_trainable_checkpoint(self, checkpoint_path, device="cpu"):
        print(f"starting trainable checkpoint loading: {checkpoint_path}")
        state_dict = load_state_dict(checkpoint_path, device=device)
        if self.pipe.wildicon_adapter is not None:
            for prefix in ("pipe.wildicon_adapter.", "wildicon_adapter."):
                if any(key.startswith(prefix) for key in state_dict):
                    state_dict = {
                        key[len(prefix):] if key.startswith(prefix) else key: value
                        for key, value in state_dict.items()
                    }
            load_result = self.pipe.wildicon_adapter.load_state_dict(state_dict, strict=False)
            missing, unexpected = load_result
            print(
                f"loaded trainable checkpoint into wildicon_adapter: keys={len(state_dict)}, missing={len(missing)}, unexpected={len(unexpected)}"
            )
            if len(missing) > 0:
                print(f"wildicon_adapter missing keys: {missing}")
            if len(unexpected) > 0:
                print(f"wildicon_adapter unexpected keys: {unexpected}")
        else:
            load_result = self.load_state_dict(state_dict, strict=False)
            missing, unexpected = load_result
            print(
                f"loaded trainable checkpoint into training module: keys={len(state_dict)}, missing={len(missing)}, unexpected={len(unexpected)}"
            )

    def attach_wildicon_adapter(
        self,
        device,
        selected_block_ids=None,
        segmentor_type="center_prior",
        highpass_sigma=3.0,
        encoder_hidden_dim=128,
        encoder_token_dim=256,
        token_grid_size=(4, 4),
        encoder_type="light_cnn",
        backbone_model_name_or_path="facebook/dinov3-vitl16-pretrain-lvd1689m",
        backbone_train_mode="frozen",
        backbone_trainable_layers=1,
        backbone_local_files_only=True,
    ):
        num_blocks = len(self.pipe.dit.blocks)
        if selected_block_ids is None:
            selected_block_ids = tuple(range((num_blocks * 2) // 3, num_blocks))
        adapter = WildIconIdentityAdapter(
            context_dim=self.pipe.dit.dim,
            selected_block_ids=selected_block_ids,
            segmentor_type=segmentor_type,
            highpass_sigma=highpass_sigma,
            encoder_hidden_dim=encoder_hidden_dim,
            encoder_token_dim=encoder_token_dim,
            token_grid_size=token_grid_size,
            encoder_type=encoder_type,
            backbone_model_name_or_path=backbone_model_name_or_path,
            backbone_train_mode=backbone_train_mode,
            backbone_trainable_layers=backbone_trainable_layers,
            backbone_local_files_only=backbone_local_files_only,
        ).to(device=device)
        self.pipe.wildicon_adapter = adapter
        self.pipe.dit.wildicon_selected_block_ids = adapter.selected_block_ids
        if self.pipe.dit2 is not None:
            self.pipe.dit2.wildicon_selected_block_ids = adapter.selected_block_ids
        
    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image":
                value = data[extra_input]
                value = value[0] if isinstance(value, list) else value
                if self.pipe.wildicon_adapter is not None:
                    inputs_shared["identity_reference_image"] = value
                else:
                    inputs_shared["reference_image"] = value
            elif extra_input in ("vace_reference_image", "identity_reference_image", "segmented_image"):
                value = data[extra_input]
                value = value[0] if isinstance(value, list) else value
                inputs_shared[extra_input] = value
            else:
                inputs_shared[extra_input] = data[extra_input]
        return inputs_shared
    
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to the audio processor. If provided, the processor will be used for Wan2.2-S2V model.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    parser.add_argument("--disable_wan_common_file_redirect", default=False, action="store_true", help="Disable Wan shared-file redirection so local original .pth files are used directly.")
    parser.add_argument("--wildicon_enabled", default=False, action="store_true", help="Enable the WildIcon identity-conditioning branch.")
    parser.add_argument("--wildicon_selected_block_ids", type=str, default=None, help="Comma-separated Wan DiT block ids that receive identity tokens.")
    parser.add_argument("--wildicon_segmentor_type", type=str, default="center_prior", choices=("external", "center_prior", "deeplabv3_resnet50"), help="Segmentor source for the WildIcon identity branch. Use `external` when segmented reference images are provided by preprocessing.")
    parser.add_argument("--wildicon_highpass_sigma", type=float, default=3.0, help="Gaussian sigma used by the WildIcon high-pass residual branch.")
    parser.add_argument("--wildicon_encoder_hidden_dim", type=int, default=128, help="Hidden width of the lightweight WildIcon identity encoder.")
    parser.add_argument("--wildicon_encoder_token_dim", type=int, default=256, help="Token width produced by the WildIcon identity encoder before projection.")
    parser.add_argument("--wildicon_token_grid_size", type=str, default="4,4", help="Token grid size for the WildIcon identity encoder, formatted as 'H,W'.")
    parser.add_argument("--wildicon_encoder_type", type=str, default="light_cnn", choices=("light_cnn", "dinov3"), help="Identity encoder backbone used by the WildIcon branch.")
    parser.add_argument("--wildicon_backbone_model_name_or_path", type=str, default="facebook/dinov3-vitl16-pretrain-lvd1689m", help="Model name or local path for the WildIcon visual backbone when `wildicon_encoder_type` is not `light_cnn`.")
    parser.add_argument("--wildicon_backbone_train_mode", type=str, default="frozen", choices=("frozen", "last_n", "full"), help="How much of the WildIcon visual backbone to fine-tune.")
    parser.add_argument("--wildicon_backbone_trainable_layers", type=int, default=1, help="Number of final backbone blocks to unfreeze when `wildicon_backbone_train_mode=last_n`.")
    parser.add_argument("--wildicon_backbone_local_files_only", default=False, action="store_true", help="Load the WildIcon backbone only from local Hugging Face cache / local paths.")
    parser.add_argument("--wildicon_identity_loss_weight", type=float, default=0.0, help="Weight for the optional frozen feature-space identity loss described in the paper.")
    parser.add_argument("--wildicon_identity_loss_model_name_or_path", type=str, default=None, help="Frozen ReID / image encoder used for WildIcon identity loss. Defaults to --wildicon_backbone_model_name_or_path.")
    parser.add_argument("--wildicon_identity_loss_num_frames", type=int, default=4, help="Number of decoded predicted frames sampled for WildIcon identity loss.")
    parser.add_argument("--wildicon_identity_loss_local_files_only", default=False, action="store_true", help="Load the identity-loss encoder only from local Hugging Face cache / local paths.")
    parser.add_argument("--resume_trainable_checkpoint", type=str, default=None, help="Resume trainable branch weights from a previous checkpoint, e.g. a WildIcon adapter checkpoint from the high-noise stage.")
    return parser


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    # Fail before loading multi-billion-parameter models when local foregrounds are absent.
    if (args.wildicon_enabled and args.wildicon_segmentor_type == "external"
            and args.dataset_metadata_path and args.dataset_metadata_path.endswith(".csv")):
        import csv
        from pathlib import Path
        with open(args.dataset_metadata_path, newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        missing = [row.get("video", "") for row in rows
                   if not row.get("segmented_image", "").strip()
                   or not (Path(args.dataset_base_path) / row["segmented_image"]).is_file()]
        if not rows or missing:
            raise ValueError(
                f"External foreground images are required; {len(missing)} rows have missing images. "
                "Public annotations leave segmented_image empty. Prepare local RGB foregrounds "
                "and run WildIcon/dataset/prepare_local_metadata.py before training."
            )
    log_main_process("starting dataset construction", accelerator)
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
        special_operator_map={
            "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
        }
    )
    log_main_process(
        f"starting model construction: dataset_size={len(dataset)}, repeat={args.dataset_repeat}, extra_inputs={args.extra_inputs}",
        accelerator,
    )
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        redirect_common_files=not args.disable_wan_common_file_redirect,
        wildicon_enabled=args.wildicon_enabled,
        wildicon_selected_block_ids=parse_int_list(args.wildicon_selected_block_ids),
        wildicon_segmentor_type=args.wildicon_segmentor_type,
        wildicon_highpass_sigma=args.wildicon_highpass_sigma,
        wildicon_encoder_hidden_dim=args.wildicon_encoder_hidden_dim,
        wildicon_encoder_token_dim=args.wildicon_encoder_token_dim,
        wildicon_token_grid_size=parse_grid_size(args.wildicon_token_grid_size),
        wildicon_encoder_type=args.wildicon_encoder_type,
        wildicon_backbone_model_name_or_path=args.wildicon_backbone_model_name_or_path,
        wildicon_backbone_train_mode=args.wildicon_backbone_train_mode,
        wildicon_backbone_trainable_layers=args.wildicon_backbone_trainable_layers,
        wildicon_backbone_local_files_only=args.wildicon_backbone_local_files_only,
        wildicon_identity_loss_weight=args.wildicon_identity_loss_weight,
        wildicon_identity_loss_model_name_or_path=args.wildicon_identity_loss_model_name_or_path,
        wildicon_identity_loss_num_frames=args.wildicon_identity_loss_num_frames,
        wildicon_identity_loss_local_files_only=args.wildicon_identity_loss_local_files_only,
        resume_trainable_checkpoint=args.resume_trainable_checkpoint,
    )
    log_main_process("starting model logger setup", accelerator)
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        loss_log_steps=args.loss_log_steps,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    log_main_process(f"starting task launcher: task={args.task}", accelerator)
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
