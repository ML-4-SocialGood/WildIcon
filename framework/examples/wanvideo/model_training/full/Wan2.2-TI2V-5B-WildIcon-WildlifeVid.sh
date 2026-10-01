#!/usr/bin/env bash
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 WildIcon authors.
#
# This file is part of WildIcon and is licensed under the GNU Lesser
# General Public License, version 3 or (at your option) any later version.
# Distributed without any warranty; see WildIcon LICENSE and COPYING.
# Installed overlays include these terms under LICENSES/WildIcon/.

set -euo pipefail

DATASET_BASE_PATH="${DATASET_BASE_PATH:-/path/to/WildlifeVid}"
DATASET_METADATA_PATH="${DATASET_METADATA_PATH:-${DATASET_BASE_PATH}/WildlifeVid.csv}"
LOCAL_MODEL_ROOT="${LOCAL_MODEL_ROOT:-/path/to/Wan2.2-TI2V-5B}"
LOCAL_MODEL_BASE_PATH="${LOCAL_MODEL_BASE_PATH:-/tmp/diffsynth_models}"
OUTPUT_PATH="${OUTPUT_PATH:-./models/train/Wan2.2-TI2V-5B_WildIcon_WildlifeVid}"
WILDICON_DINO_MODEL="${WILDICON_DINO_MODEL:-facebook/dinov3-vitl16-pretrain-lvd1689m}"
WILDICON_BACKBONE_TRAIN_MODE="${WILDICON_BACKBONE_TRAIN_MODE:-frozen}"
WILDICON_BACKBONE_TRAINABLE_LAYERS="${WILDICON_BACKBONE_TRAINABLE_LAYERS:-1}"
WILDICON_IDENTITY_LOSS_WEIGHT="${WILDICON_IDENTITY_LOSS_WEIGHT:-0.05}"
WILDICON_IDENTITY_LOSS_MODEL="${WILDICON_IDENTITY_LOSS_MODEL:-${WILDICON_DINO_MODEL}}"
WILDICON_IDENTITY_LOSS_NUM_FRAMES="${WILDICON_IDENTITY_LOSS_NUM_FRAMES:-4}"
SAVE_STEPS="${SAVE_STEPS:-500}"
LOSS_LOG_STEPS="${LOSS_LOG_STEPS:-1}"

mkdir -p "${LOCAL_MODEL_BASE_PATH}/Wan-AI"
ln -sfn "${LOCAL_MODEL_ROOT}" "${LOCAL_MODEL_BASE_PATH}/Wan-AI/Wan2.2-TI2V-5B"

export DIFFSYNTH_MODEL_BASE_PATH="${LOCAL_MODEL_BASE_PATH}"
export DIFFSYNTH_SKIP_DOWNLOAD=true

accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${DATASET_METADATA_PATH}" \
  --data_file_keys "video,reference_image,segmented_image" \
  --height 480 \
  --width 832 \
  --num_frames 81 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
  --tokenizer_path "${LOCAL_MODEL_ROOT}/google/umt5-xxl" \
  --optimizer adam \
  --weight_decay 0 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --num_epochs 2 \
  --save_steps "${SAVE_STEPS}" \
  --loss_log_steps "${LOSS_LOG_STEPS}" \
  --output_path "${OUTPUT_PATH}" \
  --remove_prefix_in_ckpt "pipe.wildicon_adapter." \
  --trainable_models "wildicon_adapter" \
  --extra_inputs "input_image,reference_image,segmented_image" \
  --wildicon_enabled \
  --wildicon_encoder_type "dinov3" \
  --wildicon_backbone_model_name_or_path "${WILDICON_DINO_MODEL}" \
  --wildicon_backbone_train_mode "${WILDICON_BACKBONE_TRAIN_MODE}" \
  --wildicon_backbone_trainable_layers "${WILDICON_BACKBONE_TRAINABLE_LAYERS}" \
  --wildicon_backbone_local_files_only \
  --wildicon_segmentor_type "external" \
  --wildicon_selected_block_ids "20,21,22,23,24,25,26,27,28,29" \
  --wildicon_identity_loss_weight "${WILDICON_IDENTITY_LOSS_WEIGHT}" \
  --wildicon_identity_loss_model_name_or_path "${WILDICON_IDENTITY_LOSS_MODEL}" \
  --wildicon_identity_loss_num_frames "${WILDICON_IDENTITY_LOSS_NUM_FRAMES}" \
  --wildicon_identity_loss_local_files_only \
  --disable_wan_common_file_redirect
