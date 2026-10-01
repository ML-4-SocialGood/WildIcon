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
LOCAL_MODEL_ROOT="${LOCAL_MODEL_ROOT:-/path/to/Wan2.2-I2V-A14B}"
LOCAL_MODEL_BASE_PATH="${LOCAL_MODEL_BASE_PATH:-/tmp/diffsynth_models}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-examples/wanvideo/model_training/full/accelerate_config.yaml}"
TRAIN_STAGE="${TRAIN_STAGE:-high_noise}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./models/train/Wan2.2-I2V-A14B_WildIcon_WildlifeVid}"
WILDICON_DINO_MODEL="${WILDICON_DINO_MODEL:-facebook/dinov3-vitl16-pretrain-lvd1689m}"
WILDICON_BACKBONE_TRAIN_MODE="${WILDICON_BACKBONE_TRAIN_MODE:-frozen}"
WILDICON_BACKBONE_TRAINABLE_LAYERS="${WILDICON_BACKBONE_TRAINABLE_LAYERS:-1}"
WILDICON_SELECTED_BLOCK_IDS="${WILDICON_SELECTED_BLOCK_IDS:-26,27,28,29,30,31,32,33,34,35,36,37,38,39}"
WILDICON_IDENTITY_LOSS_WEIGHT="${WILDICON_IDENTITY_LOSS_WEIGHT:-0.05}"
WILDICON_IDENTITY_LOSS_MODEL="${WILDICON_IDENTITY_LOSS_MODEL:-${WILDICON_DINO_MODEL}}"
WILDICON_IDENTITY_LOSS_NUM_FRAMES="${WILDICON_IDENTITY_LOSS_NUM_FRAMES:-4}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
NUM_FRAMES="${NUM_FRAMES:-81}"
DATASET_REPEAT="${DATASET_REPEAT:-1}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
SAVE_STEPS="${SAVE_STEPS:-250}"
LOSS_LOG_STEPS="${LOSS_LOG_STEPS:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
RESUME_TRAINABLE_CHECKPOINT="${RESUME_TRAINABLE_CHECKPOINT:-}"

WAN_MODEL_ID="Wan-AI/Wan2.2-I2V-A14B"

if [ ! -f "${DATASET_METADATA_PATH}" ]; then
  echo "Missing dataset metadata: ${DATASET_METADATA_PATH}"
  exit 1
fi

if [ ! -d "${LOCAL_MODEL_ROOT}" ]; then
  echo "Missing local A14B model root: ${LOCAL_MODEL_ROOT}"
  exit 1
fi

mkdir -p "${LOCAL_MODEL_BASE_PATH}/Wan-AI"
ln -sfn "${LOCAL_MODEL_ROOT}" "${LOCAL_MODEL_BASE_PATH}/${WAN_MODEL_ID}"

export DIFFSYNTH_MODEL_BASE_PATH="${LOCAL_MODEL_BASE_PATH}"
export DIFFSYNTH_SKIP_DOWNLOAD=true

find_latest_checkpoint() {
  local checkpoint_dir="$1"
  local latest_checkpoint=""
  latest_checkpoint=$(find "${checkpoint_dir}" -maxdepth 1 -type f -name 'step-*.safetensors' | sort -V | tail -n 1 || true)
  if [ -z "${latest_checkpoint}" ]; then
    latest_checkpoint=$(find "${checkpoint_dir}" -maxdepth 1 -type f -name 'epoch-*.safetensors' | sort -V | tail -n 1 || true)
  fi
  echo "${latest_checkpoint}"
}

run_stage() {
  local stage="$1"
  local resume_checkpoint="${2:-}"
  local dit_path=""
  local min_timestep_boundary=""
  local max_timestep_boundary=""
  local output_path="${OUTPUT_ROOT}_${stage}"

  case "${stage}" in
    high_noise)
      dit_path="high_noise_model/diffusion_pytorch_model*.safetensors"
      min_timestep_boundary="0"
      max_timestep_boundary="0.358"
      ;;
    low_noise)
      dit_path="low_noise_model/diffusion_pytorch_model*.safetensors"
      min_timestep_boundary="0.358"
      max_timestep_boundary="1"
      ;;
    *)
      echo "Unsupported stage=${stage}. Use high_noise or low_noise."
      exit 1
      ;;
  esac

  RESUME_ARGS=()
  if [ -n "${resume_checkpoint}" ]; then
    RESUME_ARGS+=(--resume_trainable_checkpoint "${resume_checkpoint}")
  fi

  echo "starting Wan2.2-I2V-A14B WildIcon training"
  echo "starting stage=${stage}"
  echo "starting local_model_root=${LOCAL_MODEL_ROOT}"
  echo "starting dataset_metadata_path=${DATASET_METADATA_PATH}"
  echo "starting output_path=${output_path}"
  if [ -n "${resume_checkpoint}" ]; then
    echo "starting resume_trainable_checkpoint=${resume_checkpoint}"
  fi

  accelerate launch --config_file "${ACCELERATE_CONFIG}" examples/wanvideo/model_training/train.py \
    --dataset_base_path "${DATASET_BASE_PATH}" \
    --dataset_metadata_path "${DATASET_METADATA_PATH}" \
    --data_file_keys "video,reference_image,segmented_image" \
    --height "${HEIGHT}" \
    --width "${WIDTH}" \
    --num_frames "${NUM_FRAMES}" \
    --dataset_repeat "${DATASET_REPEAT}" \
    --model_id_with_origin_paths "${WAN_MODEL_ID}:${dit_path},${WAN_MODEL_ID}:models_t5_umt5-xxl-enc-bf16.pth,${WAN_MODEL_ID}:Wan2.1_VAE.pth" \
    --tokenizer_path "${LOCAL_MODEL_ROOT}/google/umt5-xxl" \
    --optimizer adam \
    --weight_decay 0 \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --learning_rate "${LEARNING_RATE}" \
    --num_epochs "${NUM_EPOCHS}" \
    --save_steps "${SAVE_STEPS}" \
    --loss_log_steps "${LOSS_LOG_STEPS}" \
    --output_path "${output_path}" \
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
    --wildicon_selected_block_ids "${WILDICON_SELECTED_BLOCK_IDS}" \
    --wildicon_identity_loss_weight "${WILDICON_IDENTITY_LOSS_WEIGHT}" \
    --wildicon_identity_loss_model_name_or_path "${WILDICON_IDENTITY_LOSS_MODEL}" \
    --wildicon_identity_loss_num_frames "${WILDICON_IDENTITY_LOSS_NUM_FRAMES}" \
    --wildicon_identity_loss_local_files_only \
    --use_gradient_checkpointing_offload \
    --min_timestep_boundary "${min_timestep_boundary}" \
    --max_timestep_boundary "${max_timestep_boundary}" \
    --disable_wan_common_file_redirect \
    "${RESUME_ARGS[@]}"
}

case "${TRAIN_STAGE}" in
  high_noise)
    run_stage "high_noise" "${RESUME_TRAINABLE_CHECKPOINT}"
    ;;
  low_noise)
    run_stage "low_noise" "${RESUME_TRAINABLE_CHECKPOINT}"
    ;;
  both)
    run_stage "high_noise" "${RESUME_TRAINABLE_CHECKPOINT}"
    HIGH_STAGE_OUTPUT="${OUTPUT_ROOT}_high_noise"
    LATEST_HIGH_CHECKPOINT="$(find_latest_checkpoint "${HIGH_STAGE_OUTPUT}")"
    if [ -z "${LATEST_HIGH_CHECKPOINT}" ]; then
      echo "Failed to find a high-noise checkpoint in ${HIGH_STAGE_OUTPUT}"
      exit 1
    fi
    run_stage "low_noise" "${LATEST_HIGH_CHECKPOINT}"
    ;;
  *)
    echo "Unsupported TRAIN_STAGE=${TRAIN_STAGE}. Use high_noise, low_noise, or both."
    exit 1
    ;;
esac
