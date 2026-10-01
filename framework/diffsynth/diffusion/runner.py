# SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-or-later
#
# Based on DiffSynth-Studio revision ba0626e38f7b8c7908e4f6f597d38282ebba0d38.
# Upstream file: diffsynth/diffusion/runner.py
# Modified for WildIcon by the WildIcon authors.
# Upstream portions retain the Apache License, Version 2.0.
# WildIcon additions and modifications: Copyright (C) 2026 WildIcon authors;
# licensed under GNU LGPL version 3 or (at your option) any later version.
# See WildIcon NOTICE, LICENSE, COPYING, and LICENSES/Apache-2.0.txt;
# installed overlays include these terms under LICENSES/WildIcon/.

import os, torch
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
    accelerator.print(
        f"starting training loop: dataset_size={len(dataset)}, num_epochs={num_epochs}, save_steps={save_steps}, learning_rate={learning_rate}"
    )
    
    optimizer_class = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[getattr(args, "optimizer", "adamw")]
    optimizer = optimizer_class(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    model.to(device=accelerator.device)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    initialize_deepspeed_gradient_checkpointing(accelerator)
    for epoch_id in range(num_epochs):
        accelerator.print(f"starting epoch {epoch_id}")
        progress_bar = tqdm(dataloader, disable=not accelerator.is_local_main_process, desc=f"Epoch {epoch_id}")
        for step_in_epoch, data in enumerate(progress_bar, start=1):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                metrics = model_logger.on_step_end(
                    accelerator,
                    model,
                    save_steps,
                    loss=loss,
                    epoch_id=epoch_id,
                    step_in_epoch=step_in_epoch,
                    learning_rate=optimizer.param_groups[0]["lr"],
                )
                if accelerator.is_local_main_process and len(metrics) > 0:
                    progress_bar.set_postfix(
                        loss=f"{metrics['loss']:.4f}",
                        ema=f"{metrics['loss_ema']:.4f}",
                    )
                scheduler.step()
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model.to(device=accelerator.device)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)


def initialize_deepspeed_gradient_checkpointing(accelerator: Accelerator):
    if getattr(accelerator.state, "deepspeed_plugin", None) is not None:
        ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
        if "activation_checkpointing" in ds_config:
            import deepspeed
            act_config = ds_config["activation_checkpointing"]
            deepspeed.checkpointing.configure(
                mpu_=None, 
                partition_activations=act_config.get("partition_activations", False),
                checkpoint_in_cpu=act_config.get("cpu_checkpointing", False),
                contiguous_checkpointing=act_config.get("contiguous_memory_optimization", False)
            )
        else:
            print("Do not find activation_checkpointing config in deepspeed config, skip initializing deepspeed gradient checkpointing.")
