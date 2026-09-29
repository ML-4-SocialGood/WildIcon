import os, torch
from accelerate import Accelerator


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x, loss_log_steps: int = 1):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0
        self.loss_ema = None
        self.loss_ema_beta = 0.98
        self.loss_log_path = os.path.join(output_path, "loss_log.csv")
        self.loss_log_initialized = False
        self.loss_log_steps = max(1, int(loss_log_steps))


    def initialize_loss_log(self):
        if self.loss_log_initialized:
            return
        os.makedirs(self.output_path, exist_ok=True)
        if not os.path.exists(self.loss_log_path):
            with open(self.loss_log_path, "w", encoding="utf-8") as file:
                file.write("step,epoch,step_in_epoch,loss,loss_ema,learning_rate\n")
        self.loss_log_initialized = True


    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None, **kwargs):
        self.num_steps += 1
        metrics = {}
        loss = kwargs.get("loss")
        if loss is not None:
            loss_value = accelerator.reduce(loss.detach().float(), reduction="mean").item()
            if self.loss_ema is None:
                self.loss_ema = loss_value
            else:
                self.loss_ema = self.loss_ema_beta * self.loss_ema + (1.0 - self.loss_ema_beta) * loss_value
            metrics["loss"] = loss_value
            metrics["loss_ema"] = self.loss_ema
            should_write_loss = self.num_steps == 1 or self.num_steps % self.loss_log_steps == 0
            if accelerator.is_main_process and should_write_loss:
                self.initialize_loss_log()
                epoch_id = kwargs.get("epoch_id", 0)
                step_in_epoch = kwargs.get("step_in_epoch", self.num_steps)
                learning_rate = kwargs.get("learning_rate", "")
                with open(self.loss_log_path, "a", encoding="utf-8") as file:
                    file.write(f"{self.num_steps},{epoch_id},{step_in_epoch},{loss_value:.8f},{self.loss_ema:.8f},{learning_rate}\n")
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")
        return metrics


    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)


    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)
