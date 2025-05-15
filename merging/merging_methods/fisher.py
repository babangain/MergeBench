import os
import gc
import shutil

from tqdm import tqdm

import torch
import torch.distributed as dist

from accelerate.state import AcceleratorState, GradientState

from merging_methods.utils import *
from merging_methods.merger import Merger

import sys
sys.path.append('<YOUR PATH HERE>/MergeBench/merging')
from .fisher_utils import FisherTrainer, save_tensor_dict, cleanup_task_loader, get_expected_fisher_keys, is_tensor_dict_complete
from taskloader import *

from accelerate.utils.deepspeed import DeepSpeedEngineWrapper
from deepspeed.utils import safe_get_full_grad
import os
import gc
import shutil

from tqdm import tqdm

import torch
import torch.distributed as dist

from accelerate.state import AcceleratorState, GradientState

from merging_methods.utils import *
from merging_methods.merger import Merger

import sys
sys.path.append('/home/cindy2000_sh/MergeBench/merging')
from .fisher_utils import FisherTrainer, save_tensor_dict, cleanup_task_loader, get_expected_fisher_keys, is_tensor_dict_complete
from taskloader import *

from accelerate.utils.deepspeed import DeepSpeedEngineWrapper
from deepspeed.utils import safe_get_full_grad

# https://github.com/mmatena/model_merging/blob/master/model_merging/fisher.py

class Fisher(Merger):
    def __init__(self, base_model, ft_models, save_path):
        super().__init__(base_model, ft_models, save_path)
        self.base_model = self.base_model.to('cpu')
        self.ft_ckpts = [ft_model.to('cpu') for ft_model in self.ft_ckpts]
    
    def merge(self, **kwargs):
        fisher_only = kwargs["fisher_only"]
        merge_only = kwargs["merge_only"]
        model_coeff_value = kwargs["model_coeff_value"]
        keep_checkpoints = kwargs['keep_checkpoints']
        save_group = kwargs['save_group']

        exam_datasets = kwargs["task_names"].split("-")
        all_tasks = save_group.split("-")
        self.ft_ckpts = {
            task: model.to('cpu')
            for task, model in zip(all_tasks, self.ft_ckpts)
        }

        save_dir = self.save_path
        fisher_dir = os.path.join(save_dir, "fisher")
        param_dir = os.path.join(save_dir, "params")

        all_param_names = set()
        model_params = self.base_model.state_dict()
        all_param_names.update(model_params.keys())

        fisher_dirs = [os.path.join(fisher_dir, dataset_name) for dataset_name in exam_datasets]
        param_dirs = [os.path.join(param_dir, dataset_name) for dataset_name in exam_datasets]

        if fisher_only:
            for idx, dataset_name in enumerate(exam_datasets):
                
                fisher = {}
                n_steps_ref = [0] # Mutable n_steps (so it updates across steps)

                def make_patched_backward(fisher_dict, n_steps_ref):
                    def patched_backward(self, loss, **kwargs):
                        self.engine.backward(loss, **kwargs)
                        with torch.no_grad():
                            for name, param in self.engine.module.named_parameters():
                                if not param.requires_grad:
                                    continue
                                grad_ds = safe_get_full_grad(param)
                                if grad_ds is not None:
                                    grad_cpu = grad_ds.detach().cpu()
                                    grad_sq_cpu = grad_cpu ** 2
                                    if name not in fisher_dict:
                                        fisher_dict[name] = grad_sq_cpu
                                    else:
                                        fisher_dict[name] += grad_sq_cpu
                                    del grad_ds, grad_cpu, grad_sq_cpu
                            torch.cuda.empty_cache()
                            n_steps_ref[0] += 1
                        self.engine.step()
                    return patched_backward

                
                DeepSpeedEngineWrapper.backward = make_patched_backward(fisher, n_steps_ref)
                
                finetuned_model = self.ft_ckpts[dataset_name]
                expected_fisher_keys = get_expected_fisher_keys(finetuned_model)
                param_path = param_dirs[idx]
                fisher_path = fisher_dirs[idx]

                fisher_complete = is_tensor_dict_complete(fisher_path, expected_fisher_keys)
                params_complete = is_tensor_dict_complete(param_path, all_param_names)

                print(exam_datasets,"fisher_complete:",fisher_complete,"params_complete:",params_complete)

                if fisher_complete and params_complete:
                    print(f"Skipping {dataset_name} — already processed.")
                    self.ft_ckpts[dataset_name] = finetuned_model.to("cpu")
                    continue

                elif fisher_complete and not params_complete:
                    print(f"Processing {dataset_name}")
                    if dist.get_rank() == 0:
                        save_tensor_dict(finetuned_model.state_dict(), os.path.join(param_dir, dataset_name))
                    continue

                print(f"Processing {dataset_name}")
                if not params_complete:
                    if dist.get_rank() == 0:
                        save_tensor_dict(finetuned_model.state_dict(), os.path.join(param_dir, dataset_name))
                        
                finetuned_model = self.ft_ckpts[dataset_name].to("cuda")

                task_loader = TaskLoader(dataset_name, finetuned_model, self.tokenizer, sample_size=1000)
                sft_trainer = task_loader.trainer

                sft_args = sft_trainer.args
                sft_model = sft_trainer.model
                sft_train_dataset = sft_trainer.train_dataset
                sft_formatting_func = getattr(sft_trainer, "formatting_func", None)

                sft_model.gradient_checkpointing_enable()

                AcceleratorState._reset_state(True)
                GradientState._reset_state()

                fisher_trainer = FisherTrainer(
                                model=sft_model,
                                args=sft_args,
                                train_dataset=sft_train_dataset,
                                formatting_func=sft_formatting_func,
                            )

                fisher_trainer.train()
                
                for k in fisher:
                    fisher[k] /= n_steps_ref[0]  
                
                if dist.get_rank() == 0:       
                    save_tensor_dict(fisher, os.path.join(fisher_dir, dataset_name))

                self.ft_ckpts[dataset_name] = self.ft_ckpts[dataset_name].to("cpu")
                cleanup_task_loader(task_loader)
                del fisher_trainer
                gc.collect()
                torch.cuda.empty_cache()

            self.ft_ckpts = []
            torch.cuda.empty_cache()
            gc.collect()

        if merge_only:
            if not dist.is_initialized() or dist.get_rank() == 0:
                # https://github.com/uiuctml/MergeBench/blob/main/merging/clip_merging_code/src/main_fisher.py
                model_coeffs = torch.ones(len(exam_datasets)) * model_coeff_value
                avg_params = {}
                fisher_module_names = {f[:-3] for f in os.listdir(fisher_dirs[0]) if f.endswith(".pt")}

                for n in tqdm(all_param_names, desc='Merging'):
                    if n in fisher_module_names and not n.startswith('model.embed'):
                        param_list = []
                        fisher_list = []

                        fisher_list = [torch.load(os.path.join(fisher_dirs[model_id], n + ".pt"), map_location="cpu") for model_id in range(len(fisher_dirs))]
                        param_list = [torch.load(os.path.join(param_dirs[model_id], n + ".pt"), map_location="cpu") for model_id in range(len(fisher_dirs))]

                        params = torch.stack(param_list)  # [N, *]
                        fisher = torch.stack(fisher_list) + 1.0e-10  # [N, *]

                        coeff = model_coeffs.view(-1, *[1 for _ in range(params.dim() - 1)]).to(params.device)
                        fisher = fisher.to(params.device)
                        sum_p = (params * fisher * coeff).sum(0)
                        denom = (fisher * coeff).sum(0)
                        avg_p = sum_p / denom

                        avg_params[n] = avg_p.cpu()  

                        del param_list, fisher_list, params, fisher, sum_p, denom, avg_p
                        torch.cuda.empty_cache()
                
                # remove intermediate checkpoints
                if not keep_checkpoints:
                    shutil.rmtree(fisher_dir)
                    shutil.rmtree(param_dir)

                incompatible_params = self.base_model.load_state_dict(avg_params, strict=False)
                self.base_model.save_pretrained(save_dir)
                self.tokenizer.save_pretrained(save_dir)