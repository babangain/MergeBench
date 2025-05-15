import os
import gc
import shutil

from tqdm import tqdm

import torch

from merging_methods.utils import *
from merging_methods.merger import Merger

import sys
sys.path.append('<YOUR PATH HERE>/MergeBench/merging')
from .regmean_utils import compute_grams, save_tensor_dict, reduce_non_diag, cleanup_task_loader
from taskloader import *

# https://github.com/bloomberg/dataless-model-merging/blob/main/regmean_demo.ipynb

class RegMean(Merger):
    def __init__(self, base_model, ft_models, save_path):
        super().__init__(base_model, ft_models, save_path)
    
    def merge(self, **kwargs):
        reduction = kwargs["reduction"]

        exam_datasets = kwargs["task_names"].split("-")

        save_dir = self.save_path
        gram_dir = os.path.join(save_dir, "regmean")
        param_dir = os.path.join(save_dir, "params")

        all_param_names = set()
        model_params = self.base_model.state_dict()
        all_param_names.update(model_params.keys())
        
        gram_dirs = [os.path.join(gram_dir, dataset_name) for dataset_name in exam_datasets]
        param_dirs = [os.path.join(param_dir, dataset_name) for dataset_name in exam_datasets]

        for idx, dataset_name in enumerate(exam_datasets):
            finetuned_model = self.ft_ckpts[idx]
            task_loader = TaskLoader(dataset_name, self.base_model, self.tokenizer, sample_size=1000)
            trainer = task_loader.trainer
            dataloader = trainer.get_train_dataloader()
            with torch.no_grad():
                grams = compute_grams(trainer, finetuned_model, dataloader)
            save_tensor_dict(grams, os.path.join(gram_dir, dataset_name)) # contains most (linear) params grams
            save_tensor_dict(finetuned_model.state_dict(), os.path.join(param_dir, dataset_name)) # contains all params

            finetuned_model.to("cpu")
            cleanup_task_loader(task_loader)
            del finetuned_model, grams #, trainer, dataloader, task_loader
            torch.cuda.empty_cache()
            gc.collect()
        self.ft_ckpts = []
        gc.collect()


        with torch.no_grad():
            gram_module_names = {f[:-3] for f in os.listdir(gram_dirs[0]) if f.endswith(".pt")}
            avg_params = {}
            for name in tqdm(all_param_names, desc='Merging'):
                h_avged = False
                if name.endswith('.weight') and not name.startswith('lm_head'):
                    module_name = name[:-len('.weight')]
                    if module_name in gram_module_names:
                        sum_gram, grams = None, None
                        for model_id in range(len(gram_dirs)):
                            param_grams = torch.load(os.path.join(gram_dirs[model_id], module_name + ".pt"), map_location='cpu').detach()
                            param_grams = reduce_non_diag(param_grams, a=reduction) # avoid degeneration
                            param = torch.load(os.path.join(param_dirs[model_id], name + ".pt"), map_location='cpu').detach()
                            gram_m_w = torch.matmul(param_grams, param.transpose(0, 1))
                            if sum_gram is None:
                                sum_gram = param_grams.clone()
                                sum_gram_m_ws = gram_m_w.clone()
                            else:
                                sum_gram.add_(param_grams)
                                sum_gram_m_ws.add_(gram_m_w)
                            del param_grams, param, gram_m_w
                            gc.collect()
                        sum_gram_f32 = sum_gram.to(dtype=torch.float32)
                        cond_number = torch.linalg.cond(sum_gram_f32)
                        threshold = 1e8 
                        if cond_number > threshold or torch.any(torch.diag(sum_gram_f32) == 0):
                            sum_gram_inv = torch.linalg.pinv(sum_gram_f32).to(dtype=sum_gram_m_ws.dtype)
                        else:
                            sum_gram_inv = torch.inverse(sum_gram_f32).to(dtype=sum_gram_m_ws.dtype)
                        wt = torch.matmul(sum_gram_inv, sum_gram_m_ws)
                        avg_params[name] = wt.transpose(0, 1)
                        h_avged = True
                
                if not h_avged: # if not averaged with regmean, then do simple avg
                    filtered_model_params = None
                    for model_id in range(len(gram_dirs)):
                        if not name.startswith('model.embed') and not name.startswith('lm_head'): # embed_tokens.weight have incompatible dimensions due to vocab size difference
                            filtered_model_param = torch.load(os.path.join(param_dirs[model_id], name + ".pt"), map_location='cpu').detach()
                            if filtered_model_params is None:
                                filtered_model_params = filtered_model_param.clone()
                            else:
                                filtered_model_params.add_(filtered_model_param)
                            del filtered_model_param
                            gc.collect()
                            avg_params[name] = filtered_model_params.div(len(gram_dirs))
            
        shutil.rmtree(gram_dir)
        shutil.rmtree(param_dir)

        incompatible_params = self.base_model.load_state_dict(avg_params, strict=False)
        self.base_model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
