import torch
from merging_methods.utils import *
from merging_methods.ties_merging_utils import *
from merging_methods.merger import Merger
import time

class TIES(Merger):
    def __init__(self, base_model, ft_models, save_path):
        super().__init__(base_model, ft_models, save_path)
    
    def merge(self, **kwargs):
        scaling_coef = kwargs['scaling_coef']
        task_vectors = [get_task_vector(ft_model, self.base_model) for ft_model in self.ft_ckpts]
        
        start = time.time()

        merged_tv = scaling_coef * ties_merging(torch.stack(task_vectors), reset_thresh=kwargs['K'], merge_func=kwargs['merge_func'])
        merged_model = vector_to_state_dict(merged_tv, self.base_model)
        print("Time taken for ties: ", time.time() - start)

        merged_model.save_pretrained(self.save_path)
        self.tokenizer.save_pretrained(self.save_path)
