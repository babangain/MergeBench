import torch
from merging_methods.utils import *
from merging_methods.merger import Merger
import time

class DARE(Merger):
    def __init__(self, base_model, ft_models, save_path):
        super().__init__(base_model, ft_models, save_path)
    

    def random_drop_and_rescale(self, task_vector, p=0.8):
        if not 0 <= p < 1:
            raise ValueError("p must be in the range [0, 1).")
        
        # Generate a binary mask: 1 with probability (1-p) and 0 with probability p.
        mask = torch.bernoulli(torch.full(task_vector.shape, 1 - p, device=task_vector.device))
        
        # Apply the mask and rescale the kept values by 1/(1-p)
        return task_vector * mask / (1 - p)
    
    def merge(self, **kwargs):
        p = kwargs['p']
        coeff = kwargs['scaling_coef']
        
        task_vectors = [get_task_vector(ft_model, self.base_model) for ft_model in self.ft_ckpts]
        start = time.time()
        task_vectors = [self.random_drop_and_rescale(task_vector, p) for task_vector in task_vectors]
        merged_tv = sum(task_vectors) * coeff
        print("Time taken for random drop and rescale: ", time.time() - start)
        merged_model = vector_to_state_dict(merged_tv, self.base_model)

        merged_model.save_pretrained(self.save_path)
        self.tokenizer.save_pretrained(self.save_path)
