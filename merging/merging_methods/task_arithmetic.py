from merging_methods.utils import *
from merging_methods.merger import Merger

class TaskArithmetic(Merger):
    def __init__(self, base_model, ft_models, save_path):
        super().__init__(base_model, ft_models, save_path)
    
    def merge(self, **kwargs):
        scaling_coef = kwargs['scaling_coef']
        task_vectors = [get_task_vector(ft_model, self.base_model) for ft_model in self.ft_ckpts]
        merged_tv = scaling_coef * sum(task_vectors)
        merged_model = vector_to_state_dict(merged_tv, self.base_model)

        merged_model.save_pretrained(self.save_path)
        self.tokenizer.save_pretrained(self.save_path)
