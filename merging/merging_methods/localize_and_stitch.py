from merging_methods.utils import *
from merging_methods.merger import Merger
from merging_methods.localize_utils import *
from transformers import AutoModelForCausalLM
from datasets import load_dataset


class LocalizeAndStitch(Merger):
    def __init__(self, base_model, ft_models, save_path):
        super().__init__(base_model, ft_models, save_path)

        self.task_names = ['safety','instruction', 'math', 'coding', 'multilingual']
    
    def extract_format_keys(self, task):
        dataset = load_dataset(f'MergeBench/{task}_val', split='train')

        if task == 'safety':
            format_keys = {"instruction_key": "prompt", "output_key": "response"}
        elif task == 'multilingual':
            format_keys = {"instruction_key": "inputs", "output_key": "targets"}
        elif task == 'math': 
            format_keys = {"instruction_key": "query", "output_key": "response"}
        elif task == 'instruction':
            format_keys = {"instruction_key": "instruction", "output_key": "output"}
        elif task == 'coding': 
            format_keys = {"output_key": "response"}
        
        return dataset, format_keys
        
    def merge(self, **kwargs):
        graft_args = {}
        dataless = kwargs['dataless']
        graft_args['sparsity'] = kwargs['sparsity']
        graft_args['sigmoid_bias'] = kwargs['sigmoid_bias']
        if not dataless:
            graft_args['lr'] = kwargs['learning_rate']
            graft_args['num_train_epochs'] = kwargs['num_train_epochs']
            graft_args['l1_strength'] = kwargs['l1_strength']

        # Localize
        masks = []
        for i in range(len(self.ft_ckpts)):
            current_task = self.task_names[i]
            print(f'Localizing {current_task} model')
            ft_model = self.ft_ckpts[i]
            trainable_params = select_trainable_params(ft_model)

            localizer = Localizer(trainable_params, self.base_model, ft_model, graft_args, self.base_model_name)
            
            if not dataless:
                print(f'Training mask {current_task} model')
                dataset, format_keys = self.extract_format_keys(self.task_names[i])
                if current_task == 'safety':
                    dataset = dataset.filter(lambda ex: ex.get("response") is not None)
                localizer.train_mask(dataset, format_keys) 
            
            mask, _ = localizer.interpolate_model(round_=True, return_mask=True, train=False)
            masks.append(mask)
        
        # Stitch
        final_model = AutoModelForCausalLM.from_pretrained(self.base_model_name)
        stitcher = Stitcher(trainable_params, final_model, self.base_model, self.ft_ckpts, masks)
        merged_model = stitcher.interpolate_models()

        merged_model.save_pretrained(self.save_path)
        self.tokenizer.save_pretrained(self.save_path)
