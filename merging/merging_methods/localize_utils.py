import torch
import torch.nn as nn
from tqdm import tqdm
from merging_methods.utils import get_task_vector, vector_to_state_dict
from taskloader import formatting_prompts_func
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, TrainerCallback
from accelerate import dispatch_model
from accelerate import infer_auto_device_map

class Localizer():
    def __init__(self, trainable_params, pretrained_model, finetuned_model, graft_args, base_model_name):
        super().__init__()
        
        self.params = trainable_params
        self.pretrained_model = pretrained_model
        self.finetuned_model = finetuned_model
        self.graft_args = graft_args
        self.base_model_name = base_model_name

        self.pretrained_model.to("cpu")
        self.finetuned_model.to("cpu")
        self.finetuned_model.eval()
        self.pretrained_model.eval()
        for param in self.pretrained_model.parameters():
            param.requires_grad = False   
        for param in self.finetuned_model.parameters():
            param.requires_grad = False

        self.task_vector = get_task_vector(self.finetuned_model, self.pretrained_model)
        self.num_params = len(self.task_vector)

        # self.create_binary_masks()
        self.mask = self.create_topk_mask()


    def reset_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.base_model_name, 
                                                    torch_dtype="bfloat16", 
                                                    device_map='auto')
        #self.device_map = self.model.hf_device_map
        self.device_map = infer_auto_device_map(self.model, max_memory={0: "60GiB"})
        self.model = self.model.to("cuda")  
        self.model = dispatch_model(self.model, device_map=self.device_map)


    def create_topk_mask(self):

        abs_tv = torch.abs(self.task_vector)
        k = int(self.graft_args['sparsity'] * abs_tv.numel())  # 1% of the total number of elements

        # Get the k largest values; returns values and their indices
        values, indices = torch.topk(abs_tv.view(-1), k)
        threshold = values.min()

        mask = torch.zeros_like(self.task_vector, requires_grad=False)
        mask[torch.abs(self.task_vector) >= threshold] = self.graft_args['sigmoid_bias']
        # print non-zero count in mask
        print('Initial topk sparsity in my mask: ', torch.nonzero(mask).numel() / self.num_params)

        mask[torch.abs(self.task_vector) < threshold] = -self.graft_args['sigmoid_bias']
        # mask[torch.abs(self.task_vector) > threshold] = 1

        return mask


    def interpolate_model(self, round_, return_mask=False, train=True):  

        sigmoid = torch.nn.Sigmoid()
        frac = sigmoid(self.mask)
        
        if round_:
            frac = torch.round(frac)
        
        final_tv = self.task_vector.clone()
        final_tv = final_tv * frac 
        self.model = vector_to_state_dict(final_tv, self.pretrained_model, return_dict=False)
        self.pretrained_model = AutoModelForCausalLM.from_pretrained(self.base_model_name,  
                                                    torch_dtype="bfloat16")
        if train:
            self.model = self.model.to("cuda")
            self.model = dispatch_model(self.model, device_map=self.device_map)

        if round_:
            proportion = len(torch.nonzero(frac.bool())) / self.num_params
            print('Proportion in my mask: ', proportion)
        
        if return_mask:
            return frac, proportion


    def train_mask(self, dataset, format_keys):
        
        sigmoid = torch.nn.Sigmoid()
        
        # Create the interpolated model with the current mask
        self.reset_model()

        for i in range(self.graft_args['num_train_epochs']):
            print(f"Training epoch {i+1}")

            self.interpolate_model(round_=False)
            self.model.train()
            for param in self.model.parameters():
                param.requires_grad = True
            
            training_args = SFTConfig(
                            per_device_train_batch_size=2,  # Minimum batch size
                            packing=False,
                            gradient_checkpointing=True,
                            save_strategy="no",
                            optim="adamw_torch_fused",
                            bf16=True,
                            report_to=None,
                            do_eval=False,
                            num_train_epochs=self.graft_args['num_train_epochs'],
                            output_dir="output",
                            max_seq_length=3072,
                        )
            
            # Create SFTTrainer
            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset,
                formatting_func=lambda examples: formatting_prompts_func(
                                    examples, **format_keys
                                ),
            )
            
            # Define a callback to track gradients during training
            class GradientTrackingCallback(TrainerCallback):
                def __init__(self):
                    self.accumulated_grads = {}
                    self.num_backward_calls = 0
                
                def on_optimizer_step(self, args, state, control, model, **kwargs):
                    self.num_backward_calls += 1
                    for name, param in model.named_parameters():
                        # if 'embed' not in name.lower() and 'lm_head' not in name.lower():
                        if 'embed' not in name.lower():
                            if name not in self.accumulated_grads:
                                self.accumulated_grads[name] = param.grad.to('cpu').detach().clone()
                            else:
                                self.accumulated_grads[name] += param.grad.to('cpu').detach().clone()
                    return control
                
                def get_total_grads(self):
                    grad_vector = torch.cat([grad.flatten() for k, grad in self.accumulated_grads.items()])
                    return grad_vector

            # Convert accumulated gradients dict to a single tensor
            gradient_callback = GradientTrackingCallback()
            trainer.add_callback(gradient_callback)
            
            # Train for one epoch
            trainer.train()

            # gradient of the loss with respect to the model
            grad = gradient_callback.get_total_grads()
            grad = grad * self.task_vector

            # Reset model for next epoch
            self.reset_model()
            
            # Take the gradient step to update the mask
            with torch.no_grad():
                # gradient of the model with respect to the mask
                derivative = sigmoid(self.mask) * (1 - sigmoid(self.mask))
                reg_term = self.graft_args['l1_strength'] * torch.where(self.mask > 0, derivative, -derivative)
                grad.to(self.mask.device)
                # print("total_grad: ", (total_grad * derivative).mean())
                print(self.graft_args['lr'] * grad * derivative - reg_term)
                self.mask -= self.graft_args['lr'] * grad * derivative - reg_term
                print("Gradient step on mask complete")

                cur_mask = self.mask.clone()
                cur_mask = torch.round(sigmoid(cur_mask))
                print('Proportion in my mask: ', len(torch.nonzero(cur_mask.bool())) / self.num_params)


class Stitcher(nn.Module):
    def __init__(self, trainable_params, model, pretrained_model, finetuned_models, masks):
        super().__init__()
        self.params = trainable_params
        self.pretrained_model = pretrained_model
        self.finetuned_models = finetuned_models
        self.model = model

        self.masks = masks
        if len(self.masks) > 1:
            self.masks = self.get_average_masks()
        self.task_vector = torch.zeros_like(get_task_vector(self.finetuned_models[0], self.pretrained_model))


    def get_average_masks(self):
            
        def reciprocal_with_zero(tensor):
            mask = tensor == 0
            reciprocal = torch.reciprocal(tensor)
            reciprocal = reciprocal.masked_fill(mask, 0)
            return reciprocal

        output_masks = []
        for i in range(len(self.masks)):
            output_mask = self.masks[i].clone().detach()
            for j in range(len(self.masks)):
                if i == j: continue
                intersect = torch.logical_and(self.masks[i], self.masks[j])
            output_mask = output_mask + intersect
            output_mask = reciprocal_with_zero(output_mask)
            output_masks.append(output_mask)

        return output_masks


    def interpolate_models(self):

        for finetuned_model, mask in zip(self.finetuned_models, self.masks):
            with torch.no_grad():
                self.task_vector += mask * get_task_vector(finetuned_model, self.pretrained_model)
    
        self.model = vector_to_state_dict(self.task_vector, self.pretrained_model, return_dict=False)
        
        return self.model
                        
# import torch
# import torch.nn as nn
# from tqdm import tqdm
# from merging_methods.utils import get_task_vector, vector_to_state_dict
# from taskloader import formatting_prompts_func
# from trl import SFTTrainer, SFTConfig
# from transformers import AutoModelForCausalLM, TrainerCallback

# class Localizer():
#     def __init__(self, trainable_params, pretrained_model, finetuned_model, graft_args, base_model_name):
#         super().__init__()
        
#         self.params = trainable_params
#         self.pretrained_model = pretrained_model
#         self.finetuned_model = finetuned_model
#         self.graft_args = graft_args
#         self.base_model_name = base_model_name

#         # Keep models on CPU by default
#         self.pretrained_model.to("cpu")
#         self.finetuned_model.to("cpu")
#         self.finetuned_model.eval()
#         self.pretrained_model.eval()
        
#         for param in self.pretrained_model.parameters():
#             param.requires_grad = False   
#         for param in self.finetuned_model.parameters():
#             param.requires_grad = False

#         self.task_vector = get_task_vector(self.finetuned_model, self.pretrained_model)
#         self.num_params = len(self.task_vector)

#         # Keep task vector and mask on CPU
#         self.task_vector = self.task_vector.cpu()
#         self.mask = self.create_topk_mask()
        
#         # Store device for GPU operations
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     def reset_model(self):
#         # Load model on CPU first to save GPU memory
#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.base_model_name, 
#             torch_dtype=torch.bfloat16, # attn_implementation="flash_attention_2",
#             device_map="cpu"  # Explicitly load on CPU
#         )

#     def create_topk_mask(self):
#         abs_tv = torch.abs(self.task_vector)
#         k = int(self.graft_args['sparsity'] * abs_tv.numel())

#         values, indices = torch.topk(abs_tv.view(-1), k)
#         threshold = values.min()

#         mask = torch.zeros_like(self.task_vector, requires_grad=False, device="cpu")
#         mask[torch.abs(self.task_vector) >= threshold] = self.graft_args['sigmoid_bias']
#         print('Initial topk sparsity in my mask: ', torch.nonzero(mask).numel() / self.num_params)

#         mask[torch.abs(self.task_vector) < threshold] = -self.graft_args['sigmoid_bias']
#         return mask

#     def interpolate_model(self, round_, return_mask=False, train=True):  
#         sigmoid = torch.nn.Sigmoid()
#         frac = sigmoid(self.mask)
        
#         if round_:
#             frac = torch.round(frac)
        
#         final_tv = self.task_vector.clone()
#         final_tv = final_tv * frac 
        
#         # Create model state dict on CPU
#         self.model = vector_to_state_dict(final_tv, self.pretrained_model, return_dict=False)
        
#         # Reload pretrained model on CPU
#         self.pretrained_model = AutoModelForCausalLM.from_pretrained(
#             self.base_model_name, 
#             torch_dtype=torch.bfloat16,# attn_implementation="flash_attention_2",
#             device_map="cpu"
#         )
        
#         # Only move to GPU when training is needed
#         if train:
#             print("Moving model to GPU for training...")
#             self.model = self.model.to(self.device)

#         if round_:
#             proportion = len(torch.nonzero(frac.bool())) / self.num_params
#             print('Proportion in my mask: ', proportion)
        
#         if return_mask:
#             return frac, proportion

#     def train_mask(self, dataset, format_keys):
#         sigmoid = torch.nn.Sigmoid()
        
#         for i in range(self.graft_args['num_train_epochs']):
#             print(f"Training epoch {i+1}")

#             # Reset and prepare model
#             self.reset_model()
#             self.interpolate_model(round_=False, train=True)  # This moves to GPU
            
#             self.model.train()
#             for param in self.model.parameters():
#                 param.requires_grad = True
            
#             training_args = SFTConfig(
#                             per_device_train_batch_size=2,
#                             packing=False,
#                             gradient_checkpointing=True,
#                             save_strategy="no",
#                             optim="adamw_torch_fused",
#                             bf16=True,
#                             report_to=None,
#                             do_eval=False,
#                             num_train_epochs=1,  # Train for 1 epoch per iteration
#                             output_dir="output",
#                             max_seq_length=3072,
#                         )

#             formatted_dataset = dataset.map(
#                     lambda examples: formatting_prompts_func(examples, **format_keys)
#             )

        
#             # 2. Define SFTTrainer without formatting_func
#             trainer = SFTTrainer(
#                 model=self.model,
#                 args=training_args,
#                 train_dataset=formatted_dataset,
#                 )
#             # trainer = SFTTrainer(
#             #     model=self.model,
#             #     args=training_args,
#             #     train_dataset=dataset,
#             #     formatting_func=lambda examples: formatting_prompts_func(
#             #                         examples, **format_keys
#             #                     ),
#             # )
            
#             class GradientTrackingCallback(TrainerCallback):
#                 def __init__(self):
#                     self.accumulated_grads = {}
#                     self.num_backward_calls = 0
                
#                 def on_optimizer_step(self, args, state, control, model, **kwargs):
#                     self.num_backward_calls += 1
#                     for name, param in model.named_parameters():
#                         if 'embed' not in name.lower():
#                             if param.grad is not None:
#                                 if name not in self.accumulated_grads:
#                                     self.accumulated_grads[name] = param.grad.to('cpu').detach().clone()
#                                 else:
#                                     self.accumulated_grads[name] += param.grad.to('cpu').detach().clone()
#                     return control
                
#                 def get_total_grads(self):
#                     if not self.accumulated_grads:
#                         return torch.zeros_like(self.task_vector)
#                     grad_vector = torch.cat([grad.flatten() for k, grad in self.accumulated_grads.items()])
#                     return grad_vector

#             gradient_callback = GradientTrackingCallback()
#             trainer.add_callback(gradient_callback)
            
#             # Train for one epoch
#             trainer.train()

#             # Get gradients and move to CPU immediately
#             grad = gradient_callback.get_total_grads().cpu()
#             grad = grad * self.task_vector.cpu()

#             # Clear GPU memory
#             print("Clearing GPU memory...")
#             del self.model
#             del trainer
#             torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
#             # Update mask on CPU
#             with torch.no_grad():
#                 derivative = sigmoid(self.mask) * (1 - sigmoid(self.mask))
#                 reg_term = self.graft_args['l1_strength'] * torch.where(self.mask > 0, derivative, -derivative)
                
#                 self.mask -= self.graft_args['lr'] * grad * derivative - reg_term
#                 print("Gradient step on mask complete")

#                 cur_mask = self.mask.clone()
#                 cur_mask = torch.round(sigmoid(cur_mask))
#                 print('Proportion in my mask: ', len(torch.nonzero(cur_mask.bool())) / self.num_params)


# class Stitcher(nn.Module):
#     def __init__(self, trainable_params, model, pretrained_model, finetuned_models, masks):
#         super().__init__()
#         self.params = trainable_params
#         self.pretrained_model = pretrained_model.cpu()  # Keep on CPU
#         self.finetuned_models = [model.cpu() for model in finetuned_models]  # Keep on CPU
#         self.model = model

#         self.masks = [mask.cpu() for mask in masks]  # Keep masks on CPU
#         if len(self.masks) > 1:
#             self.masks = self.get_average_masks()
#         self.task_vector = torch.zeros_like(get_task_vector(self.finetuned_models[0], self.pretrained_model)).cpu()
        
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     def get_average_masks(self):
#         def reciprocal_with_zero(tensor):
#             mask = tensor == 0
#             reciprocal = torch.reciprocal(tensor)
#             reciprocal = reciprocal.masked_fill(mask, 0)
#             return reciprocal

#         output_masks = []
#         for i in range(len(self.masks)):
#             output_mask = self.masks[i].clone().detach()
#             for j in range(len(self.masks)):
#                 if i == j: continue
#                 intersect = torch.logical_and(self.masks[i], self.masks[j])
#             output_mask = output_mask + intersect
#             output_mask = reciprocal_with_zero(output_mask)
#             output_masks.append(output_mask)

#         return output_masks

#     def interpolate_models(self, move_to_gpu=False):
#         # Compute task vectors on CPU
#         self.task_vector.zero_()
#         for finetuned_model, mask in zip(self.finetuned_models, self.masks):
#             with torch.no_grad():
#                 self.task_vector += mask * get_task_vector(finetuned_model, self.pretrained_model)
    
#         # Create model on CPU first
#         self.model = vector_to_state_dict(self.task_vector, self.pretrained_model, return_dict=False)
        
#         # Only move to GPU if requested
#         if move_to_gpu:
#             print("Moving stitched model to GPU...")
#             self.model = self.model.to(self.device)
        
#         return self.model
