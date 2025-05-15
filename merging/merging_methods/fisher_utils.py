import os
import gc

import torch
from torch.nn import functional as F
from trl import SFTTrainer



class FisherTrainer(SFTTrainer):
    
    def __init__(
        self,
        fisher_variant="hard", 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.fisher_variant = fisher_variant  

    def compute_loss(self, model, inputs, num_items_in_batch, return_outputs=False):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True
        )

        logits = outputs.logits[:, -1, :]  # (batch_size, vocab_size), check last token

        if self.fisher_variant == "hard":
            log_probs = F.log_softmax(logits, dim=-1)
            _, target_labels = logits.max(dim=-1)
            loss = F.nll_loss(log_probs, target_labels)

        elif self.fisher_variant == "soft":
            probs = torch.softmax(logits, dim=-1).detach() 
            log_probs = torch.log_softmax(logits, dim=-1)

            vocab_size = probs.size(-1)
            nll_losses = []
            for label_id in range(vocab_size):
                targets = torch.full(
                    (probs.size(0),), label_id,
                    dtype=torch.long, device=probs.device
                )
                nll_loss_per_label = F.nll_loss(
                    log_probs, targets, reduction="none"
                )
                nll_losses.append(nll_loss_per_label)

            nll_losses = torch.stack(nll_losses, dim=-1)
            weighted_nll_losses = probs * nll_losses
            loss = weighted_nll_losses.sum(dim=-1).mean()

        else:
            loss = outputs.loss  
        return (loss, outputs) if return_outputs else loss
    

def save_tensor_dict(tensor_dict, path):
    os.makedirs(path, exist_ok=True)
    for key, tensor in tensor_dict.items():
        filename = os.path.join(path, key + ".pt")
        torch.save(tensor, filename)
    return


def cleanup_task_loader(task_loader):
    """
    Safely clean up task_loader and its nested trainer to reduce CPU and GPU memory usage.
    """
    trainer = getattr(task_loader, 'trainer', None)

    if trainer is not None:
        for attr in [
            'model', 'processing_class', 'train_dataset', 'eval_dataset',
            'callback_handler', 'args', 'data_collator',
            'train_dataloader', 'eval_dataloader',
            'optimizer', 'lr_scheduler',
        ]:
            if hasattr(trainer, attr):
                try:
                    setattr(trainer, attr, None)
                except Exception as e:
                    print(f"Warning: couldn't clear trainer.{attr}: {e}")
        try:
            del trainer
        except Exception as e:
            print(f"Warning: couldn't delete trainer: {e}")

    for attr in ['training_dataset', 'training_args']:
        if hasattr(task_loader, attr):
            try:
                setattr(task_loader, attr, None)
            except Exception as e:
                print(f"Warning: couldn't clear task_loader.{attr}: {e}")

    try:
        del task_loader
    except Exception as e:
        print(f"Warning: couldn't delete task_loader: {e}")

    gc.collect()
    torch.cuda.empty_cache()


def get_expected_fisher_keys(model):

    return {
        name for name, param in model.named_parameters()
        if param.requires_grad and "lm_head" not in name
    }

def is_tensor_dict_complete(path, keys):
    if not os.path.exists(path):
        print(f'{path} doesn\'t exist')
        return False
    for k in keys:
        file_path = os.path.join(path, k + ".pt")
        if not os.path.exists(file_path):
            print(f'{file_path} doesn\'t exist')
            return False
        try:
            _ = torch.load(file_path, map_location="cpu")
        except Exception:
            print(f'{file_path} is corrupted')
            return False  # File exists but is corrupted or unreadable
    return True