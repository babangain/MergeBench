import sys
import os, copy
import torch
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import OrderedDict
import torch.nn.functional as F
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

## Model conversion utils
def state_dict_to_vector(state_dict, remove_keys=[]):
    shared_state_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in shared_state_dict:
            del shared_state_dict[key]
    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    return torch.nn.utils.parameters_to_vector(
        [value.reshape(-1) for key, value in sorted_shared_state_dict.items()]
    )


def add_ptm_to_tv(tv_dict, ptm_dict):
    assert set(tv_dict.keys()) == set(
        ptm_dict.keys()
    ), "Differing parameter names in models."
    final_dict = copy.deepcopy(tv_dict)
    for k, v in ptm_dict.items():
        final_dict[k] = tv_dict[k] + v
    return final_dict


def check_parameterNamesMatch(checkpoints):
    parameter_names = set(checkpoints[0].keys())

    if len(checkpoints) >= 2:
        # raise ValueError("Number of models is less than 2.")
        for checkpoint in checkpoints[1:]:
            current_parameterNames = set(checkpoint.keys())
            if current_parameterNames != parameter_names:
                raise ValueError(
                    "Differing parameter names in models. "
                    f"The different parameters are {parameter_names.symmetric_difference(current_parameterNames)}"
                )

def check_state_dicts_equal(state_dict1, state_dict2):
    if set(state_dict1.keys()) != set(state_dict2.keys()):
        return False

    for key in state_dict1.keys():
        if not torch.equal(state_dict1[key], state_dict2[key]):
            return False

    return True



## TIES MERGING UTILS

def topk_values_mask(M, K=0.7, return_mask=False):
    if K > 1:
        K /= 100

    original_shape = M.shape
    if M.dim() == 1:
        M = M.unsqueeze(0)

    n, d = M.shape
    k = int(d * K)
    k = d - k  # Keep top k elements instead of bottom k elements

    # Find the k-th smallest element by magnitude for each row
    kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
    # Create a mask tensor with True for the top k elements in each row
    mask = M.abs() >= kth_values
    final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

    if return_mask:
        return M * final_mask, final_mask.float().mean(dim=1), final_mask
    return M * final_mask, final_mask.float().mean(dim=1)


def resolve_zero_signs(sign_to_mult, method="majority"):
    majority_sign = torch.sign(sign_to_mult.sum())

    if method == "majority":
        sign_to_mult[sign_to_mult == 0] = majority_sign
    elif method == "minority":
        sign_to_mult[sign_to_mult == 0] = -1 * majority_sign
    return sign_to_mult


def resolve_sign(Tensor):
    sign_to_mult = torch.sign(Tensor.sum(dim=0))
    sign_to_mult = resolve_zero_signs(sign_to_mult, "majority")
    return sign_to_mult


def disjoint_merge(Tensor, merge_func, sign_to_mult):
    merge_func = merge_func.split("-")[-1]

    # If sign is provided then we select the corresponding entries and aggregate.
    if sign_to_mult is not None:
        rows_to_keep = torch.where(
            sign_to_mult.unsqueeze(0) > 0, Tensor > 0, Tensor < 0
        )
        selected_entries = Tensor * rows_to_keep
    # Else we select all non-zero entries and aggregate.
    else:
        rows_to_keep = Tensor != 0
        selected_entries = Tensor * rows_to_keep

    if merge_func == "mean":
        non_zero_counts = (selected_entries != 0).sum(dim=0).float()
        disjoint_aggs = torch.sum(selected_entries, dim=0) / torch.clamp(non_zero_counts, min=1)
    elif merge_func == "sum":
        disjoint_aggs = torch.sum(selected_entries, dim=0)
    elif merge_func == "max":
        disjoint_aggs = selected_entries.abs().max(dim=0)[0]
        disjoint_aggs *= sign_to_mult
    else:
        raise ValueError(f"Merge method {merge_func} is not defined.")

    return disjoint_aggs


def ties_merging(
        flat_task_checks,
        reset_thresh=None,
        merge_func="",
):
    all_checks = flat_task_checks.clone()
    updated_checks, *_ = topk_values_mask(
        all_checks, K=reset_thresh, return_mask=False
    )
    print(f"RESOLVING SIGN")
    final_signs = resolve_sign(updated_checks)
    assert final_signs is not None

    print(f"Disjoint AGGREGATION: {merge_func}")
    merged_tv = disjoint_merge(updated_checks, merge_func, final_signs)

    return merged_tv

def disjoint_merge_split(Tensor, merge_func, sign_to_mult):
    merge_func = merge_func.split("-")[-1]

    # If sign is provided then we select the corresponding entries and aggregate.
    if sign_to_mult is not None:
        rows_to_keep = torch.where(
            sign_to_mult.unsqueeze(0) > 0, Tensor > 0, Tensor < 0
        )
        selected_entries = Tensor * rows_to_keep
    # Else we select all non-zero entries and aggregate.
    else:
        rows_to_keep = Tensor != 0
        selected_entries = Tensor * rows_to_keep

    if merge_func == "sum":
        disjoint_aggs = torch.sum(selected_entries, dim=0)
    else:
        raise ValueError(f"Merge method {merge_func} is not defined.")

    return selected_entries, disjoint_aggs


def ties_merging_split(
        flat_task_checks,
        reset_thresh=None,
        merge_func="",
):
    all_checks = flat_task_checks.clone()
    updated_checks, *_ = topk_values_mask(
        all_checks, K=reset_thresh, return_mask=False
    )
    print(f"RESOLVING SIGN")
    final_signs = resolve_sign(updated_checks)
    assert final_signs is not None

    print(f"Disjoint AGGREGATION: {merge_func}")
    selected_entries, merged_tv = disjoint_merge_split(updated_checks, merge_func, final_signs)

    return selected_entries, merged_tv