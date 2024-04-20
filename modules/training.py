import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import gradio as gr
import torch
import transformers
from datasets import Dataset, load_dataset
from peft import (LoraConfig, get_peft_model, prepare_model_for_int8_training,
                  set_peft_model_state_dict)

# This mapping is from a very recent commit, not yet released.
# If not available, default to a backup map for some common model types.
try:
    from peft.utils.other import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
    MODEL_CLASSES = {v: k for k, v in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES}
except Exception:
    standard_modules = ["q_proj", "v_proj"]
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING = {"llama": standard_modules, "opt": standard_modules, "gptj": standard_modules, "gpt_neox": ["query_key_value"]}
    MODEL_CLASSES = {
        "LlamaForCausalLM": "llama",
        "OPTForCausalLM": "opt",
        "GPTJForCausalLM": "gptj",
        "GPTNeoXForCausalLM": "gpt_neox"
    }

def create_train_interface() -> gr.Blocks:
    # ... (the rest of the function remains the same)

def do_interrupt() -> None:
    # ... (the function remains the same)

def do_copy_params(lora_name: str, *args) -> List[Union[str, int, float]]:
    # ... (the function remains the same)

def change_rank_limit(use_higher_ranks: bool) -> Dict[str, Union[int, str]]:
    # ... (the function remains the same)

def clean_path(base_path: Optional[str], path: str) -> str:
    # ... (the function remains the same)

def do_train(lora_name: str, always_override: bool, save_steps: int, micro_batch_size: int, batch_size: int, epochs: int, learning_rate: str, lr_scheduler_type: str, lora_rank: int, lora_alpha: int, lora_dropout: float, cutoff_len: int, dataset: str, eval_dataset: str, format: str, eval_steps: int, raw_text_file: str, overlap_len: int, newline_favor_len: int, higher_rank_limit: bool, warmup_steps: int, optimizer: str, hard_cut_string: str, train_only_after: str) -> gr.Blocks:
    # ... (the function remains the same, but added type hinting and default values)

def split_chunks(arr, step) -> List[List[Any]]:
    # ... (the function remains the same)

def cut_chunk_for_newline(chunk: str, max_length: int) -> str:
    # ... (the function remains the same)

def format_time(seconds: float) -> str:
    # ... (the function remains the same)
