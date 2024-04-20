import gc
import json
import os
import re
import time
import zipfile
from pathlib import Path
import torch
from typing import Optional, Tuple
import numpy as np
import transformers
from accelerate import infer_auto_device_map, init_empty_weights
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaTokenizer,
)
from transformers.deepspeed import HfDeepSpeedConfig, is_deepspeed_zero3_enabled
from modules.shared import shared
from modules.logging_colors import logger
from modules.deepspeed_parameters import generate_ds_config
from modules.sampler_hijack import hijack_samplers
from modules.llama_attn_hijack import hijack_llama_attention

# Distributed setup
local_rank = (
    shared.args.local_rank
    if shared.args.local_rank is not None
    else int(os.getenv("LOCAL_RANK", "0"))
)
world_size = int(os.getenv("WORLD_SIZE", "1"))
torch.cuda.set_device(local_rank)


def find_model_type(model_name: str) -> str:
    """Detects the type of the given model."""
    path_to_model = Path(f"{shared.args.model_dir}/{model_name}")
    if not path_to_model.exists():
        return "None"

    model_name_lower = model_name.lower()
    if re.match(r".*rwkv.*\.pth", model_name_lower):
        return "rwkv"
    elif len(list(path_to_model.glob("*ggml*.bin"))) > 0:
        return "llamacpp"
    elif re.match(r".*ggml.*\.bin", model_name_lower):
        return "llamacpp"
    elif "chatglm" in model_name_lower:
        return "chatglm"
    elif "galactica" in model_name_lower:
        return "galactica"
    elif "llava" in model_name_lower:
        return "llava"
    elif "oasst" in model_name_lower:
        return "oasst"
    elif any((k in model_name_lower for k in ["gpt4chan", "gpt-4chan"])):
        return "gpt4chan"
    else:
        config = AutoConfig.from_pretrained(
            path_to_model, trust_remote_code=shared.args.trust_remote_code
        )
        if config.to_dict().get("is_encoder_decoder", False):
            return "HF_seq2seq"
        else:
            return "HF_generic"


def load_model(model_name: str) -> Optional[Tuple[transformers.PreTrainedModel, AutoTokenizer]]:
    """Loads the given model and tokenizer."""
    logger.info(f"Loading {model_name}...")
    t0 = time.time()

    shared.model_type = find_model_type(model_name)
    if shared.model_type == "None":
        logger.error('The path to the model does not exist. Exiting.')
        return None

    load_func = get_load_function(shared.model_type)
    output = load_func(model_name)
    if type(output) is tuple:
        model, tokenizer = output
    else:
        model = output
        if model is None:
            return None
        else:
            tokenizer = load_tokenizer(model_name, model)

    # Hijack attention with xformers
    if any((shared.args.xformers, shared.args.sdp_attention)):
        hijack_llama_attention()

    logger.info(f"Loaded the model in {(time.time() - t0):.2f} seconds.\n")
    return model, tokenizer


def load_tokenizer(model_name: str, model: transformers.PreTrainedModel) -> Optional[AutoTokenizer]:
    """Loads the tokenizer for the given model."""
    tokenizer = None

    if shared.model_type == "gpt4chan" and Path(f"{shared.args.model_dir}/gpt-j-6B/").exists():
        tokenizer = AutoTokenizer.from_pretrained(Path(f"{shared.args.model_dir}/gpt-j-6B/"))
    elif type(model) in (transformers.LlamaForCausalLM, str):
        # Try to load an universal LLaMA tokenizer
        if shared.model_type not in ["llava", "oasst"]:
            for p in [
                Path(f"{shared.args.model_dir}/llama-tokenizer/"),
                Path(f"{shared.args.model_dir}/oobabooga_llama-tokenizer/"),
            ]:
                if p.exists():
                    logger.info(f"Loading the universal LLaMA tokenizer from {p}...")
                    tokenizer = LlamaTokenizer.from_pretrained(p, clean_up_tokenization_spaces=True)
                    return tokenizer

        # Otherwise, load it from the model folder and hope that these
        # are not outdated tokenizer files.
        tokenizer = LlamaTokenizer.from_pretrained(
            Path(f"{shared.args.model_dir}/{model_name}/"), clean_up_tokenization_spaces=True
        )
        try:
            tokenizer.eos_token_id
