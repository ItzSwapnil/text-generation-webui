import inspect
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Union

import accelerate
import torch
import transformers
from safetensors.torch import load_file as safe_load
from transformers import AutoConfig, AutoModelForCausalLM, LlamaDecoderLayer

try:
    import llama_inference_offload
except ImportError:
    print('Failed to load GPTQ-for-LLaMa')
    print('See https://github.com/oobabooga/text-generation-webui/blob/main/docs/GPTQ-models-(4-bit-mode).md')
    sys.exit(-1)

try:
    from quant import make_quant, make_quant_attn, make_quant_linear, make_fused_mlp, autotune_warmup_linear, autotune_warmup_fused
    is_triton = False
except ImportError:
    import quant
    is_triton = True

def _load_quant(
    model: AutoModelForCausalLM,
    checkpoint: str,
    wbits: int,
    groupsize: int = -1,
    faster_kernel: bool = False,
    exclude_layers: List[str] = None,
    kernel_switch_threshold: int = 128,
    eval: bool = True,
) -> AutoModelForCausalLM:
    """
    Loads a quantized model checkpoint.

    Args:
        model (AutoModelForCausalLM): The model to load the quantized weights into.
        checkpoint (str): The path to the quantized model checkpoint.
        wbits (int): The number of bits to use for quantization.
        groupsize (int, optional): The group size to use for quantization. Defaults to -1.
        faster_kernel (bool, optional): Whether to use a faster kernel for quantization. Defaults to False.
        exclude_layers (List[str], optional): A list of layer names to exclude from quantization. Defaults to None.
        kernel_switch_threshold (int, optional): The kernel switch threshold for quantization. Defaults to 128.
        eval (bool, optional): Whether to set the model to evaluation mode. Defaults to True.

    Returns:
        AutoModelForCausalLM: The model with the loaded quantized weights.
    """
    exclude_layers = exclude_layers or ['lm_head']

    def noop(*args, **kwargs):
        pass

    config = AutoConfig.from_pretrained(model.config.model_type, trust_remote_code=shared.args.trust_remote_code)
    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=shared.args.trust_remote_code)
    torch.set_default_dtype(torch.float)
    if eval:
        model = model.eval()

    layers = {name: value for name, value in model.named_modules() if not any(exclude in name for exclude in exclude_layers)}

    if not is_triton:
        gptq_args = inspect.getfullargspec(make_quant).args

        make_quant_kwargs: Dict[str, Union[AutoModelForCausalLM, List[str], int, bool]] = {
            'module': model,
            'names': list(layers.keys()),
            'bits': wbits,
        }
        if 'groupsize' in gptq_args:
            make_quant_kwargs['groupsize'] = groupsize
        if 'faster' in gptq_args:
            make_quant_kwargs['faster'] = faster_kernel
        if 'kernel_switch_threshold' in gptq_args:
            make_quant_kwargs['kernel_switch_threshold'] = kernel_switch_threshold

        make_quant(**make_quant_kwargs)
    else:
        quant.make_quant_linear(model, list(layers.keys()), wbits, groupsize)

    del layers
    if checkpoint.endswith('.safetensors'):
        model.load_state_dict(safe_load(checkpoint), strict=False)
    else:
        model.load_state_dict(torch.load(checkpoint), strict=False)

    if is_triton:
        if shared.args.quant_attn:
            make_quant_attn(model)

        if eval and shared.args.fused_mlp:
            make_fused_mlp(model)

        if shared.args.warmup_autotune:
            autotune_warmup_linear(model, transpose=not eval)
            if eval and shared.args.fused_mlp:
                autotune_warmup_fused(model)

    model.seqlen = 2048
    return model

def find_quantized_model_file(model_name: str) -> Path:
    """
    Finds the quantized model file for the given model name.

    Args:
        model_name (str): The name of the model.

    Returns:
        Path: The path to the quantized model file.
    """
    path_to_model = Path(f'{shared.args.model_dir}/{model_name}')
    pt_path = None
    priority_name_list = [
        Path(f'{shared.args.model_dir}/{model_name}{hyphen
