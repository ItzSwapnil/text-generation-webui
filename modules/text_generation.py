import ast
import torch
import numpy as np
import random

import transformers
from modules.shared import shared, clear_torch_cache, local_rank
from modules.callbacks import Iteratorize, Stream, _SentinelTokenStoppingCriteria
from modules.extensions import apply_extensions
from modules.logging_colors import logger
from modules.models import get_model

def generate_reply(*args, **kwargs):
    shared.generation_lock.acquire()
    try:
        for result in _generate_reply(*args, **kwargs):
            yield result
    finally:
        shared.generation_lock.release()

def get_max_prompt_length(state):
    max_length = state['truncation_length'] - state['max_new_tokens']
    if shared.max_prompt_length_adjustment:
        max_length -= shared.soft_prompt_tensor.shape[1]

    return max_length, max_length - shared.soft_prompt_tensor.shape[1] if shared.soft_prompt_tensor is not None else max_length

def encode(prompt, add_special_tokens=True, add_bos_token=True, truncation_length=None):
    model = get_model()
    if model.model_type in ['rwkv', 'llamacpp']:
        input_ids = model.tokenizer.encode(str(prompt))
        input_ids = np.array(input_ids).reshape(1, len(input_ids))
        return input_ids
    else:
        input_ids = model.tokenizer.encode(str(prompt), return_tensors='pt', add_special_tokens=add_special_tokens)

        if not add_bos_token and input_ids[0][0] == model.tokenizer.bos_token_id:
            input_ids = input_ids[:, 1:]

        if model.tokenizer_name == 'LlamaTokenizer' and input_ids[0][0] == 29871:
            input_ids = input_ids[:, 1:]

    if truncation_length is not None:
        input_ids = input_ids[:, -truncation_length:]

    if model.model_type == 'llamacpp' or shared.args.cpu:
        return input_ids
    elif shared.args.flexgen:
        return input_ids.numpy()
    elif shared.args.deepspeed:
        return input_ids.to(device=local_rank)
    elif torch.has_mps:
        device = torch.device(('mps'))
        return input_ids.to(device)
    else:
        return input_ids.cuda()

def decode(output_ids, skip_special_tokens=True):
    model = get_model()
    return model.tokenizer.decode(output_ids, skip_special_tokens=skip_special_tokens)

def generate_softprompt_input_tensors(input_ids):
    model = get_model()
    inputs_embeds = model.transformer.wte(input_ids)
    inputs_embeds = torch.cat((shared.soft_prompt_tensor, inputs_embeds), dim=1)
    filler_input_ids = torch.zeros((1, inputs_embeds.shape[1]), dtype=input_ids.dtype).to(model.device)
    return inputs_embeds, filler_input_ids

def fix_gpt4chan(s):
    for i in range(10):
        s = re.sub("--- [0-9]*\n>>[0-9]*\n---", "---", s)
        s = re.sub("--- [0-9]*\n *\n---", "---", s)
        s = re.sub("--- [0-9]*\n\n\n---", "---", s)

    return s

def fix_galactica(s):
    s = s.replace(r'\[', r'$')
    s = s.replace(r'\]', r'$')
    s = s.replace(r'\(', r'$')
    s = s.replace(r'\)', r'$')
    s = s.replace(r'$$', r'$')
    s = re.sub(r'\n', r'\n\n', s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s

def get_reply_from_output_ids(output_ids, input_ids, original_question, state, is_chat=False):
    model = get_model()
    if model.model_type == 'HF_seq2seq':
        reply = decode(output_ids, state['skip_special_tokens'])
    else:
        new_tokens = len(output_ids) - len(input_ids[0])
        reply = decode(output_ids[-new_tokens:], state['skip_special_tokens'])

        if model.tokenizer_name == 'LlamaTokenizer' and len(output_ids) > 0:
            if model.tokenizer.convert_ids_to_tokens(int(output_ids), skip_special_tokens=False)[0].startswith(' '):
                reply = ' ' + reply

    if not is_chat:
        reply = apply_extensions('output', reply)

    return reply

def formatted_outputs(reply, model_name):
    if model_name == 'gpt4chan':
        reply = fix_gpt4chan(reply)
        return reply, generate_4chan_html(reply, model_name)
    else:
        return reply, generate_basic_html(reply, model_name)

def set_manual_seed(seed):
    seed = int(seed)
    if seed == -1:
        seed = random.randint(1, 1 << 31)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return seed

def stop_everything_event():
    shared.stop
