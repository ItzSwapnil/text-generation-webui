import copy
import os
from pathlib import Path
import numpy as np

import modules.shared as shared
from modules.callbacks import Iteratorize
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

np.set_printoptions(precision=4, suppress=True, linewidth=200)

def get_model_path(path):
    if not path.is_file():
        raise FileNotFoundError(f"Model file '{path}' not found.")
    return str(path)

def get_tokenizer_path(path):
    tokenizer_path = Path(f"{path.parent}/20B_tokenizer.json")
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Tokenizer file '{tokenizer_path}' not found.")
    return tokenizer_path

class RWKVModel:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(cls, path, dtype="fp16", device="cuda"):
        path = get_model_path(path)
        tokenizer_path = get_tokenizer_path(path)
        os.environ["RWKV_CUDA_ON"] = '1' if shared.args.rwkv_cuda_on else '0'
        model = RWKV(model=path, strategy=f'{device} {dtype}')
        pipeline = PIPELINE(model, str(tokenizer_path))
        result = cls()
        result.pipeline = pipeline
        result.model = model
        result.cached_context = ""
        result.cached_model_state = None
        result.cached_output_logits = None
        return result

    # ... rest of the class methods ...

class RWKVTokenizer:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(cls, path):
        tokenizer_path = get_tokenizer_path(path)
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        result = cls()
        result.tokenizer = tokenizer
        return result

    # ... rest of the class methods ...

    def __str__(self):
        return f"RWKVTokenizer (path: {self.tokenizer.file})"

    def __repr__(self):
        return self.__str__()

RWKVModel.__str__ = lambda self: f"RWKVModel (path: {get_model_path(Path(self.model.model))})"
RWKVModel.__repr__ = RWKVModel.__str__
