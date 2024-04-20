import os
import re
import typing as ty
from dataclasses import dataclass
from threading import Thread

import llama_cpp
import torch
from modules import shared, callbacks
from modules.logging_colors import logger


@dataclass
class LlamaCppModel:
    model: llama_cpp.Llama
    cache: ty.Optional[llama_cpp.LlamaCache] = None

    def __post_init__(self):
        self.initialized = True

    def __del__(self):
        if self.initialized:
            self.model.__del__()

    @classmethod
    def from_pretrained(cls, path: str) -> "LlamaCppModel":
        result = cls(model=None)

        cache_capacity = None
        if shared.args.cache_capacity is not None:
            capacity = shared.args.cache_capacity
            unit = capacity[-2:].upper()
            capacity = int(capacity[:-2])
            if unit == "Gi":
                cache_capacity = capacity * 1024 * 1024 * 1024
            elif unit == "Mi":
                cache_capacity = capacity * 1024 * 1024
            else:
                cache_capacity = capacity

        logger.info(f"Cache capacity is {cache_capacity} bytes")

        params = {
            "model_path": str(path),
            "n_ctx": shared.args.n_ctx,
            "seed": int(shared.args.llama_cpp_seed),
            "n_threads": shared.args.threads or None,
            "n_batch": shared.args.n_batch,
            "use_mmap": not shared.args.no_mmap,
            "use_mlock": shared.args.mlock,
            "n_gpu_layers": shared.args.n_gpu_layers,
        }
        result.model = llama_cpp.Llama(**params)
        if cache_capacity is not None:
            result.cache = llama_cpp.LlamaCache(capacity_bytes=cache_capacity)
            result.model.set_cache(result.cache)

        return result

    def encode(self, string: str) -> torch.Tensor:
        if isinstance(string, str):
            string = string.encode()
        return torch.tensor(self.model.tokenize(string), dtype=torch.long)

    def generate(
        self,
        context: ty.Union[str, torch.Tensor],
        token_count: int = 20,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        callback: ty.Optional[callbacks.Callback] = None,
    ) -> str:
        if not isinstance(context, torch.Tensor):
            context = self.encode(context)

        def stream_callback(completion_chunk: dict) -> None:
            text = completion_chunk["choices"][0]["text"]
            output += text
            if callback:
                callback(text)

        completion_chunks = self.model.create_completion(
            prompt=context,
            max_tokens=token_count,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repetition_penalty,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            stream=True,
            callback=stream_callback,
        )
        output = ""
        return output

    def generate_with_streaming(
        self,
        context: ty.Union[str, torch.Tensor],
        token_count: int = 20,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        callback: ty.Optional[callbacks.Callback] = None,
    ) -> ty.Generator[str, ty.Any, ty.Any]:
        if not isinstance(context, torch.Tensor):
            context = self.encode(context)

        def stream_generator() -> ty.Generator[str, ty.Any, ty.Any]:
            def stream_callback(completion_chunk:
