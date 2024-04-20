import gc
import traceback
from queue import Queue
from threading import Thread
from typing import Any
from typing import List
from typing import Optional

import torch
from transformers import StoppingCriteria

import modules.shared as shared
from typing import TypeVar

T = TypeVar("T")


class _SentinelTokenStoppingCriteria(StoppingCriteria):
    """
    Custom StoppingCriteria that stops generation when a sentinel token is found.
    """

    def __init__(self, sentinel_token_ids: List[torch.Tensor], starting_idx: int):
        super().__init__()
        self.sentinel_token_ids = sentinel_token_ids
        self.starting_idx = starting_idx
        self.shortest = min([x.shape[-1] for x in sentinel_token_ids])

    def __call__(self, input_ids: torch.LongTensor, _scores: torch.FloatTensor) -> bool:
        """
        Checks if the input_ids tensor contains any of the sentinel tokens.

        :param input_ids: The input_ids tensor.
        :param _scores: The scores tensor.
        :return: True if a sentinel token is found, False otherwise.
        """
        for sample in input_ids:
            trimmed_sample = sample[self.starting_idx:]
            trimmed_len = trimmed_sample.shape[-1]
            if trimmed_len < self.shortest:
                continue

            for sentinel in self.sentinel_token_ids:
                sentinel_len = sentinel.shape[-1]
                if trimmed_len < sentinel_len:
                    continue

                window = trimmed_sample[-sentinel_len:]
                if torch.all(torch.eq(sentinel, window)):
                    return True

        return False


class Stream(StoppingCriteria):
    """
    Custom StoppingCriteria that calls a callback function for each input_ids tensor.
    """

    def __init__(self, callback_func: Optional[Callable[[torch.Tensor], Any]] = None):
        super().__init__()
        self.callback_func = callback_func

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        """
        Calls the callback function if it's not None.

        :param input_ids: The input_ids tensor.
        :param scores: The scores tensor.
        :return: False.
        """
        if self.callback_func is not None:
            self.callback_func(input_ids[0])
        return False


class Iteratorize:
    """
    Transforms a function that takes a callback
    into a lazy iterator (generator).
    """

    def __init__(self, func: Callable, kwargs: dict = None, callback: Optional[Callable] = None):
        self.mfunc = func
        self.c_callback = callback
        self.q = Queue()
        self.sentinel = object()
        self.kwargs = kwargs or {}
        self.stop_now = False

        def _callback(val: T) -> None:
            if self.stop_now or shared.stop_everything:
                raise ValueError
            self.q.put(val)

        def gentask() -> None:
            try:
                ret = self.mfunc(callback=_callback, **self.kwargs)
            except ValueError:
                pass
            except Exception as e:
                traceback.print_exc()
                pass

            clear_torch_cache()
            self.q.put(self.sentinel)
            if self.c_callback:
                self.c_callback(ret)

        self.thread = Thread(target=gentask)
        self.thread.start()

    def __iter__(self) -> 'Iteratorize':
        return self

    def __next__(self) -> T:
        obj = self.q.get(True, None)
        if obj is self.sentinel:
            raise StopIteration
        else:
            return obj

    def __del__(self) -> None:
        clear_torch_cache()

    def __enter__(self) -> 'Iteratorize':
        return
