from abc import ABC, abstractmethod
from typing import List, Optional, TypeVar, Union

import torch
from PIL import Image

T = TypeVar('T')

class AbstractMultimodalPipeline(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        'name of the pipeline, should be same as in --multimodal-pipeline'
        pass

    @property
    @abstractmethod
    def image_start(self) -> Optional[str]:
        'return image start string, string representation of image start token, or None if not applicable'
        pass

    @property
    @abstractmethod
    def image_end(self) -> Optional[str]:
        'return image end string, string representation of image end token, or None if not applicable'
        pass

    @property
    @abstractmethod
    def placeholder_token_id(self) -> int:
        'return placeholder token id'
        pass

    @property
    @abstractmethod
    def num_image_embeds(self) -> int:
        'return the number of embeds used by a single image (for example: 256 for LLaVA)'
        pass

    def embed_images(self, images: List[Image.Image]) -> torch.Tensor:
        'forward the images through vision pipeline, and return their embeddings'
        pass

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        'embed tokens, the exact function varies by LLM, for LLaMA it is `shared.model.model.embed_tokens`'
        pass

    def placeholder_embeddings(self) -> torch.Tensor:
        'get placeholder embeddings if there are multiple images, and `add_all_images_to_prompt` is False'
        return torch.zeros(self.num_image_embeds, device=self.device, dtype=self.dtype)

    def _get_device(self, setting_name: str, params: dict, requires_grad: bool = False) -> torch.device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if params[setting_name] is None else torch.device(params[setting_name])
        return device if not requires_grad else device.new('cuda:0' if torch.cuda.is_available() else 'cpu')

    def _get_dtype(self, setting_name: str, params: dict) -> torch.dtype:
        return torch.float32 if int(params[setting_name]) == 32 else torch.float16

    def to(self, device: Union[str, torch.device], dtype: torch.dtype = None, non_
