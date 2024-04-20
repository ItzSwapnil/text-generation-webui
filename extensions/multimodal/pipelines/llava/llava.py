import time
from abc import abstractmethod
from typing import List, Tuple

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModel

from extensions.multimodal.abstract_pipeline import AbstractMultimodalPipeline
from modules import shared
from modules.logging_colors import logger
from modules.text_generation import encode

class LLaVA_Vision_Projector:
    def __init__(self, projector_path: str, device: torch.device, dtype: torch.dtype):
        self.projector = torch.nn.Linear(*self._get_projector_shape(projector_path)).to(device, dtype=dtype)
        self.projector.weight = torch.nn.Parameter(torch.load(f"{projector_path}/weight.pt", map_location=device), False)
        self.projector.bias = torch.nn.Parameter(torch.load(f"{projector_path}/bias.pt", map_location=device), False)

    @staticmethod
    def _get_projector_shape(projector_path: str) -> Tuple[int, int]:
        with open(f"{projector_path}/shape.txt", "r") as f:
            return tuple(map(int, f.read().strip().split()))

class LLaVA_v0_Pipeline(AbstractMultimodalPipeline):
    CLIP_REPO = "openai/clip-vit-large-patch14"

    def __init__(self, params: dict) -> None:
        super().__init__()
        self.clip_device = self._get_device("vision_device", params)
        self.clip_dtype = self._get_dtype("vision_bits", params)
        self.projector_device = self._get_device("projector_device", params)
        self.projector_dtype = self._get_dtype("projector_bits", params)
        self.image_processor, self.vision_tower, self.mm_projector = self._load_models()

    def _load_models(self):
        start_ts = time.time()

        logger.info(f"LLaVA - Loading CLIP from {LLaVA_v0_Pipeline.CLIP_REPO} as {self.clip_dtype} on {self.clip_device}...")
        image_processor = CLIPImageProcessor.from_pretrained(LLaVA_v0_Pipeline.CLIP_REPO, torch_dtype=self.clip_dtype)
        vision_tower = CLIPVisionModel.from_pretrained(LLaVA_v0_Pipeline.CLIP_REPO, torch_dtype=self.clip_dtype).to(self.clip_device)

        logger.info(f"LLaVA - Loading projector...")
        projector_path = hf_hub_download(self.llava_projector_repo(), self.llava_projector_filename())
        mm_projector = LLaVA_Vision_Projector(projector_path, self.projector_device, self.projector_dtype)

        logger.info(f"LLaVA supporting models loaded, took {time.time() - start_ts:.2f} seconds")
        return image_processor, vision_tower, mm_projector

    @staticmethod
    def image_start() -> str:
        return "<im_start>"

    @staticmethod
    def image_end() -> str:
        return "<im_end>"

    @staticmethod
    def num_image_embeds() -> int:
        return 256

    @staticmethod
    def embed_tokens(input_ids: torch.Tensor) -> torch.Tensor:
        return shared.model.model.embed_tokens(input_ids).to(shared.model.device, dtype=shared.model.dtype)

    @staticmethod
    def placeholder_embeddings() -> torch.Tensor:
        return LLaVA_v0_Pipeline.embed_tokens(encode("<im_patch>"*256, add_bos_token=False)[0])

    def embed_images(self, images: List[Image.Image]) -> torch.Tensor:
        images = self.image_processor(images, return_tensors='pt')['pixel_values']
        images = images.to(self.clip_device, dtype=self.clip_dtype)

        with torch.no_grad():
            image_forward_outs = self.vision_tower(images, output_hidden_states=True)
            select_hidden_state_layer = -2
            select_hidden_state = image_forward_outs.hidden_states[select_hidden_state_layer]
            image_features = select_hidden_state[:, 1:].to(self.projector_device, dtype=self.projector_dtype)
            image_features = self.mm_projector.projector(image_features)
        return image_features.to(shared.model.device, dtype=shared.model.dtype)

    @abstractmethod
    def llava_projector_repo(self) -> str:
        pass

    @abstractmethod
    def llava_projector_filename(self) -> str:
        pass

class LLaVA_v0_13B_Pipeline(LLaVA_v0_Pipeline):
    def __init__(self, params: dict) -> None:
        super().__init__(params)

    @staticmethod
    def name() -> str:
        return "llava-13b"

    @staticmethod
    def placeholder_token_id() -> int:
        return 32000

    def llava_projector_repo(self) -> str:
        return "liuhaotian/LLaVA-13b-delta-v0"

    def llava_projector_filename(self) -> str:
        return "mm_projector.bin"

class LLaVA_v0_7B_Pipeline(LLaVA_v0_Pipeline):
    def __init__(self, params: dict) -> None:
        super().__init__(params)

    @static
