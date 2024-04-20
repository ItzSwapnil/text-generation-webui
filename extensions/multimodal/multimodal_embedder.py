import base64
import re
import torch
from io import BytesIO
from PIL import Image
from typing import Any, List, Optional

import torch
from typing import Any, List, Optional
from dataclasses import dataclass
from io import BytesIO
from PIL import Image
from typing import Any, List, Optional

import torch
from typing import Any, List, Optional

from modules import shared
from modules.logging_colors import logger
from modules.text_generation import encode, get_max_prompt_length


@dataclass
class PromptPart:
    text: str
    image: Optional[Image.Image] = None
    is_image: bool = False
    input_ids: Optional[torch.Tensor] = None
    embedding: Optional[torch.Tensor] = None


class MultimodalEmbedder:
    def __init__(self, params: dict):
        if "load_pipeline" not in params:
            raise ValueError("The 'params' dictionary must contain the 'load_pipeline' key.")
        pipeline, source = load_pipeline(params)
        self.pipeline = pipeline
        logger.info(f'Multimodal: loaded pipeline {self.pipeline.name()} from pipelines/{source} ({self.pipeline.__class__.__name__})')

    def _split_prompt(self, prompt: str, load_images: bool = False) -> List[PromptPart]:
        """Splits a prompt into a list of `PromptParts` to separate image data from text.
        It will also append `image_start` and `image_end` before and after the image, and optionally parse and load the images,
        if `load_images` is `True`.
        """
        parts: List[PromptPart] = []
        curr = 0
        while True:
            match = re.search(r'<img src="data:image/jpeg;base64,([A-Za-z0-9+/=]+)">', prompt[curr:])
            if match is None:
                # no more image tokens, append the rest of the prompt
                if curr > 0:
                    # add image end token after last image
                    parts.append(PromptPart(text=self.pipeline.image_end() + prompt[curr:]))
                else:
                    parts.append(PromptPart(text=prompt))
                break
            # found an image, append image start token to the text
            if match.start() > 0:
                parts.append(PromptPart(text=prompt[curr:curr + match.start()] + self.pipeline.image_start()))
            else:
                parts.append(PromptPart(text=self.pipeline.image_start()))
            # append the image
            try:
                parts.append(PromptPart(
                    text=match.group(0),
                    image=Image.open(BytesIO(base64.b64decode(match.group(1)))) if load_images else None,
                    is_image=True
                ))
            except Exception as e:
                logger.error(f"Error loading image: {e}")
                parts.append(PromptPart(text=match.group(0), is_image=True))
            curr += match.end()
        return parts

    def _len_in_tokens_prompt_parts(self, parts: List[PromptPart]) -> int:
        """Total length in tokens of all `parts`"""
        tokens = 0
        for part in parts:
            if part.is_image:
                tokens += self.pipeline.num_image_embeds()
            elif part.input_ids is not None:
                tokens += len(part.input_ids)
            else:
                tokens += len(encode(part.text)[0])
        return tokens

    def len_in_tokens(self, prompt: str) -> int:
        """Total length in tokens for a given text `prompt`"""
        parts = self._split_prompt(prompt, False)
        return self._len_in_tokens_prompt_parts(parts)

    def _encode_single_text(self, part: PromptPart, add_bos_token: bool) -> PromptPart:
        """Encode a single prompt `part` to `input_ids`. Returns a `PromptPart`"""
        if part.is_image:
            placeholders = torch.ones((self.pipeline.num_image_embeds())) * self.pipeline.placeholder_token_id()
            part.input_ids = placeholders.to(shared.model.device, dtype=torch.int64)
        else:
            part.input_ids = encode(part.text, add_bos_token=add_bos_token)[0].to(shared.model.device, dtype=torch.int64)
        return part

    @staticmethod
    def _num_images(parts: List[PromptPart]) -> int:
        count = 0
        for part in parts:
            if part.is_image:
                count += 1
        return count

    def _encode_text(self, state, parts: List[PromptPart]) -> List[PromptPart]:
        """Encode text to token_ids, also truncate the prompt, if necessary.

        The chat/instruct mode should make prompts that fit in get_max_prompt_length, but if max_new_tokens are set
        such that the context + min_rows don't fit, we can get a prompt which is too long.
        We can't truncate image embeddings, as it leads to broken generation, so remove the images instead and warn the user
        """
        encoded: List[PromptPart] = []
        for i, part in enumerate(parts):
            encoded.append(self._encode_single_text(part, i == 0
