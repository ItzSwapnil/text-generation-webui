import torch
import transformers
from transformers import LogitsWarper, LogitsProcessorList, LogitNormalization

class TailFreeLogitsWarper(LogitsWarper):
    def __init__(self, tfs: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        super().__init__()
        self.tfs = tfs
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        probs = sorted_logits.softmax(dim=-1)

        d2 = probs.diff().diff().abs()
        normalized_d2 = d2 / d2.sum(dim=-1, keepdim=True)
        normalized_d2_cdf = normalized_d2.cumsum(dim=-1)

        sorted_indices_to_remove = (normalized_d2_cdf > self.tfs) & (sorted_indices != 0)

        sorted_indices_to_remove = torch.cat(
            (
                torch.zeros(scores.shape[0], 1, dtype=torch.bool, device=scores.device),
                sorted_indices_to_remove,
                torch.ones(scores.shape[0], 1, dtype=torch.bool, device=scores.device),
            ),
            dim=-1,
        )

        if self.min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :self.min_tokens_to_keep] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores

class TopALogitsWarper(LogitsWarper):
    def __init__(self, top_a: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        super().__init__()
        self.top_a = top_a
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        probs = sorted_logits.softmax(dim=-1)

        probs_max = probs[..., 0, None]
        sorted_indices_to_remove = probs < probs_max * probs_max * self.top_a

        if self.min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :self.min_tokens_to_keep] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores

def get_logits_warper_patch(self, generation_config):
    warpers = self._get_logits_warper(generation_config)
    warpers_to_add = LogitsProcessorList()
    min_tokens_to_keep = 2 if generation_config.num_beams > 1 else 1

    if generation_config.tfs is not None and 0.0 <= generation_config.tfs <= 1.0:
        warpers_to_add.append(TailFreeLogitsWarper(tfs=generation_config.tfs, min_tokens_to_keep=min_tokens_to_keep))
    if generation_config.top_a is not None and 0.0 <= generation_config.top_a <= 1.0:
        warpers_to_add.append(TopALogitsWarper(top_a=generation_config.top_a, min_tokens_to_keep=min_tokens_to_keep))

    if warpers and isinstance(warpers[-1], LogitNormalization):
        warpers = warpers[:-1] + warpers_to_add + [warpers[-1]]
    else:
        warpers += warpers_to_add

    return warpers

def generation_config_init_patch(self, **kwargs):
    super().__init__(**kwargs)
    self.tfs = kwargs.pop("tfs", 1.0)
    self.top_a = kwargs.pop("top_a", 0.0)

transformers.GenerationMixin._get_logits_warper = get_logits_warper_patch
transformers.GenerationConfig.__init__ = generation_config_init_patch
