from pathlib import Path
from typing import Optional

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from modules.shared import shared
from modules.logging_colors import logger
from modules.models import get_max_memory_dict

