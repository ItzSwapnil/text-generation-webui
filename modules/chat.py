import ast
import base64
import copy
import functools
import io
import json
import re
import yaml
from datetime import datetime
from pathlib import Path
from PIL import Image
from typing import Any, Dict, List, Optional

import modules.shared as shared
from modules.extensions import apply_extensions
from modules.html_generator import chat_html_wrapper, make_thumbnail
from modules.logging_colors import logger
from modules.text_generation import (generate_reply, get_encoded_length, get_max_prompt_length)
from modules.utils import replace_all


def get_turn_substrings(state: Dict[str, str], instruct: bool = False) -> Dict[str, str]:
    ...


def generate_chat_prompt(user_input: str, state: Dict[str, str], **kwargs: Any) -> str:
    ...


def get_stopping_strings(state: Dict[str, str]) -> List[str]:
    ...


def extract_message_from_reply(reply: str, state: Dict[str, str]) -> Tuple[str, bool]:
    ...


def chatbot_wrapper(text: str, history: Dict[str, List[Tuple[str, str]]], state: Dict[str, str], regenerate: bool = False, _continue: bool = False, loading_message: bool = True) -> List[Dict[str, List[Tuple[str, str]]]]:
    ...


def impersonate_wrapper(text: str, start_with: str, state: Dict[str, str]) -> str:
    ...


def generate_chat_reply(text: str, history: Dict[str, List[Tuple[str, str]]], state: Dict[str, str], regenerate: bool = False, _continue: bool = False) -> List[Dict[str, List[Tuple[str, str]]]]:
    ...


def generate_chat_reply_wrapper(text: str, start_with: str, state: Dict[str, str], regenerate: bool = False, _continue: bool = False) -> str:
    ...


def remove_last_message() -> str:
    ...


def send_last_reply_to_input() -> str:
    ...


def replace_last_reply(text: str) -> None:
    ...


def send_dummy_message(text: str) -> None:
    ...


def send_dummy_reply(text: str) -> None:
    ...


def clear_chat_log(greeting: str, mode: str) -> None:
    ...


def redraw_html(name1: str, name2: str, mode: str, style: str, reset_cache: bool = False) -> str:
    ...


def tokenize_dialogue(dialogue: str, name1: str, name2: str) -> List[List[str]]:
    ...


def save_history(mode: str, timestamp: bool = False) -> Path:
    ...


def load_history(file: bytes, name1: str, name2: str) -> None:
    ...


def replace_character_names(text: str, name1: str, name2: str) -> str:
    ...


def build_pygmalion_style_context(data: Dict[str, str]) -> str:
    ...


def generate_pfp_cache(character: str) -> Optional[Image.Image]:
    ...


def load_character(character: str, name1: str, name2: str, instruct: bool = False) -> Tuple[str, str, Optional[Image.Image], str, str, str]:
    ...


def upload_character(json_file: str, img: Optional[bytes], tavern: bool = False) -> str:
    ...


def upload_tavern_character(img: Optional[bytes], name1: str, name2: str) -> str:
    ...


def upload_your_profile_picture(img: Optional[Image.Image]) -> None:
    ...


def delete_file(path: Path) -> None:
    ...


def save_character(name: str, greeting: str, context: str, picture: Optional[Image.Image], filename: str, instruct: bool = False) -> None:
    ...


def delete_character(name: str, instruct: bool = False) -> None:
    ...

