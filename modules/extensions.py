import importlib
import inspect
import atexit
import typing as t

import gradio as gr

from modules.shared import shared

state = {}
setup_called = set()


def load_extension(name: str):
    if name in available_extensions:
        if name != 'api':
            print(f'Loading the extension "{name}"...')
        try:
            extension_module = importlib.import_module(f"extensions.{name}.script")
            extension = getattr(extension_module, "script")
            apply_settings(extension, name)
            if extension not in setup_called and hasattr(extension, "setup"):
                setup_called.add(extension)
                extension.setup()

            state[name] = [True, i]
        except Exception as e:
            print(f'Failed to load the extension "{name}".')
            print(e)


def apply_settings(extension: t.Any, name: str):
    if not hasattr(extension, 'params'):
        return

    for param in extension.params:
        _id = f"{name}-{param}"
        if _id not in shared.settings:
            continue

        extension.params[param] = shared.settings[_id]


def get_display_name(extension: t.Any, default_name: str) -> str:
    return getattr(extension, 'params', {}).get('display_name', default_name)


def load_extensions():
    for i, name in enumerate(shared.args.extensions):
        load_extension(name)


@functools.lru_cache()
def iterator():
    for name in sorted(state, key=lambda x: state[x][1]):
        if state[name][0]:
            yield getattr(extensions, name).script, name


def is_function(func: t.Any) -> bool:
    return inspect.isfunction(func)


atexit.register(lambda: print("Cleaning up extensions..."))


import traceback
from functools import partial

import gradio as gr
from extension_manager import load_extensions, iterator, is_function, get_display_name

import extensions
import modules.shared as shared
from modules.logging_colors import logger

EXTENSION_MAP = {
    "input": partial(_apply_string_extensions, "input_modifier"),
    "output": partial(_apply_string_extensions, "output_modifier"),
    "state": _apply_state_modifier_extensions,
    "history": _apply_history_modifier_extensions,
    "bot_prefix": partial(_apply_string_extensions, "bot_prefix_modifier"),
    "tokenizer": partial(_apply_tokenizer_extensions, "tokenizer_modifier"),
    "input_hijack": _apply_input_hijack,
    "custom_generate_chat_prompt": _apply_custom_generate_chat_prompt,
    "custom_generate_reply": _apply_custom_generate_reply,
    "tokenized_length": _apply_custom_tokenized_length,
    "css": _apply_custom_css,
    "js": _apply_custom_js
}


def _apply_string_extensions(function_name: str, text: str, *args) -> str:
    for extension, _ in iterator():
        if hasattr(extension, function_name) and is_function(getattr(extension, function_name)):
            text = getattr(extension, function_name)(text, *args)

    return text


# ... other functions here ...

def create_extensions_block():
    to_display = []
    for extension, name in iterator():
        if hasattr(extension, "ui") and not (hasattr(extension, 'params') and extension.params.get('is_tab', False)):
            to_display.append((extension, name))

    # Creating the extension ui elements
    if len(to_display) > 0:
        with gr.Column(elem_id="extensions"):
            for row in to_display:
                extension, name = row
                display_name = get_display_name(extension, name)
                gr.Markdown(f"\n### {display_name}")
                extension.ui()


def create_extensions_tabs():
    for extension, name in iterator():
        if hasattr(extension, "ui") and (hasattr(extension, 'params') and extension.params.get('is_tab', False)):
            display_name = get_display_name(extension, name)
            with gr.Tab(display_name, elem_classes="extension-tab"):
                extension.ui()


def apply_extensions(typ: str, *args, **kwargs) -> t.Any:
    if typ not in EXTENSION_MAP:
        raise ValueError(f
