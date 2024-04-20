"""
Downloads models from Hugging Face to models/model-name.

Example:
python download-model.py facebook/opt-1.3b
"""

import argparse
import base64
import datetime
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import requests
from tqdm import tqdm


def select_model_from_default_options():
    models = {
        "OPT 6.7B": ("facebook", "opt-6.7b", "main"),
        "OPT 2.7B": ("facebook", "opt-2.7b", "main"),
        "OPT 1.3B": ("facebook", "opt-1.3b", "main"),
        "OPT 350M": ("facebook", "opt-350m", "main"),
        "GALACTICA 6.7B": ("facebook", "galactica-6.7b", "main"),
        "GALACTICA 1.3B": ("facebook", "galactica-1.3b", "main"),
        "GALACTICA 125M": ("facebook", "galactica-125m", "main"),
        "Pythia-6.9B-deduped": ("EleutherAI", "pythia-6.9b-deduped", "main"),
        "Pythia-2.8B-deduped": ("EleutherAI", "pythia-2.8b-deduped", "main"),
        "Pythia-1.4B-deduped": ("EleutherAI", "pythia-1.4b-deduped", "main"),
        "Pythia-410M-deduped": ("EleutherAI", "pythia-410m-deduped", "main"),
    }

    choices = {}
    print("Select the model that you want to download:\n")
    for i, name in enumerate(models):
        char = chr(ord('A') + i)
        choices[char] = name
        print(f"{char}) {name}")

    char_hugging = chr(ord('A') + len(models))
    print(f"{char_hugging}) Manually specify a Hugging Face model")
    char_exit = chr(ord('A') + len(models) + 1)
    print(f"{char_exit}) Do not download a model")
    print()
    print("Input> ", end='')
    choice = input()[0].strip().upper()
    if choice == char_exit:
        exit()
    elif choice == char_hugging:
        print("""\nType the name of your desired Hugging Face model in the format organization/name.

Examples:
facebook/opt-1.3b
EleutherAI/pythia-1.4b-deduped
""")

        print("Input> ", end='')
        model = input()
        branch = "main"
    else:
        arr = models[choices[choice]]
        model = f"{arr[0]}/{arr[1]}"
        branch = arr[2]

    return model, branch


class ModelDownloader:
    def __init__(self):
        self.s = requests.Session()
        if os.getenv('HF_USER') is not None and os.getenv('HF_PASS') is not None:
            self.s.auth = (os.getenv('HF_USER'), os.getenv('HF_PASS'))

    def sanitize_model_and_branch_names(self, model, branch):
        if model[-1] == '/':
            model = model[:-1]

        if branch is None:
            branch = "main"
        else:
            pattern = re.compile(r"^[a-zA-Z0-9._-]+$")
            if not pattern.match(branch):
                raise ValueError(
                    "Invalid branch name. Only alphanumeric characters, period, underscore and dash are allowed.")

        return model, branch

    def get_download_links_from_huggingface(self, model, branch, text_only=False):
        base = "https://huggingface.co"
        page = f"/api/models/{model}/tree/{branch}"
        cursor = b""

        links = []
        sha256 = []
        classifications = []
        has_pytorch = False
        has_pt = False
        has_ggml = False
        has_safetensors = False
        is_lora = False
        while True:
            url = f"{base}{page}" + (f"?cursor={cursor.decode()}" if cursor else "")
            r = self.s.get(url, timeout=10)
            r.raise_for_status()
            content = r.content

            dict = json.loads(content)
            if len(dict) == 0:
                break

            for i in range(len(dict)):
                fname = dict[i]['path']
                if not is_lora and fname.endswith(('adapter_config.json', 'adapter_model.bin')):
                    is_lora = True

                is_pytorch = re.match("(pytorch|adapter)_model.*\.bin", fname)
                is_safetensors = re.match(".*\.safetensors", fname)
                is_pt = re.match(".*\.pt", fname)
                is_ggml = re.match(".*ggml.*\.bin", fname)
                is_tokenizer = re.match("(tokenizer|ice).*\.model", fname)
                is_text = re.match(".*\.(txt|json|py|md)", fname) or is_tokenizer
                if any((is_pytorch, is_safetensors, is_pt, is_ggml, is_tokenizer, is_text)):
                    if '
