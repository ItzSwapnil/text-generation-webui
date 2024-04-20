import os
import re
from pathlib import Path
from functools import lru_cache


@lru_cache()
def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def atoi(text):
    return int(text) if text.isdigit() else text.lower()


def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)

    return text


def get_available_models(model_dir: str):
    if shared.args.flexgen:
        return sorted(
            (item.name for item in Path(model_dir).glob("*") if item.name.endswith("np")),
            key=natural_keys,
        )
    else:
        return sorted(
            (item.name for item in Path(model_dir).glob("*") if item.name not in (".txt", "np", ".pt", ".json", ".yaml")),
            key=natural_keys,
        )


def get_available_presets():
    return sorted(set(Path("presets").glob("*.yaml")).difference(set(("",))), key=natural_keys)


def get_available_prompts():
    prompts = []
    files = Path("prompts").glob("*.txt")
    prompts += sorted(
        [k for k in files if re.match(r"^[0-9]", k.stem)],
        key=natural_keys,
        reverse=True,
    )
    prompts += sorted(
        [k for k in files if re.match(r"^[^0-9]", k.stem)],
        key=natural_keys,
    )
    prompts += [f"Instruct-{k}" for k in get_available_instruction_templates() if k != "None"]
    prompts += ["None"]
    return prompts


def get_available_characters():
    paths = Path("characters").glob("*.json") | Path("characters").glob("*.yaml") | Path("characters").glob("*.yml")
    return ["None"] + sorted(set(k.stem for k in paths if k.stem != "instruction-following"), key=natural_keys)


def get_available_instruction_templates():
    path = Path("characters", "instruction-following")
    paths = path.glob("*.json") | path.glob("*.yaml") | path.glob("*.yml")
    return ["None"] + sorted(set(k.stem for k in paths), key=natural_keys)


def get_available_extensions():
    return sorted(set(k.parts[1] for k in Path("extensions").glob("*/script.py")), key=natural_keys)


def get_available_softprompts():
    return ["None"] + sorted(set(k.stem for k in Path("softprompts").glob("*.zip")), key=natural_keys)


def get_available_loras(lora_dir: str):
    return sorted(
        (item.name for item in Path(lora_dir).glob("*") if not item.name.endswith((".txt", "-np", ".pt", ".json"))),
        key=natural_keys,
    )


def get_datasets(path: str, ext: str):
    return ["None"] + sorted(
        set(k.stem for k in Path(path).glob(f"*.{ext}") if k.stem != "put-trainer-datasets-here"),
        key=natural_keys,
    )


def get_available_chat_styles():
    return sorted(
        set(k
