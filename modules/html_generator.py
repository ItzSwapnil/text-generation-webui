"""
This is a library for formatting text outputs as nice HTML.
"""

import os
import re
import time
from pathlib import Path
from typing import List, Tuple

import markdown
from PIL import Image, ImageOps

from modules.utils import get_available_chat_styles

IMAGE_CACHE_DIR = "cache"

def fix_newlines(string: str) -> str:
    string = string.replace("\n", "\n\n")
    string = re.sub(r"\n{3,}", "\n\n", string)
    string = string.strip()
    return string

def replace_blockquote(m) -> str:
    return m.group().replace("\n", "\n> ").replace("\\begin{blockquote}", "").replace("\\end{blockquote}", "")

def convert_to_markdown(string: str) -> str:
    # Blockquote
    pattern = re.compile(r"\\begin{blockquote}(.*?)\\end{blockquote}", re.DOTALL)
    string = pattern.sub(replace_blockquote, string)

    # Code
    string = string.replace("\\begin{code}", "```")
    string = string.replace("\\end{code}", "```")
    string = re.sub(r"(.)```", r"\1\n```", string)

    result = ""
    is_code = False
    for line in string.split("\n"):
        if line.lstrip(" ").startswith("```"):
            is_code = not is_code

        result += line
        if is_code or line.startswith("|"):  # Don't add an extra \n for tables or code
            result += "\n"
        else:
            result += "\n\n"

    if is_code:
        result = result + "```"  # Unfinished code block

    string = result.strip()
    return markdown.markdown(string, extensions=["fenced_code", "tables"])

def generate_basic_html(string: str) -> str:
    string = convert_to_markdown(string)
    string = f'<style>{readable_css}</style><div class="container">{string}</div>'
    return string

def process_post(post: str, c: int) -> str:
    t = post.split("\n")
    number = t[0].split(" ")[1]
    if len(t) > 1:
        src = "\n".join(t[1:])
    else:
        src = ""
    src = re.sub(">", "&gt;", src)
    src = re.sub("(&gt;&gt;[0-9]*)", "<span class=\"quote\">\\1</span>", src)
    src = re.sub("\n", "<br>\n", src)
    src = f'<blockquote class="message">{src}\n'
    src = f'<span class="name">Anonymous </span> <span class="number">No.{number}</span>\n{src}'
    return src

def generate_4chan_html(f: str) -> str:
    posts = []
    post = ""
    c = -2
    for line in f.splitlines():
        line += "\n"
        if line == "-----\n":
            continue
        elif line.startswith("--- "):
            c += 1
            if post != "":
                src = process_post(post, c)
                posts.append(src)
            post = line
        else:
            post += line
    if post != "":
        src = process_post(post, c)
        posts.append(src)

    for i in range(len(posts)):
        if i == 0:
            posts[i] = f'<div class="op">{posts[i]}</div>\n'
        else:
            posts[i] = f'<div class="reply">{posts[i]}</div>\n'

    output = ""
    output += f'<style>{_4chan_css}</style><div id="parent"><div id="container">'
    for post in posts:
        output += post
    output += '</div></div>'
    output = output.split("\n")
    for i in range(len(output)):
        output[i] = re.sub(r'^(&gt;(.*?)(<br>|</div>))', r'<span class="greentext">\1</span>', output[i])
        output[i] = re.sub(r'^<blockquote class="message">(&gt;(.*?)(<br>|</div>))', r'<blockquote class="message"><span class="greentext">\1</span>', output[i])
    output = "\n".join(output)

    return output

def make_thumbnail(image: Image.Image) -> Image.Image:
    image = image.resize(
        (350, round(image.size[1] / image.size[0] * 350)),
        Image.Resampling.LANCZOS
    )
    if image.size[1] > 470:
        image = ImageOps.fit(image, (350, 470), Image.ANTIALIAS)

    return image

def get_image_cache(path: str) -> str:
    cache_folder = Path(IMAGE_CACHE_DIR)
    if not cache_folder.exists():
        cache_folder.mkdir()

    mtime = os.stat(path).st_mtime
    cache_path = Path(f'{IMAGE_CACHE_DIR}/{path.name}_cache.png')

    if (path in image_cache and mtime != image_cache[path][0]) or (path not in image_cache):
        img = make_thumbnail(Image.open(path))
        img.convert('RGB').save(cache_path, format='PNG')
        image_cache[path] = [mtime, cache_path.as_posix()]

    return image_cache[path][1]

def generate_instruct
