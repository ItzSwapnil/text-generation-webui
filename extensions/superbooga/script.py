import re
import textwrap
from typing import Any, Dict, List, Tuple

import gradio as gr
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse

import modules
from modules import chat, shared
from modules.logging_colors import logger
from modules.shared import OPTIONS
from modules.utils import download_file

from .chromadb import add_chunks_to_collector, make_collector
from .download_urls import download_urls

params = {
    'chunk_count': 5,
    'chunk_count_initial': 10,
    'time_weight': 0,
    'chunk_length': 700,
    'chunk_separator': '',
    'strong_cleanup': False,
    'threads': 4,
}

collector = make_collector()
chat_collector = make_collector()


def feed_data_into_collector(corpus: str, chunk_len: int, chunk_sep: str) -> List[str]:
    # Defining variables
    chunk_len = int(chunk_len)
    chunk_sep = chunk_sep.replace(r'\n', '\n')
    cumulative = ''

    # Breaking the data into chunks and adding those to the db
    cumulative += "Breaking the input dataset...\n\n"
    yield cumulative
    if chunk_sep:
        data_chunks = corpus.split(chunk_sep)
        data_chunks = [[data_chunk[i:i + chunk_len] for i in range(0, len(data_chunk), chunk_len)] for data_chunk in data_chunks]
        data_chunks = [x for y in data_chunks for x in y]
    else:
        data_chunks = [corpus[i:i + chunk_len] for i in range(0, len(corpus), chunk_len)]

    cumulative += f"{len(data_chunks)} chunks have been found.\n\nAdding the chunks to the database...\n\n"
    yield cumulative
    add_chunks_to_collector(data_chunks, collector)
    cumulative += "Done."
    yield cumulative


def feed_file_into_collector(file: bytes, chunk_len: int, chunk_sep: str) -> List[str]:
    try:
        text = file.decode('utf-8')
    except UnicodeDecodeError:
        return ['Error: Could not decode the file as UTF-8.']

    return list(feed_data_into_collector(text, chunk_len, chunk_sep))


def feed_url_into_collector(urls: str, chunk_len: int, chunk_sep: str, strong_cleanup: bool, threads: int) -> List[str]:
    urls = urls.strip().split('\n')
    cumulative = ''

    cumulative += f'Loading {len(urls)} URLs with {threads} threads...\n\n'
    yield cumulative
    for update, contents in download_urls(urls, threads=threads):
        cumulative += update
        yield cumulative

    cumulative += 'Processing the HTML sources...'
    yield cumulative
    for content in contents:
        soup = BeautifulSoup(content, features="html.parser")
        for script in soup(["script", "style"]):
            script.extract()

        strings = soup.stripped_strings
        if strong_cleanup:
            strings = [s for s in strings if re.search("[A-Za-z] ", s)]

        text = '\n'.join([s.strip() for s in strings])
        try:
            yield from feed_data_into_collector(text, chunk_len, chunk_sep)
        except Exception as e:
            yield [f'Error: Could not process the HTML source: {e}']


def apply_settings(chunk_count: int, chunk_count_initial: int, time_weight: float) -> str:
    global params
    params['chunk_count'] = int(chunk_count)
    params['chunk_count_initial'] = int(chunk_count_initial)
    params['time_weight'] = time_weight
    settings_to_display = {k: params[k] for k in params if k in ['chunk_count', 'chunk_count_initial', 'time_weight']}
    return f"The following settings are now active: {str(settings_to_display)}"


def custom_generate_chat_prompt(
    user_input: str,
    state: Dict[str, Any],
    **kwargs
) -> Tuple[str, Dict[str, Any]]:
    global chat_collector

    if state['mode'] == 'instruct':
        try:
            results = collector.get_sorted(user_input, n_results=params['chunk_count'])
        except Exception as e:
            return chat.generate_chat_prompt(f'Error: Could not query the database: {e}', state, **kwargs), kwargs

        additional_context = '\nYour reply should be based on the context below:\n\n' + '\n'.join(results)
        user_input += additional_context
    else:

        def make_single_exchange(id
