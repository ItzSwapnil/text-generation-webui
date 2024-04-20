import time
import traceback
import argparse
from threading import Thread
from typing import Dict, Optional, Callable

import shared
from modules.chat import load_character_memoized
try:
    from flask_cloudflared import _run_cloudflared
except ImportError:
    print('You should install flask_cloudflared manually')
    raise Exception(
        'flask_cloudflared not installed. Make sure you installed the requirements.txt for this extension.')

def build_parameters(body: Dict, chat: bool = False) -> Dict:

    generate_params = {
        'max_new_tokens': body.get('max_new_tokens', 200),
        'do_sample': body.get('do_sample', True),
        'temperature': body.get('temperature', 0.5),
        'top_p': body.get('top_p', 1),
        'typical_p': body.get('typical_p', body.get('typical', 1)),
        'epsilon_cutoff': body.get('epsilon_cutoff', 0),
        'eta_cutoff': body.get('eta_cutoff', 0),
        'tfs': body.get('tfs', 1),
        'top_a': body.get('top_a', 0),
        'repetition_penalty': body.get('repetition_penalty', body.get('rep_pen', 1.1)),
        'encoder_repetition_penalty': body.get('encoder_repetition_penalty', 1.0),
        'top_k': body.get('top_k', 0),
        'min_length': body.get('min_length', 0),
        'no_repeat_ngram_size': body.get('no_repeat_ngram_size', 0),
        'num_beams': body.get('num_beams', 1),
        'penalty_alpha': body.get('penalty_alpha', 0),
        'length_penalty': body.get('length_penalty', 1),
        'early_stopping': body.get('early_stopping', False),
        'mirostat_mode': body.get('mirostat_mode', 0),
        'mirostat_tau': body.get('mirostat_tau', 5),
        'mirostat_eta': body.get('mirostat_eta', 0.1),
        'seed': body.get('seed', -1),
        'add_bos_token': body.get('add_bos_token', True),
        'truncation_length': body.get('truncation_length', body.get('max_context_length', 2048)),
        'ban_eos_token': body.get('ban_eos_token', False),
        'skip_special_tokens': body.get('skip_special_tokens', True),
        'custom_stopping_strings': '',  # leave this blank
        'stopping_strings': body.get('stopping_strings', []),
    }

    if chat:
        character = body.get('character')
        instruction_template = body.get('instruction_template')
        name1, name2, _, greeting, context, _ = load_character_memoized(character, str(body.get('your_name', shared.settings['name1'])), shared.settings['name2'], instruct=False)
        name1_instruct, name2_instruct, _, _, context_instruct, turn_template = load_character_memoized(instruction_template, '', '', instruct=True)
        generate_params.update({
            'stop_at_newline': body.get('stop_at_newline', shared.settings['stop_at_newline']),
            'chat_prompt_size': body.get('chat_prompt_size', shared.settings['chat_prompt_size']),
            'chat_generation_attempts': body.get('chat_generation_attempts', shared.settings['chat_generation_attempts']),
            'mode': body.get('mode', 'chat'),
            'name1': name1,
            'name2': name2,
            'context': context,
            'greeting': greeting,
            'name1_instruct': name1_instruct,
            'name2_instruct': name2_instruct,
            'context_instruct': context_instruct,
            'turn_template': turn_template,
            'chat-instruct_command': body.get('chat-instruct_command', shared.settings['chat-instruct_command']),
        })

    return generate_params

def try_start_cloudflared(port: int, max_attempts: int = 3, on_start: Optional[Callable[[str], None]] = None):
    """
    Try to start Cloudflared on the given port with the specified number of attempts.
    If the `on_start` callback is provided, it will be called with the public URL of the Cloudflared instance.
    """
    Thread(target=_start_cloudflared, args=[port, max_attempts, on_start], daemon=True).start()

def _start_cloudflared(port: int, max_attempts: int = 3, on_start: Optional[Callable[[str], None]] = None):
    for _ in range(max_attempts):
        try:
            public_url = _run
