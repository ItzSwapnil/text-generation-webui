import os
import re
from pathlib import Path
import traceback

import gradio as gr
import elevenlabs  # type: ignore
from modules import chat, shared  # type: ignore

params = {
    "activate": True,
    "api_key": None,
    "selected_voice": "None",
    "autoplay": False,
    "show_text": True,
}

voices = None
wav_idx = 0


def update_api_key(key: str) -> None:
    params["api_key"] = key
    if key is not None and elevenlabs:
        elevenlabs.set_api_key(key)


def refresh_voices() -> list[str]:
    if not elevenlabs:
        return []
    return [voice.name for voice in elevenlabs.voices()]


def refresh_voices_dd() -> gr.Dropdown:
    all_voices = refresh_voices()
    return gr.Dropdown.update(value=all_voices[0], choices=all_voices) if all_voices else gr.Dropdown()


def remove_tts_from_history() -> None:
    if not shared:
        return
    for i, entry in enumerate(shared.history["internal"]):
        shared.history["visible"][i] = [shared.history["visible"][i][0], entry[1]]


def toggle_text_in_history() -> None:
    if not shared:
        return
    for i, entry in enumerate(shared.history["visible"]):
        visible_reply = entry[1]
        if visible_reply.startswith("<audio"):
            if params["show_text"]:
                reply = shared.history["internal"][i][1]
                shared.history["visible"][i] = [
                    shared.history["visible"][i][0], f"{visible_reply.split('</audio>')[0]}</audio>\n\n{reply}"
                ]
            else:
                shared.history["visible"][i] = [
                    shared.history["visible"][i][0], f"{visible_reply.split('</audio>')[0]}</audio>"
                ]


def remove_surrounded_chars(string: str) -> str:
    # this expression matches to 'as few symbols as possible (0 upwards) between any asterisks' OR
    # 'as few symbols as possible (0 upwards) between an asterisk and the end of the string'
    return re.sub("\*[^\*]*?(\*|$)", "", string)


def state_modifier(state: dict) -> dict:
    if not params["activate"]:
        return state

    state["stream"] = False
    return state


def input_modifier(string: str) -> str:
    if not params["activate"]:
        return string

    shared.processing_message = "*Is recording a voice message...*"
    return string


def history_modifier(history: dict) -> dict:
    # Remove autoplay from the last reply
    if len(history["internal"]) > 0:
        history["visible"][-1] = [
            history["visible"][-1][0],
            history["visible"][-1][1].replace('controls autoplay>', 'controls>')
        ]

    return history


def output_modifier(string: str) -> str:
    global params, wav_idx

    if not params["activate"]:
        return string

    original_string = string
    string = remove_surrounded_chars(string)
    string = string.replace('"', '')
    string = string.replace('“', '')
    string = string.replace('\n', ' ')
    string = string.strip()
    if string == '':
        string = 'empty reply, try regenerating'

    output_file = Path(f'extensions/elevenlabs_tts/outputs/{wav_idx:06d}.mp3'.format(wav_idx))
    print(f'Outputting audio to {str(output_file)}')
    try:
        audio = elevenlabs.generate(text=string, voice=params["selected_voice"], model="eleven_monolingual_v1")
        elevenlabs.save(audio, str(output_file))

        autoplay = 'autoplay' if params["autoplay"] else ''
        string = f'<audio src="file/{output_file.as_posix()}" controls {autoplay}></audio>'
        wav_idx += 1
    except elevenlabs.api.error.UnauthenticatedRateLimitError:
        string = "🤖 ElevenLabs Unauthenticated Rate Limit Reached - Please create an API key to continue\n\n"
    except elevenlabs.api.error.RateLimitError:
        string = "🤖 ElevenLabs API Tier Limit Reached\n\n"
    except elevenlabs.api.error.APIError as err:
        string = f"🤖 ElevenLabs Error: {err}\n\n"
    except Exception as e:
        traceback.print_exc()
        string = f"🤖 Unexpected Error: {str(e)}\n\n"

    if params["show_text"]:
        string += f'\n\n{original_string}'

    shared.processing_message = "*Is typing...*"
    return string


def ui() -> gr.Blocks:
    global voices
    if not voices:
        voices = refresh_voices()
        params["selected_
