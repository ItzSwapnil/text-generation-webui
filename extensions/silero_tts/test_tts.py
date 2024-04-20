import time
import sys
from pathlib import Path
import torch
import tts_preprocessor

torch._C._jit_set_profiling_mode(False)

PARAMS_DEFAULT = {
    'activate': True,
    'speaker': 'en_49',
    'language': 'en',
    'model_id': 'v3_en',
    'sample_rate': 48000,
    'device': 'cpu',
    'show_text': True,
    'autoplay': True,
    'voice_pitch': 'medium',
    'voice_speed': 'medium',
}

PARAMS_PITCHES = ['x-low', 'low', 'medium', 'high', 'x-high']
PARAMS_SPEEDS = ['x-slow', 'slow', 'medium', 'fast', 'x-fast']

def xmlesc(txt):
    return txt.translate(str.maketrans({
        "<": "&lt;",
        ">": "&gt;",
        "&": "&amp;",
        "'": "&apos;",
        '"': "&quot;",
    }))

def load_model():
    model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_tts', language=PARAMS_DEFAULT['language'], speaker=PARAMS_DEFAULT['model_id'])
    model.to(PARAMS_DEFAULT['device'])
    return model

def output_modifier(string):
    model = load_model()

    original_string = string
    string = tts_preprocessor.preprocess(string)
    processed_string = string

    if string == '':
        string = '*Empty reply, try regenerating*'
    else:
        output_file = Path(f'extensions/silero_tts/outputs/test_{int(time.time())}.wav')
        prosody = f'<prosody rate="{PARAMS_DEFAULT["voice_speed"]}" pitch="{PARAMS_DEFAULT["voice_pitch"]}">'
        silero_input = f'<speak>{prosody}{xmlesc(string)}</prosody></speak>'
        model.save_wav(ssml_text=silero_input, speaker=PARAMS_DEFAULT['speaker'], sample_rate=int(PARAMS_DEFAULT['sample_rate']), audio_path=str(output_file))

        autoplay = 'autoplay' if PARAMS_DEFAULT['autoplay'] else ''
        string = f'<audio src="file/{output_file.as_posix()}" controls {autoplay}></audio>'

        if PARAMS_DEFAULT['show_text']:
            string += f'\n\n{original_string}\n\nProcessed:\n{processed_string}'

    print(string)

if __name__ == '__main__':
    output_modifier(sys.argv[1])
