import base64
import json
import os
import time
import requests
import numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional, Union
from sentence_transformers import SentenceTransformer


def is_base64(string: str) -> bool:
    """Check if a string is a base64 encoded string."""
    try:
        base64.b64decode(string, validate=True)
    except (TypeError, binascii.Error):
        return False
    return True


def get_current_time_ms() -> dict:
    """Get the current time in milliseconds."""
    return {'time_ms': int(time.time() * 1000)}


class Handler(BaseHTTPRequestHandler):
    # ... (rest of the code)

    def do_GET(self):
        if self.path.startswith('/v1/models'):
            # ... (rest of the code)

        elif '/billing/usage' in self.path:
            # ... (rest of the code)

        else:
            self.send_error(404)

    def do_POST(self):
        if debug:
            print(self.path)  # did you know... python-openai?
        content_length = len(self.rfile.read(self.headers['Content-Length']))
        body = self.rfile.read(content_length).decode('utf-8')
        body_dict = self.parse_json(body)

        if debug:
            print(body_dict)

        if '/completions' in self.path or '/generate' in self.path:
            # ... (rest of the code)

        elif '/edits' in self.path:
            # ... (rest of the code)

        elif '/images/generations' in self.path and 'SDs_WEBUI_URL' in os.environ:
            # ... (rest of the code)

        elif '/embeddings' in self.path:
            # ... (rest of the code)

        elif '/moderations' in self.path:
            # ... (rest of the code)

        elif self.path == '/api/v1/token-count':
            # ... (rest of the code)

        else:
            print(self.path, self.headers)
            self.send_error(404)

    def parse_json(self, data: str) -> Union[Dict, List]:
        """Parse JSON data."""
        try:
            return json.loads(data)
        except ValueError:
            self.send_error(400, 'JSON parse error')


def run_server():
    global embedding_model
    try:
        embedding_model = SentenceTransformer(os.environ["OPENEDAI_EMBEDDING_MODEL"])
        print(f"\nLoaded embedding model: {os.environ['OPENEDAI_EMBEDDING_MODEL']}, max sequence length: {embedding_model.max_seq_length}")
    except KeyError:
        print("\nError: OPENEDAI_EMBEDDING_MODEL environment variable not set")
        embedding_model = None
    except Exception as e:
        print(f"\nFailed to load embedding model: {e}")
        embedding_model = None

    server_addr = ('0.0.0.0' if shared.args.listen else '127.0.0.1', params['port'])
    server = ThreadingHTTPServer(server_addr, Handler)
    if shared.args.share:
        try:
            from flask_cloudflared import _run_cloudflared
            public_url = _run_cloudflared(params['port'], params['port'] + 1)
            print(f'Starting OpenAI compatible api: OPENAI_API_BASE={public_url}/v1')
        except ImportError:
            print('You should install flask_cloudflared manually')
    else:
        print(f'Starting OpenAI compatible api: OPENAI_API_BASE=http://{server_addr}/v1')
        
    server.serve_forever()


def setup():
    Thread(target=run_server, daemon=True).start()


if __name__ == '__main__':
    setup()
