import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any, Dict, List, Optional

from extensions.api.util import build_parameters, try_start_cloudflared
from modules import shared
from modules.chat import generate_chat_reply
from modules.text_generation import encode, generate_reply, stop_everything_event


class ImprovedHandler(BaseHTTPRequestHandler):
    def send_json_response(self, status_code: int, data: Dict[str, Any]) -> None:
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        response = json.dumps(data)
        self.wfile.write(response.encode('utf-8'))

    def do_GET(self) -> None:
        if self.path == '/api/v1/model':
            self.send_json_response(2
