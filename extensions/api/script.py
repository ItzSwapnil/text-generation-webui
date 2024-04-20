import extensions.api.blocking_api as blocking_api
import extensions.api.streaming_api as streaming_api
from modules import shared

def start_api_servers():
    blocking_port = get_blocking_port()
    streaming_port = get_streaming_port()

    blocking_api.start_server(blocking_port, share=shared.args.public_api)
    streaming_api.start_server(streaming_port, share=shared.args.public_api)

def get_blocking_port():
    blocking_port = shared.args.api_blocking_port
    if not blocking_port:
        blocking_port = 8000  # Default port
    return blocking_port

