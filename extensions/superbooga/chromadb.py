import chromadb
import posthog
import torch
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Union, Optional

# Disable Posthog telemetry
posthog.capture = lambda *args, **kwargs: None

