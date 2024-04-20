#!/usr/bin/env python3

import os
import sentence_transformers

# Preload the embedding model, useful for Docker images to prevent re-download on config change
# Set the environment variable OPENEDAI_EMBEDDING_MODEL to the desired model name before running the script
# or set it to 'all-mpnet-base-v2' as the default model

# Check if the environment variable is set, if not use the default model
embedding_model = os.getenv('OPENEDAI_EMBEDDING_MODEL', 'all-mpnet-base-v2')

# Load the model
try:
    model = sentence_transformers.SentenceTransformer(embedding_model)
except Exception as e:
    print(f"Error loading model {embedding_model}: {e}")
    raise
