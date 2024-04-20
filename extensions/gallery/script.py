from pathlib import Path
import gradio as gr
from modules.html_generator import get_image_cache
from modules.shared import gradio

def generate_css():
    css = """
      .character-gallery > .gallery {
        margin: 1rem 0;
        display: grid !important;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        grid-column-gap: 0.4rem;
        grid-row-gap: 1.2rem;
      }

      .character-gallery > .label {
        display: none !important;

