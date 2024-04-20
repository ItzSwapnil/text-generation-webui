import datetime
import os
import traceback
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm

import modules.shared as shared
from modules.models import load_model, unload_model
from modules.text_generation import encode
from server import get_model_specific_settings, update_model_parameters

def load_past_evaluations():
    csv_file = 'logs/evaluations.csv'
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file, dtype=str)
        df['Perplexity'] = pd.to_numeric(df['Perplexity'])
        return df
    else:
        return pd.DataFrame(columns=['Model', 'LoRAs', 'Dataset', 'Perplexity', 'stride', 'max_length', 'Date', 'Comment'])


past_evaluations = load_past_evaluations()


def save_past_evaluations(df):
    global past_evaluations
    past_evaluations = df
    filepath = Path('logs/evaluations.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


def calculate_perplexity(models, input_dataset, stride, max_length):
    '''
    Based on:
    https://huggingface.co/docs/transformers/perplexity#calculating-ppl-with-fixedlength-models
    '''

    cumulative_log = ''
    cumulative_log += "Loading the input dataset...\n\n"
    yield cumulative_log

    def load_input_data(dataset):
        if dataset == 'wikitext':
            data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
            text = "\n\n".join(data['text'])
        elif dataset == 'ptb':
            data = load_dataset('ptb_text_only', 'penn_treebank', split='validation')
            text = "\n\n".join(data['sentence'])
        elif dataset == 'ptb_new':
            data = load_dataset('ptb_text_only', 'penn_treebank', split='test')
            text = " ".join(data['sentence'])
        else:
            with open(Path(f'training/datasets/{dataset}.txt'), 'r', encoding='utf-8') as f:
                text = f.read()
        return text

    for model in models:
        if is_in_past_evaluations(model, input_dataset, stride, max_length):
            cumulative_log += f"{model} has already been tested. Ignoring.\n\n"
            yield cumulative_log
            continue

        if model != 'current model':
            try:
                yield cumulative_log + f"Loading {model}...\n\n"
                model_settings = get_model_specific_settings(model)
                shared.settings.update(model_settings)  # hijacking the interface defaults
                update_model_parameters(model_settings)  # hijacking the command-line arguments
                shared.model_name = model
                unload_model()
                shared.model, shared.tokenizer = load_model(shared.model_name)
            except:
                cumulative_log += f"Failed to load {model}. Moving on.\n\n"
                yield cumulative_log
                continue

        cumulative_log += f"Processing {shared.model_name}...\n\n"
        yield cumulative_log + "Tokenizing the input dataset...\n\n"
        text = load_input_data(input_dataset)
        encodings = encode(text, add_special_tokens=False)
        seq_len = encodings.shape[1]
        max_length = max_length if max_length else (
            hasattr(shared.model.config, 'max_position_embeddings')
            and shared.model.config.max_position_embeddings
            or 2048
        )

        nlls = []
        prev_end_loc = 0
        for begin_loc in tqdm(range(0, seq_len, stride)):
            yield cumulative_log + f"Evaluating... {100*begin_loc/seq_len:.2f}%"
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = shared.model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).mean())
        add_entry_to_past_evaluations(float(ppl), shared.model_name, input_dataset, stride, max_length)
        save_past_evaluations(past_evaluations)
        cumulative_log += f"The perplexity for {shared.model_name} is: {float(ppl)}\n\n"
        yield cumulative_log


def add_entry_to_past_evaluations(perplexity, model, dataset, stride, max_length):
    global past_evaluations
    entry = {
        '
