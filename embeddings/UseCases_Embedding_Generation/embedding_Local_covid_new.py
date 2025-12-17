#############################
# Library Imports and Setup
#############################
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import numpy as np
import os
import gc
import time
from torch.cuda.amp import autocast
import h5py
import sys
from torch.cuda import empty_cache

#############################
# Configuration Parameters
#############################
CSV_PATH = '/blue/simone.marini/share/Code_Mistral/covid/unique/omicron/newseqs.csv'
OUTPUT_DIR = '/blue/salemi/share/varcovid/ViralLingo/Embedding/UseCases/Covid_Local/embeddings_results_covid_new'
CHUNK_SIZE = 50
BATCH_SIZE = 4
MAX_SEQUENCE_LENGTH = 4096
GRADIENT_CHECKPOINTING = True
USE_FP16 = False
POOLING_TYPE = 'max'  # *** Fixed to MAX pooling in this script ***

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.8'

#############################
# Model Definitions
#############################
MODELS = [
    {
        "name": "Local_Model",  # Custom locally trained model
        "model_path": "/blue/simone.marini/share/Code_Mistral/Minstral_422M/results/models/checkpoint-24752",
        "model_class": AutoModelForCausalLM,  # Uses a model class for causal generation
        "tokenizer_class": AutoTokenizer,
        "trust_remote_code": False,  # Do not trust remote code execution by default
        "padding_side": "left"  # Sequences will be padded on the left side
    }
]

#############################
# Memory Management Helper
#############################

def clear_memory():
    if torch.cuda.is_available():
        empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

#############################
# Batched Embedding Extraction (MAX pooling)
#############################

def get_embeddings_batched(model, tokenizer, sequence, Variant, batch_size=BATCH_SIZE):
    results = []
    total_batches = (len(sequence) + batch_size - 1) // batch_size

    for i in range(0, len(sequence), batch_size):
        print_first_batch = (i == 0)
        batch_sequence = sequence[i: i + batch_size]
        batch_Variant = Variant[i: i + batch_size]

        try:
            inputs = tokenizer(
                batch_sequence,
                padding=True,
                truncation=True,
                max_length=MAX_SEQUENCE_LENGTH,
                return_tensors='pt',
                return_overflowing_tokens=True,
                stride=512
            )

            for j in range(len(batch_sequence)):
                seq_embs = []

                if 'overflow_to_sample_mapping' in inputs:
                    seq_indices = [idx for idx, sample_idx in enumerate(inputs.overflow_to_sample_mapping) if sample_idx == j]
                else:
                    seq_indices = [j]

                for idx in seq_indices:
                    chunk_inputs = {
                        k: v[idx: idx + 1].to(model.device) if isinstance(v, torch.Tensor) else v
                        for k, v in inputs.items() if k != 'overflow_to_sample_mapping'
                    }
                    with torch.no_grad():
                        with autocast(enabled=USE_FP16):
                            output = model(**chunk_inputs)[0]
                            chunk_emb = torch.max(output, dim=1)[0]  # MAX pooling across tokens
                            seq_embs.append(chunk_emb.cpu())

                final_emb = torch.mean(torch.cat(seq_embs, dim=0), dim=0)
                numpy_emb = final_emb.numpy()


                results.append({
                    "Variant": batch_Variant[j],
                    "Embedding": numpy_emb
                })

                del seq_embs, final_emb

            del inputs
            clear_memory()

        except Exception as e:
            print(f"Error processing batch {i // batch_size + 1}/{total_batches}: {e}")
            continue

        print(f"Processed batch {i // batch_size + 1}/{total_batches}")

    return results

#############################
# Save Embeddings Utility
#############################

def save_embeddings(embeddings, labels, filename):
    with h5py.File(filename, 'w') as f:
        emb_arr = np.vstack(embeddings)
        f.create_dataset('embeddings', data=emb_arr)
        dt = h5py.special_dtype(vlen=str)
        lab_ds = f.create_dataset('labels', (len(labels),), dtype=dt)
        lab_ds[:] = labels

#############################
# Model Processing Pipeline
#############################

def process_embeddings(model_info):
    clear_memory()
    model_name = model_info['name']
    print(f"\n=== Processing {model_name} with {POOLING_TYPE.upper()} pooling ===")

    embeddings_file = os.path.join(
        OUTPUT_DIR, f"{model_name}_{POOLING_TYPE}_embeddings.h5")

    if os.path.exists(embeddings_file):
        print("Embeddings file already exists – skipping computation.")
        return embeddings_file

    start_time = time.time()

    model = model_info['model_class'].from_pretrained(
        model_info['model_path'],
        trust_remote_code=model_info.get('trust_remote_code', False),
        torch_dtype=torch.float32,
        device_map={"": 0}
    )
    tokenizer = model_info['tokenizer_class'].from_pretrained(
        model_info['model_path'],
        trust_remote_code=model_info.get('trust_remote_code', False)
    )
    if 'padding_side' in model_info:
        tokenizer.padding_side = model_info['padding_side']

    if hasattr(model, 'gradient_checkpointing_enable') and GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
    if hasattr(model, 'config'):
        model.config.use_cache = False
    model.eval()

    all_embeddings, all_Variant = [], []
    total_rows = sum(1 for _ in open(CSV_PATH)) - 1
    rows_processed = 0

    for chunk_idx, chunk in enumerate(pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE)):
        if 'sequence' not in chunk.columns or 'Variant' not in chunk.columns:
            raise ValueError("CSV must contain 'sequence' and 'Variant' columns.")

        sequence = chunk['sequence'].tolist()
        Variant = chunk['Variant'].tolist()
        print(f"Processing chunk {chunk_idx + 1} containing {len(sequence)} sequence …")

        embeddings = get_embeddings_batched(model, tokenizer, sequence, Variant)

        for emb in embeddings:
            all_embeddings.append(emb["Embedding"])
            all_Variant.append(emb["Variant"])
            rows_processed += 1                    # one sequence at a time
            if rows_processed % 100 == 0 or rows_processed == total_rows:
                progress = rows_processed / total_rows * 100
                print(f"Overall progress: {rows_processed}/{total_rows} rows ({progress:.2f}%)")
            
            # Save intermediate results every 5 chunks
            if (chunk_idx + 1) % 5 == 0:
                print("Saving intermediate results...")
                save_embeddings(all_embeddings, all_Variant, embeddings_file)

    save_embeddings(all_embeddings, all_Variant, embeddings_file)

    del model, tokenizer
    clear_memory()

    print(f"Finished {model_name} (max pooling) in {time.time() - start_time:.1f} s")
    return embeddings_file

#############################
# Main Entrypoint
#############################

def main():
    print("Starting DNA embedding extraction – MAX pooling mode")
    print(f"CSV: {CSV_PATH}\nOutput dir: {OUTPUT_DIR}")

    for model_info in MODELS:
        try:
            process_embeddings(model_info)
        except Exception as e:
            print(f"Failed to process {model_info['name']}: {e}")
        finally:
            clear_memory()

    print("All processing complete.")

if __name__ == '__main__':
    main()
