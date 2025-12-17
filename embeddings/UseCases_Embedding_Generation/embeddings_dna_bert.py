"""Extract **max-pooled** embeddings with RaphaelMourad/ModernBert-DNA-v1-37M-virus,
   writing them to HDF5 *incrementally* to keep RAM usage flat."""

from pathlib import Path
import time, gc, math
import torch, h5py, pandas as pd, numpy as np
from torch.cuda import empty_cache
from torch.cuda.amp import autocast
from transformers import AutoModel, AutoTokenizer

CSV_PATH   = '/blue/simone.marini/share/rna_dna/dna.csv'
OUTPUT_DIR = Path('embeddings_results_modern_dna'); OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_ID   = 'RaphaelMourad/ModernBert-DNA-v1-37M-virus'
FILE_OUT   = OUTPUT_DIR / 'ModernBert_max_embeddings.h5'
BATCH_SIZE = 16
CHUNK_SIZE = 200
MAX_LEN    = 4096
USE_FP16   = False


def clear_mem():
    if torch.cuda.is_available():
        empty_cache(); torch.cuda.reset_peak_memory_stats()
    gc.collect()

def embed_batch(model, tok, seqs):
    inputs = tok(seqs, padding=True, truncation=True, max_length=MAX_LEN,
                 return_tensors='pt', return_overflowing_tokens=True, stride=512)
    mapping = inputs.pop('overflow_to_sample_mapping', None)
    ids  = inputs['input_ids'].to(model.device)
    mask = inputs['attention_mask'].to(model.device)
    embs=[]
    for j in range(len(seqs)):
        idxs = [i for i,m in enumerate(mapping) if m==j] if mapping is not None else [j]
        toks=[]
        for idx in idxs:
            with torch.no_grad():
                with autocast(enabled=USE_FP16):
                    out = model(input_ids=ids[idx:idx+1],
                                attention_mask=mask[idx:idx+1])
                    toks.append(torch.max(out.last_hidden_state, dim=1)[0].cpu())
        embs.append(torch.mean(torch.cat(toks), dim=0).numpy())
    clear_mem(); return embs


def main():
    if FILE_OUT.exists():
        print('Embeddings already exist →', FILE_OUT); return

    print('Loading model…')
    model = AutoModel.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if USE_FP16 else torch.float32,
        device_map={'': 0}
    ).eval()
    tok   = AutoTokenizer.from_pretrained(MODEL_ID)

    # ------------------------------------------------------------------
    # NEW >>>  create the HDF5 file and datasets *before* the loop
    # ------------------------------------------------------------------
    total_rows = sum(1 for _ in open(CSV_PATH)) - 1      # header excluded
    h5f   = h5py.File(FILE_OUT, 'w')
    emb_ds = h5f.create_dataset(
        'embeddings',
        shape=(total_rows, model.config.hidden_size),
        dtype='float16' if USE_FP16 else 'float32'
    )
    lbl_ds = h5f.create_dataset(
        'labels',
        shape=(total_rows,),
        dtype=h5py.string_dtype()
    )
    row_ptr = 0
    # ------------------------------------------------------------------
    # NEW <<<  (we’ll write into emb_ds / lbl_ds as we go)
    # ------------------------------------------------------------------

    for cidx, chunk in enumerate(pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE)):
        seqs = chunk['Sequence'].tolist()
        labs = chunk['accession'].tolist()
        batches = math.ceil(len(seqs) / BATCH_SIZE)
        print(f'Chunk {cidx + 1}: {len(seqs)} seqs, {batches} batches')

        for b in range(batches):
            batch_seqs = seqs[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            embs = embed_batch(model, tok, batch_seqs)

            # ----------------------------------------------------------
            # NEW >>>  immediately flush the current batch to disk
            # ----------------------------------------------------------
            emb_mat = np.vstack(embs)                                # (batch, 768)
            emb_ds[row_ptr:row_ptr + len(emb_mat)] = emb_mat
            lbl_ds[row_ptr:row_ptr + len(emb_mat)] = \
                 labs[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            row_ptr += len(emb_mat)
            # ----------------------------------------------------------
            # NEW <<<  memory footprint stays roughly constant
            # ----------------------------------------------------------

            pct = row_ptr / total_rows * 100
            print(f'  • processed batch {b + 1}/{batches} | overall {pct:.1f}%')

    h5f.close()                    # NEW >>> ensures data are flushed
    print('Saved', FILE_OUT)       # NEW <<<

if __name__ == '__main__':
    main()
