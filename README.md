## Overview

**Vir2vec** is a pan-viral genomic language model (422M parameters) obtained via **continual pretraining of Mistral-DNA** on a curated corpus of **565,747 complete viral genomes** spanning **295 species**. Vir2vec produces reusable **4,096-dimensional genome-level embeddings**.

The accompanying paper also introduces **vGUE**, a unified benchmark for viral *genome understanding* across tasks ranging from **virus vs. non-virus** and **DNA vs. RNA** to **host prediction**, **HIV-1 vs. HIV-2**, **SARS-CoV-2 lineage / influenza subtype typing**, and **HIV tropism**. Results show strong performance—especially on genome-wide and host-related tasks—using **frozen embeddings + shallow classifiers** under **nested cross-validation**.

## Repository contents

This repository contains the code and supporting files used to:
1. train the model(s)
2. generate embeddings from viral sequences/genomes (and run example use-cases)
3. track the accession numbers used to build the training/validation/test splits

---

## Folder structure

### `codes/training/`
Scripts required for **model training** (continual pretraining).  
Training can be launched via `accelerate`:

```bash
accelerate launch Minstral_Embedding_422M.py --config_file=default_config.yaml --num_process=4 --main_process_port 0
accelerate launch Minstral_Embedding_138M.py --config_file=default_config.yaml --num_process=4
accelerate launch Minstral_Embedding_17M.py  --config_file=default_config.yaml --num_process=4
```

### 'embedding'
Scripts to compute embeddings from sequences and to simulate example downstream use-cases.
Example:
```bash
python Local_Max.py
```

### accession_txt/
.txt files containing the accession numbers used to assemble the dataset splits for training and evaluation.
Examples:
train_accessions.txt
val_accessions.txt
test_accessions.txt

