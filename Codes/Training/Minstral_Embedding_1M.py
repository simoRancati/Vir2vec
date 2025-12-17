# IMPORT LIBRARIES
# During the class, discuss the different python packages
import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32 "
os.environ["HF_DATASETS_CACHE"] = "/blue/salemi/share/varcovid/ViralLingo/Embedding/cache_directory"
import torch
import numpy as np
from transformers import AutoTokenizer
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments
from transformers import AutoModelForCausalLM, AutoConfig
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
from torch.optim import AdamW  # Importato da torch.optim
import transformers
transformers.__version__
import flash_attn
flash_attn.__version__
import wandb
torch.backends.cudnn.benchmark=True

cache_directory = "/blue/salemi/share/varcovid/ViralLingo/Embedding/cache_directory"
device = 'auto'

config = AutoConfig.from_pretrained("RaphaelMourad/Mistral-DNA-v1-1M-hg38", cache_dir=cache_directory) # Mixture of expert
model = AutoModelForCausalLM.from_config(config, attn_implementation="flash_attention_2", trust_remote_code=True)
#model = AutoModelForCausalLM.from_config(config,attn_implementation="eager")
model

# LOAD BPE LETTER TOKENIZER
# Question during class: what are UNK, CLS, SEP, PAD and MASK?
tokenizer = AutoTokenizer.from_pretrained("RaphaelMourad/Mistral-DNA-v1-1M-hg38", trust_remote_code=True, cache_dir=cache_directory)
tokenizer.padding_side  = 'left'
print(tokenizer)

#NUMBER OF MODEL PARAMETERS
pytorch_total_params = sum(p.numel() for p in model.parameters())
print(f"Model size: {pytorch_total_params/1000**2:.1f}M parameters")

#LOAD DATA
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
dataset_text = load_dataset(
    "csv",
    data_files={
        "train": "/blue/salemi/share/varcovid/ViralLingo/Embedding/train_data/train_dataset.csv",
        "validation": "/blue/salemi/share/varcovid/ViralLingo/Embedding/val_data/val_split.csv",
        "test": "/blue/salemi/share/varcovid/ViralLingo/Embedding/test_data/test_split.csv"
    }
)

columns_to_keep = ["Sequence"]

# Limit the dataset size before mapping
#dataset_text["train"] = dataset_text["train"].select(range(300))  # Primi 300 esempi per il train
#dataset_text["validation"] = dataset_text["validation"].select(range(100))  # Primi 100 esempi per la validazione
#dataset_text["test"] = dataset_text["test"].select(range(100))  # Primi 100 esempi per il test

for split in dataset_text:
    cols_to_remove = [col for col in dataset_text[split].column_names if col not in columns_to_keep]
    dataset_text[split] = dataset_text[split].remove_columns(cols_to_remove)

# TOKENIZE DATA
def tokenize_function(examples):
    return tokenizer(examples['Sequence'], padding="longest", truncation=True, return_tensors="pt")

dataset = dataset_text.map(tokenize_function, batched=True,  load_from_cache_file=False)

print(dataset["train"])

dataset_train = dataset["train"]
dataset_val = dataset["validation"]

#print(dataset_train[1])

# PARAMETERS FOR PRETRAINING
# During class, modify:
# - batchsize
# - number of epochs
# - learning rate
# - weight_decay
# - gradient accumulation steps

# batchsize=2
# training_args = TrainingArguments(
#         output_dir='./results/models',
#         evaluation_strategy='steps',
#         save_strategy='epoch',
#         num_train_epochs=50,
#         per_device_train_batch_size=batchsize,
#         per_device_eval_batch_size=batchsize,
#         learning_rate=5e-4,
#         weight_decay=0.01,
#         logging_dir='./logs',
#         load_best_model_at_end=True,
#         fp16=False,
#         gradient_accumulation_steps=2,
# )

batchsize = 5

training_args = TrainingArguments(
        output_dir='./results/models',
        evaluation_strategy='epoch',  # Make sure both are the same
        save_strategy='epoch',  # Make sure both are the same
        num_train_epochs=10,
        per_device_train_batch_size=batchsize,
        per_device_eval_batch_size=batchsize,
        learning_rate=5e-4,
        weight_decay=0.01,
        logging_dir='./logs',
        load_best_model_at_end=True,
        fp16=False,  # Keep fp16=False
        gradient_accumulation_steps=4,
        warmup_steps=1000,
        logging_steps=100,
        max_grad_norm=1.0,
        save_total_limit=3,
        metric_for_best_model='eval_loss',
        greater_is_better=True,
        disable_tqdm=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
)

print(training_args)

# PRETRAIN MODEL
# During class, look at:
# - the use of CPU RAM, GPU RAM
# - estimates how long it takes for 1 epoch

# Don't train until the end as it takes too much time (20h).

# with 105Mb Mixtral model, 110h / 50 epochs (1M sequences) = 2h / epoch
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
)

print ('Start a trainer...')
# Start training
trainer.train()
