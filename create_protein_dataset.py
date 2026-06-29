# %%
"""
Create a simple protein dataset for testing ESM2 training.
"""

import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer

# %%
# Load protein sequences
DATA_PATH = "/work/pi_annagreen_umass_edu/jatin/plm_circuits_private/plm_nnsight/data/full_seq_dict.json"
with open(DATA_PATH, "r") as f:
    seq_dict = json.load(f)

print(f"Loaded {len(seq_dict)} protein sequences")
print(f"Protein IDs: {list(seq_dict.keys())[:10]}...")

# %%
# Load ESM2 tokenizer
model_name = "facebook/esm2_t33_650M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
print(f"Vocab size: {len(tokenizer)}")

# %%
# Tokenize all sequences
max_length = 512  # Start with shorter sequences for testing
sequences = list(seq_dict.values())

# Filter sequences that are too long
filtered_sequences = [s for s in sequences if len(s) <= max_length]
print(f"Sequences after filtering (length <= {max_length}): {len(filtered_sequences)}")

# %%
# Tokenize sequences WITH PADDING to fixed length
tokenized = []
for seq in filtered_sequences:
    tokens = tokenizer(
        seq,
        truncation=True,
        max_length=max_length,
        padding="max_length",  # PAD TO FIXED LENGTH
        return_tensors=None,
    )
    tokenized.append(
        {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
        }
    )

print(f"Tokenized {len(tokenized)} sequences")
print(f"All sequences padded to length: {max_length}")
print(f"Example token IDs length: {len(tokenized[0]['input_ids'])}")
print(
    f"Example attention mask: {tokenized[0]['attention_mask'][:20]}..."
)  # Show first 20

# %%
# Create HuggingFace dataset
dataset = Dataset.from_list(tokenized)
print(f"Dataset created: {len(dataset)} examples")
print(f"Dataset features: {dataset.features}")
print(f"First example: {dataset[0]}")

# %%
# Save dataset
output_path = "clt-training/protein_test_dataset"
dataset.save_to_disk(output_path)
print(f"Dataset saved to: {output_path}")

# %%
# Test loading
from datasets import load_from_disk

loaded = load_from_disk(output_path)
print(f"Loaded dataset: {len(loaded)} examples")
print(f"First example shape: {len(loaded[0]['input_ids'])}")

# %%
