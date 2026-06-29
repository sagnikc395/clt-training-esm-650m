# %%
"""
Create a protein dataset from sampled UniRef sequences for ESM2 SAE training.
Optimized for 1M+ sequences: batch tokenization, streaming processing, low memory.
"""

import polars as pl
from datasets import Dataset
from transformers import AutoTokenizer
from multiprocessing import cpu_count

# %%
# Configuration
DATA_PATH = "/project/pi_annagreen_umass_edu/saishradha/plm_circuits_private/uniref_sequences/uniref_1M.parquet"
OUTPUT_PATH = "clt-training/protein_test_dataset"
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
MAX_LENGTH = 1022
BATCH_SIZE = 4096  # Tokenizer batch size (not training batch size)

# %%
# Load and filter sequences using Polars (stays in columnar format, no .to_list())
df = pl.read_parquet(DATA_PATH)
print(f"Loaded {len(df)} protein sequences")

# Filter by length directly in Polars (avoids creating a Python list)
df = df.filter(pl.col("sequence").str.len_chars() <= MAX_LENGTH)
print(f"After length filter (<= {MAX_LENGTH} chars): {len(df)} sequences")

# %%
# Load ESM2 tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"Tokenizer: {tokenizer.__class__.__name__}, vocab size: {len(tokenizer)}")

# %%
# Create HuggingFace dataset from Polars (zero-copy for strings)
# Only extract the 'sequence' column — avoids duplicating the full DataFrame
hf_dataset = Dataset.from_dict({"text": df["sequence"].to_list()})
del df  # Free the Polars DataFrame

print(f"HF Dataset created: {len(hf_dataset)} sequences")


# %%
# Batch tokenize using HuggingFace .map() — parallelized and memory-efficient
def tokenize_batch(examples):
    """Batch tokenize protein sequences with padding to fixed length."""
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_attention_mask=True,
        return_tensors=None,
    )
    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
    }


num_proc = min(cpu_count() // 2, 16)  # Cap at 16 to avoid diminishing returns
print(f"Tokenizing with batch_size={BATCH_SIZE}, num_proc={num_proc}...")

dataset = hf_dataset.map(
    tokenize_batch,
    batched=True,
    batch_size=BATCH_SIZE,
    num_proc=num_proc,
    remove_columns=["text"],  # Drop raw text to save memory
    desc="Tokenizing",
)

print(f"Tokenized dataset: {len(dataset)} sequences")
print(f"Features: {dataset.features}")
print(f"Example input_ids length: {len(dataset[0]['input_ids'])}")

# %%
# Save dataset (Arrow format — memory-mapped, fast loading)
dataset.save_to_disk(OUTPUT_PATH)
print(f"Dataset saved to: {OUTPUT_PATH}")

# %%
# Verify
from datasets import load_from_disk

loaded = load_from_disk(OUTPUT_PATH)
print(
    f"Verified: {len(loaded)} sequences, input_ids length: {len(loaded[0]['input_ids'])}"
)

# %%
