import polars as pl
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

# Configuration
fasta_file = "/project/pi_annagreen_umass_edu/saishradha/plm_circuits_private/uniref_sequences/uniref50.fasta"
output_parquet = "/project/pi_annagreen_umass_edu/saishradha/plm_circuits_private/uniref_sequences/uniref_1M.parquet"
target_samples = 1000000
max_length = 1022
chunk_size = 10000  # File chunk size
batch_size = 10000  # Write batch size (memory efficient)


def parse_fasta_chunk(lines, max_length, target_per_chunk):
    """Parse FASTA lines and filter by length. Returns (ids, sequences)."""
    ids = []
    sequences = []
    current_id = None
    current_seq_len = 0  # Track length instead of recalculating
    current_seq = []

    for line in lines:
        line = line.rstrip("\n")
        if line.startswith(">"):
            # Save previous sequence if it passes filter
            if (
                current_id is not None
                and current_seq_len > 0
                and current_seq_len <= max_length
            ):
                ids.append(current_id)
                sequences.append("".join(current_seq))
                if len(ids) >= target_per_chunk:
                    break

            # Parse new ID
            current_id = line[1:].split("|")[1] if "|" in line else line[1:].split()[0]
            current_seq_len = 0
            current_seq = []
        else:
            current_seq.append(line)
            current_seq_len += len(line)  # Running sum

    # Don't forget the last sequence
    if current_id is not None and current_seq_len > 0 and current_seq_len <= max_length:
        ids.append(current_id)
        sequences.append("".join(current_seq))

    return ids, sequences


def load_file_in_chunks(fasta_file, chunk_size):
    """Load file and yield chunks of lines (streaming)."""
    with open(fasta_file, "r") as f:
        chunk = []
        for line in f:
            chunk.append(line)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def process_file_multiprocess(
    fasta_file, target_samples, max_length, batch_size, output_parquet
):
    """Process FASTA file using multiprocessing with memory-efficient batch writing."""
    num_processes = cpu_count()
    target_per_chunk = target_samples // num_processes + 1000  # Buffer for safety

    print(
        f"Using {num_processes} processes, batch writing every {batch_size} records..."
    )

    # Stream chunks to workers (don't load entire file first)
    process_fn = partial(
        parse_fasta_chunk, max_length=max_length, target_per_chunk=target_per_chunk
    )

    batch_ids = []
    batch_sequences = []
    total_written = 0
    first_batch = True

    with Pool(num_processes) as pool:
        # Stream results as they complete
        for ids, sequences in tqdm(
            pool.imap_unordered(
                process_fn, load_file_in_chunks(fasta_file, chunk_size), chunksize=1
            ),
            desc="Processing",
        ):
            batch_ids.extend(ids)
            batch_sequences.extend(sequences)

            # Write batch when it reaches batch_size or when we have enough total samples
            remaining_needed = target_samples - total_written
            if len(batch_ids) >= batch_size or len(batch_ids) >= remaining_needed:
                # Trim batch to remaining needed, not to batch_size
                trim_to = min(batch_size, remaining_needed)
                batch_ids = batch_ids[:trim_to]
                batch_sequences = batch_sequences[:trim_to]

                df_batch = pl.DataFrame({"id": batch_ids, "sequence": batch_sequences})

                # First batch: create file, subsequent batches: append
                if first_batch:
                    df_batch.write_parquet(output_parquet, compression="snappy")
                    first_batch = False
                else:
                    # Append to existing parquet (read + combine + write)
                    df_existing = pl.read_parquet(output_parquet)
                    df_combined = pl.concat([df_existing, df_batch])
                    df_combined.write_parquet(output_parquet, compression="snappy")

                total_written += len(batch_ids)
                batch_ids = []
                batch_sequences = []

                if total_written >= target_samples:
                    break

    # Write remaining batch
    if batch_ids and total_written < target_samples:
        # Trim to exact target
        remaining = target_samples - total_written
        batch_ids = batch_ids[:remaining]
        batch_sequences = batch_sequences[:remaining]

        df_batch = pl.DataFrame({"id": batch_ids, "sequence": batch_sequences})

        if first_batch:
            df_batch.write_parquet(output_parquet, compression="snappy")
        else:
            df_existing = pl.read_parquet(output_parquet)
            df_combined = pl.concat([df_existing, df_batch])
            df_combined.write_parquet(output_parquet, compression="snappy")

        total_written += len(batch_ids)


# Process file
process_file_multiprocess(
    fasta_file, target_samples, max_length, batch_size, output_parquet
)

# Verify output
df_final = pl.read_parquet(output_parquet)
print(f"\nSaved {len(df_final)} sequences to {output_parquet}")
