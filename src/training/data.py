"""Data loading and tokenization utilities for LoopLM pre-training."""

import glob
import os
import random
from typing import Iterator
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
from torch.utils.data import (
    DataLoader,
    DistributedSampler,
    IterableDataset,
    TensorDataset,
    get_worker_info,
)


def make_tokenizer(model_id: str = "HuggingFaceTB/SmolLM2-135M"):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id)


def tokenize_and_chunk(texts: list[str], tokenizer, seq_len: int) -> Tensor:
    """Tokenize a list of strings and split into fixed-length chunks.

    Returns:
        (N, seq_len + 1) LongTensor — each row is a training example.
        The trainer will use [:, :-1] as input and [:, 1:] as targets.
    """
    all_ids: list[int] = []
    eos = tokenizer.eos_token_id or 0
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            all_ids.extend(ids)
            all_ids.append(eos)  # document separator

    tokens = torch.tensor(all_ids, dtype=torch.long)
    chunk_len = seq_len + 1
    n_chunks = len(tokens) // chunk_len
    return tokens[: n_chunks * chunk_len].view(n_chunks, chunk_len)


def make_dataloader(
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-103-v1",
    split: str = "train",
    tokenizer_id: str = "HuggingFaceTB/SmolLM2-135M",
    seq_len: int = 512,
    batch_size: int = 4,
    max_chunks: int | None = None,
    shuffle: bool = True,
    text_column: str = "text",
    seed: int = 42,
    drop_last: bool = True,
    use_distributed_sampler: bool | None = None,
) -> DataLoader:
    """Load a HuggingFace text dataset and return a ready-to-use DataLoader.

    Tokenization and chunking are applied via datasets.map so results are
    cached to disk automatically — subsequent runs with the same args skip
    re-tokenization.

    Args:
        max_chunks: cap the number of training chunks (useful for quick runs)
    """
    from datasets import load_dataset

    tokenizer = make_tokenizer(tokenizer_id)
    eos = tokenizer.eos_token_id or 0
    chunk_len = seq_len + 1

    dataset = load_dataset(dataset_name, dataset_config, split=split)

    # Step 1: tokenize each document → list of token IDs with EOS separator.
    # datasets caches this on disk keyed by the map function source + args.
    def _tokenize(batch):
        ids_per_doc = []
        for text in batch[text_column]:
            if text.strip():
                ids = tokenizer.encode(text, add_special_tokens=False)
                ids_per_doc.append(ids + [eos])
            else:
                ids_per_doc.append([])
        return {"input_ids": ids_per_doc}

    tokenized = dataset.map(
        _tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    # Step 2: concatenate all token IDs across the batch and split into
    # fixed-length chunks of (seq_len + 1).
    def _chunk(batch):
        all_ids: list[int] = []
        for ids in batch["input_ids"]:
            all_ids.extend(ids)
        n = len(all_ids) // chunk_len
        chunks = [all_ids[i * chunk_len : (i + 1) * chunk_len] for i in range(n)]
        return {"input_ids": chunks}

    chunked = tokenized.map(
        _chunk,
        batched=True,
        desc="Chunking",
    )

    if max_chunks is not None:
        chunked = chunked.select(range(min(max_chunks, len(chunked))))

    chunked.set_format(type="torch", columns=["input_ids"])

    print(
        f"Dataset: {len(chunked):,} chunks × {chunk_len} tokens "
        f"({len(chunked) * chunk_len / 1e6:.1f}M tokens)"
    )

    def _collate(batch):
        return torch.stack([x["input_ids"] for x in batch])

    sampler = None
    if use_distributed_sampler is None:
        use_distributed_sampler = dist.is_available() and dist.is_initialized()
    if use_distributed_sampler and dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(
            chunked,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

    return DataLoader(
        chunked,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        drop_last=drop_last,
        collate_fn=_collate,
    )


# ---------------------------------------------------------------------------
# Deterministic parquet streaming for FineWeb-Edu.
#
# The file and row-group ordering follows the public Parcae data pipeline:
# files are globally shuffled, sharded by rank, and documents within each row
# group are shuffled with a rank-specific deterministic RNG. Documents are
# then packed into fixed-length next-token-prediction sequences.
# ---------------------------------------------------------------------------


class PackedParquetDataset(IterableDataset):
    """Stream fixed-length packed token sequences from parquet documents."""

    def __init__(
        self,
        data_dir: str,
        tokenizer,
        seq_len: int,
        seed: int = 42,
        pack_buffer_size: int = 1000,
    ):
        if seq_len < 1:
            raise ValueError("seq_len must be positive")
        if pack_buffer_size < 1:
            raise ValueError("pack_buffer_size must be positive")

        try:
            import pyarrow.parquet as pq  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "FineWeb parquet streaming requires pyarrow in the training environment"
            ) from exc

        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Parquet data directory does not exist: {data_dir}")
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.chunk_len = self.seq_len + 1
        self.seed = int(seed)
        self.pack_buffer_size = int(pack_buffer_size)

        all_files = sorted(str(path) for path in self.data_dir.rglob("*.parquet"))
        if not all_files:
            raise FileNotFoundError(f"No parquet files found under {data_dir}")

        file_rng = random.Random(self.seed)
        file_rng.shuffle(all_files)
        self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.world_size = (
            dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        )
        self.parquet_files = all_files[self.rank :: self.world_size]
        if not self.parquet_files:
            raise RuntimeError(
                f"Rank {self.rank} received no parquet files from {len(all_files)} files"
            )

    def _documents(self) -> Iterator[list[int]]:
        import pyarrow.parquet as pq

        row_rng = random.Random(self.seed + self.rank)
        for parquet_path in self.parquet_files:
            parquet_file = pq.ParquetFile(parquet_path)
            for row_group_idx in range(parquet_file.num_row_groups):
                column = parquet_file.read_row_group(row_group_idx, columns=["text"])
                texts = [text or "" for text in column.column("text").to_pylist()]
                row_rng.shuffle(texts)
                for text in texts:
                    if not text.strip():
                        continue
                    token_ids = self.tokenizer.encode(text, add_special_tokens=False)
                    if token_ids:
                        yield list(token_ids) + [self.tokenizer.eos_token_id or 0]

    def _packed_chunks(self) -> Iterator[Tensor]:
        documents = iter(self._documents())
        buffer: list[list[int]] = []

        def refill() -> None:
            while len(buffer) < self.pack_buffer_size:
                try:
                    buffer.append(next(documents))
                except StopIteration:
                    return

        while True:
            refill()
            if not buffer:
                return

            row = torch.empty(self.chunk_len, dtype=torch.long)
            pos = 0
            while pos < self.chunk_len:
                refill()
                if not buffer:
                    return

                remaining = self.chunk_len - pos
                eligible = [
                    (index, len(document))
                    for index, document in enumerate(buffer)
                    if len(document) <= remaining
                ]
                if eligible:
                    best_index, _ = max(eligible, key=lambda item: item[1])
                    document = buffer.pop(best_index)
                    row[pos : pos + len(document)] = torch.tensor(
                        document, dtype=torch.long
                    )
                    pos += len(document)
                else:
                    shortest_index = min(range(len(buffer)), key=lambda i: len(buffer[i]))
                    document = buffer.pop(shortest_index)
                    row[pos:] = torch.tensor(document[:remaining], dtype=torch.long)
                    pos = self.chunk_len

            yield row

    def __iter__(self) -> Iterator[Tensor]:
        if get_worker_info() is not None:
            raise RuntimeError("PackedParquetDataset must be used with num_workers=0")
        yield from self._packed_chunks()


def make_parquet_dataloader(
    data_dir: str,
    tokenizer_id: str,
    seq_len: int,
    batch_size: int,
    seed: int = 42,
    pack_buffer_size: int = 1000,
    drop_last: bool = True,
) -> DataLoader:
    """Create the deterministic FineWeb-Edu parquet dataloader."""
    tokenizer = make_tokenizer(tokenizer_id)
    if hasattr(tokenizer, "model_max_length"):
        tokenizer.model_max_length = 1_000_000_000
    dataset = PackedParquetDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        seq_len=seq_len,
        seed=seed,
        pack_buffer_size=pack_buffer_size,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last,
        num_workers=0,
        pin_memory=True,
    )


# ── Pre-tokenized streaming data pipeline ────────────────────────────────────


class PreTokenizedDataset(IterableDataset):
    """Stream fixed-length chunks from pre-tokenized .npy shard files.

    Each shard is a 1-D uint16 array of concatenated, EOS-separated token IDs.
    The dataset yields (seq_len+1,) LongTensor chunks for next-token prediction.
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int = 4096,
        shuffle_shards: bool = True,
        seed: int = 42,
    ):
        self.shard_paths = sorted(glob.glob(f"{data_dir}/*.npy"))
        if not self.shard_paths:
            raise FileNotFoundError(f"No .npy shards found in {data_dir}")
        self.chunk_len = seq_len + 1
        self.shuffle_shards = shuffle_shards
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed)
        shard_paths = list(self.shard_paths)
        if self.shuffle_shards:
            rng.shuffle(shard_paths)

        # In DDP, each rank processes a disjoint subset of shards
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            shard_paths = shard_paths[rank::world_size]

        remainder = np.array([], dtype=np.uint16)

        for path in shard_paths:
            shard = np.load(path, mmap_mode="r")

            if len(remainder) > 0:
                # Need to copy mmap data before concatenating
                combined = np.concatenate([remainder, np.array(shard)])
            else:
                combined = np.array(shard)

            n_chunks = len(combined) // self.chunk_len
            for i in range(n_chunks):
                start = i * self.chunk_len
                chunk = combined[start : start + self.chunk_len]
                yield torch.from_numpy(chunk.astype(np.int64))

            remainder = combined[n_chunks * self.chunk_len :]


def make_pretokenized_dataloader(
    data_dir: str,
    seq_len: int = 4096,
    micro_batch_size: int = 2,
    num_workers: int = 2,
    seed: int = 42,
) -> DataLoader:
    """Create a streaming DataLoader from pre-tokenized .npy shards."""
    dataset = PreTokenizedDataset(data_dir, seq_len, seed=seed)

    n_shards = len(dataset.shard_paths)
    total_bytes = sum(p.stat().st_size for p in map(Path, dataset.shard_paths))
    approx_tokens = total_bytes // 2  # uint16
    print(
        f"Pre-tokenized data: {n_shards} shards, ~{approx_tokens/1e9:.1f}B tokens, "
        f"seq_len={seq_len}, micro_batch={micro_batch_size}"
    )

    return DataLoader(
        dataset,
        batch_size=micro_batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
