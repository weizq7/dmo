"""Reconstruct the frozen evaluation cohort without re-sampling or train scans."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

from .generation_protocol import (
    ASSETS,
    TOKENIZER_ID,
    TOKENIZER_REVISION,
    SCORER_ID,
    SCORER_REVISION,
    SCORER_PROTOCOL,
    atomic_json,
    file_sha,
    load_manifest,
)

TOKENIZER_FILES = {
    "tokenizer.json": "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
    "tokenizer_config.json": "4bb9af56a342753d39374f4016a16574cab299fe088e896f425ce3c433f61424",
    "special_tokens_map.json": "e786b595b9a23148bf1630df78d9037a048ea671e48bfd3549a1e3c233742bb3",
}
SCORER_FILES = {
    "config.json": "7fccdcfd6622055342a734c663ee0b61ff4fd697f42467595df0bf4448c8c170",
    "model.safetensors": "5f47f3e12f91cd33b662ce7e433b6150ad5512b5884a2cee961b50e9c3bbebce",
    "tokenizer.json": "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6",
    "tokenizer_config.json": "5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f",
}


def digest_text(text):
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def verify_model_files(path, expected):
    actual = {name: file_sha(Path(path) / name) for name in expected}
    if actual != expected:
        raise ValueError(
            f"Cached model/tokenizer differs from the pinned revision: {path}"
        )
    return actual


def resolve_snapshot(local_path, model_id, revision, patterns, offline=False):
    if local_path:
        path = Path(local_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            model_id,
            revision=revision,
            allow_patterns=patterns,
            local_files_only=offline,
        )
    )


def read_selection():
    cohort = json.loads((ASSETS / "cohort.json").read_text())
    path = ASSETS / "selection.csv"
    if file_sha(path) != cohort["selection_sha256"]:
        raise ValueError("Fixed cohort selection checksum mismatch")
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ("sample_id", "row_index", "priority"):
            row[key] = int(row[key])
        if Path(row["shard"]).name != row["shard"] or row["row_index"] < 0:
            raise ValueError("Invalid cohort source locator")
    if [r["sample_id"] for r in rows] != list(range(cohort["count"])):
        raise ValueError("Invalid cohort sample IDs")
    return rows


def reconstruct_manifest(val_data, tokenizer, output):
    import pyarrow.parquet as pq

    selection = read_selection()
    grouped = defaultdict(list)
    for row in selection:
        grouped[row["shard"]].append(row)
    result = [None] * len(selection)
    special = set(tokenizer.all_special_ids)
    for index, (shard, rows) in enumerate(sorted(grouped.items())):
        path = Path(val_data) / shard
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing validation shard: {path}; see Dataset Preparation in README"
            )
        table = pq.read_table(path, columns=["text"], use_threads=False)
        texts = table.take([r["row_index"] for r in rows]).column(0).to_pylist()
        encoded = tokenizer(
            texts, add_special_tokens=False, truncation=True, max_length=576
        )["input_ids"]
        for locator, text, ids in zip(rows, texts, encoded):
            if digest_text(text) != locator["document_sha256"]:
                raise ValueError(
                    f"Validation document mismatch: {shard}, row {locator['row_index']}"
                )
            if len(ids) != 576 or special.intersection(ids):
                raise ValueError("Tokenizer does not reproduce the fixed sample")
            prompt = tokenizer.decode(ids[:64], clean_up_tokenization_spaces=False)
            reference = tokenizer.decode(ids[64:], clean_up_tokenization_spaces=False)
            human_text = tokenizer.decode(ids, clean_up_tokenization_spaces=False)
            result[locator["sample_id"]] = dict(
                sample_id=locator["sample_id"],
                priority=locator["priority"],
                source=shard,
                row_index=locator["row_index"],
                document_sha256=locator["document_sha256"],
                prompt=prompt,
                reference=reference,
                human_text=human_text,
                prompt_ids=ids[:64],
                reference_ids=ids[64:],
            )
        if (index + 1) % 20 == 0 or index + 1 == len(grouped):
            print(
                f"Reconstructed {index + 1}/{len(grouped)} validation shards",
                flush=True,
            )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for row in result:
                stream.write(
                    json.dumps(
                        row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    )
                    + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        load_manifest(temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(args):
    from transformers import AutoTokenizer

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    tokenizer_path = resolve_snapshot(
        args.tokenizer_path,
        TOKENIZER_ID,
        TOKENIZER_REVISION,
        [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "merges.txt",
            "vocab.json",
        ],
        args.offline,
    )
    verify_model_files(tokenizer_path, TOKENIZER_FILES)
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), local_files_only=True
    )
    manifest = output / "prompts.jsonl"
    if manifest.exists():
        load_manifest(manifest)
        print("The fixed 5,000-prompt manifest is already verified.", flush=True)
    else:
        reconstruct_manifest(args.val_data, tokenizer, manifest)
    scorer_path = resolve_snapshot(
        args.scorer_path,
        SCORER_ID,
        SCORER_REVISION,
        [
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "merges.txt",
            "vocab.json",
        ],
        args.offline,
    )
    verify_model_files(scorer_path, SCORER_FILES)
    config = dict(SCORER_PROTOCOL, local_path=str(scorer_path))
    atomic_json(output / "scorer_config.json", config)
    atomic_json(
        output / "prepared.json",
        dict(
            manifest_sha256=file_sha(manifest),
            tokenizer_path=str(tokenizer_path),
            tokenizer_files=TOKENIZER_FILES,
            scorer_path=str(scorer_path),
            scorer_files=SCORER_FILES,
            scorer_config_sha256=file_sha(output / "scorer_config.json"),
            cohort=json.loads((ASSETS / "cohort.json").read_text()),
        ),
    )
    print(
        f"READY: {output} (5,000 fixed prompts and pinned GPT-2-large scorer)",
        flush=True,
    )


def verify_prepared(directory):
    directory = Path(directory)
    meta = json.loads((directory / "prepared.json").read_text())
    load_manifest(directory / "prompts.jsonl")
    if meta["manifest_sha256"] != file_sha(directory / "prompts.jsonl"):
        raise ValueError("Prepared manifest checksum mismatch")
    if file_sha(directory / "scorer_config.json") != meta["scorer_config_sha256"]:
        raise ValueError("Prepared scorer configuration changed")
    cfg = json.loads((directory / "scorer_config.json").read_text())
    if {k: v for k, v in cfg.items() if k != "local_path"} != SCORER_PROTOCOL or cfg[
        "local_path"
    ] != meta["scorer_path"]:
        raise ValueError("Scorer protocol differs from the fixed experiment")
    verify_model_files(meta["tokenizer_path"], TOKENIZER_FILES)
    verify_model_files(meta["scorer_path"], SCORER_FILES)
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--val-data", default="data/fineweb-edu-shuffled/data/val/train"
    )
    parser.add_argument("--output", default="data/generation")
    parser.add_argument(
        "--tokenizer-path", help="Optional existing pinned tokenizer snapshot directory"
    )
    parser.add_argument(
        "--scorer-path", help="Optional existing pinned GPT-2-large snapshot directory"
    )
    parser.add_argument(
        "--offline", action="store_true", help="Use only already-cached model files"
    )
    prepare(parser.parse_args())
