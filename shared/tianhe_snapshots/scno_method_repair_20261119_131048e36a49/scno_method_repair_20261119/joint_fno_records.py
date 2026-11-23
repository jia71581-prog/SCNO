"""Load sealed trajectories on their native axes for joint time-channel training."""

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from data_v2 import open_corpus, _write_json


def spread(ids, count):
    if count >= len(ids):
        return list(ids)
    indices = (np.arange(count) + .5) * len(ids) / count
    return [ids[int(index)] for index in indices]


def select_records(corpus, split, per_family):
    by_family = defaultdict(list)
    for key in getattr(corpus, split + "_ids"):
        by_family[corpus.records[key]["family"]].append(key)
    return [key for group in sorted(by_family) for key in spread(by_family[group], per_family)]


class Trajectories:
    def __init__(self, corpus, directory=None):
        self.corpus = corpus
        self.slots = np.arange(len(corpus.times))
        self.root = None
        if directory is not None:
            binding = json.dumps(corpus.binding, sort_keys=True).encode()
            self.root = Path(directory) / hashlib.sha256(binding).hexdigest()
            self.root.mkdir(parents=True, exist_ok=True)

    def read(self, key, *, cache=False):
        path = None
        if self.root is not None:
            path = self.root / (hashlib.sha256(str(key).encode()).hexdigest() + ".pt")
            if path.is_file():
                row = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
                if row["id"] != str(key) or row["target"].shape != (len(self.slots), 201, 201):
                    raise ValueError("cached trajectory conflicts with the sealed record")
                return row
        data = self.corpus.read(key, self.slots)
        row = {"id": str(key), "family": self.corpus.records[key]["family"],
               "velocity": data["velocity"], "source": data["source"], "target": data["target"]}
        if path is not None and cache:
            temporary = path.with_suffix(f".{os.getpid()}.tmp")
            try:
                torch.save(row, temporary)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--train-per-family", type=int, default=6)
    p.add_argument("--val-per-family", type=int, default=2)
    args = p.parse_args()
    corpus = open_corpus(json.loads(args.config.read_text())["data"])
    store = Trajectories(corpus, args.cache)
    roster = {split: select_records(corpus, split, n) for split, n in
              [("train", args.train_per_family), ("validation", args.val_per_family)]}
    if not roster["validation"]:
        raise ValueError("held-out records required")
    _write_json(args.out / "SPLIT.json", roster)
    for split, ids in roster.items():
        for key in ids:
            start = perf_counter()
            row = store.read(key, cache=True)
            print(split, key, row["family"], list(row["target"].shape), round(perf_counter() - start, 2), flush=True)
    _write_json(args.out / "DATA_READY.json", roster)


if __name__ == "__main__":
    main()
