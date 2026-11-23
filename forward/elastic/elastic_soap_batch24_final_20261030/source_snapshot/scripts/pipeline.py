"""Train the elastic neural operator on a prepared dataset."""
import argparse
import scno_posix_lock as fcntl
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eno.core import atomic_json
from eno.train import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stage", choices=["train"], default="train")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    with (run / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = run / "CONFIG.json"
        if old.exists() and json.loads(old.read_text(encoding="utf-8")) != config:
            raise ValueError("Run configuration differs from the saved configuration")
        atomic_json(old, config)
        train(config, run)


if __name__ == "__main__":
    main()
