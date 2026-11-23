"""Read the published joint pressure view directly from its sealed source files."""

from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
import torch

from native400_runtime.feature_enhanced_objective_20261022 import SampledLabels


class DirectJointReads:
    def __init__(self, corpus):
        self.corpus = corpus
        self.paths = [(corpus.root / row["source_h5"]).resolve() for row in corpus.records]

    def batch(self, record, frames):
        corpus = self.corpus
        path = self.paths[record]
        if corpus._identity(path) != corpus._identities[path]:
            raise ValueError(f"sealed source file changed: {path}")
        indices = frames
        if frames[-1] - frames[0] + 1 == len(frames):
            indices = slice(frames[0], frames[-1] + 1)
        with h5py.File(path, "r") as h5:
            c = np.asarray(h5["velocity401"][::2, ::2], dtype=np.float32)
            source = np.asarray(h5["source_parameters"][:], dtype=np.float32).reshape(-1)
            labels = h5["pressure401"][indices, ::2, ::2]
        expected = np.asarray(corpus.records[record]["source_parameters"], dtype=np.float32)
        np.testing.assert_array_equal(source, expected)
        if not (c.shape == (201, 201) and np.isfinite(c).all() and (c > 0).all()):
            raise ValueError(f"invalid velocity grid: {path}")
        return {"c": torch.from_numpy(c)[None, None], "source": torch.from_numpy(source)[None],
                "labels": SampledLabels(frames, torch.from_numpy(labels)[None])}

    def check_published_view(self):
        """Confirm both the source/velocity mapping and the general time-axis indexing."""
        report = {}
        frames = np.unique(np.linspace(0, len(self.corpus.times) - 1, 64, dtype=np.int64)).tolist()
        for index, record in enumerate(self.corpus.records):
            family = record["family"]
            if family in report:
                continue
            tick = perf_counter()
            reference = self.corpus.batch([index], frames)
            vds_wall = perf_counter() - tick
            tick = perf_counter()
            current = self.batch(index, frames)
            direct_wall = perf_counter() - tick
            for key in ("c", "source"):
                torch.testing.assert_close(current[key], reference[key], rtol=0, atol=0)
            torch.testing.assert_close(current["labels"].values, reference["labels"].values, rtol=0, atol=0)
            report[family] = {"record_id": record["record_id"], "queries_checked": len(frames),
                              "bit_identical": True, "vds_first_read_s": vds_wall,
                              "direct_following_read_s": direct_wall}
        return report
