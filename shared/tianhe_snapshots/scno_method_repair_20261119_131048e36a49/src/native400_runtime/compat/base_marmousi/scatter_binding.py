#!/usr/bin/env python3
"""Bind the scattering-truth cache into the training forward (20260926).

Mirrors `coarse_binding.py`: the model never reads a file, the CALLER binds the
per-record tensors onto `PreparedV3State`.  This module owns the cache layout so
exactly one place knows it.

Cache layout (written by `scatter_precompute.py`):
    $D/scatter_cache_v1/<sample_sha256>.npz
      T_src_y         [n_st]            float32   T(s -> y) at each kept station
      T_y_field       [n_st,201,201]    float32   T(y -> x) on the full query grid
      stations_iz_ix  [n_st,2]          int32     the lattice cells, (iz, ix)
      dropped_iz_ix   [n_st_drop,2]     int32     cells dropped for source proximity
      source_xz       [2]               float32
      x_m, z_m        [201]             float32
      meta            json string       schema/sample_id/sha/g/n_stations/iters/...

The head wants
    T(s -> y)   [record, g, g]
    T(y -> x)   [record, g, g, z, x]
so the flat [n_st] / [n_st,201,201] arrays are scattered back onto the g x g
lattice (dropped cells = +inf, i.e. "no scatterer there").

INDEX.json is written here (sha256 of every payload + the manifest binding) so the
cache directory carries its own integrity record, exactly like the phase cache.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

SCHEMA = "scatter_cache_index_v1"
INF = float("inf")


def index_path(cache_dir) -> Path:
    return Path(cache_dir) / "INDEX.json"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def build_index(cache_dir, *, grid: int = 7, records=None) -> dict:
    """Enumerate the cache, sha256 every payload and write INDEX.json.

    `records` is an optional iterable of dicts from `pool.roster`; when given, the
    index records which roster entries are present and fails loudly on a missing
    sha, because a silently absent record would zero the scattering term for that
    draw without any visible error.
    """
    cache = Path(cache_dir)
    files = sorted(cache.glob("*.npz"))
    entries = {}
    for p in files:
        entries[p.stem] = {"file": p.name, "bytes": p.stat().st_size,
                           "sha256": sha256_file(p)}
    payload = {
        "schema": SCHEMA,
        "grid": int(grid),
        "records": len(entries),
        "files": entries,
    }
    if records is not None:
        roster = list(records)
        missing = sorted({r["sample_sha256"] for r in roster} - set(entries))
        payload["roster_records"] = len(roster)
        payload["missing"] = missing
        if missing:
            raise ValueError(
                f"scatter cache is missing {len(missing)} roster records, e.g. {missing[:5]}")
    p = index_path(cache)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=1) + "\n")
    os.replace(tmp, p)
    return payload


def load_tables(cache_dir, sample_sha256: str, *, grid: int, query_shape, device="cpu"):
    """Read one record and lay it out for `ScatteringIntrinsicHead.forward`.

    Returns (T_source_scatterer [g,g], T_scatterer_query [g,g,z,x]).  Cells with no
    station are +inf, which the head already treats as "absent" (torch.isfinite
    masking), so a degenerate record contributes nothing rather than a wrong value.
    """
    cache = Path(cache_dir)
    path = cache / f"{sample_sha256}.npz"
    if not path.exists():
        raise FileNotFoundError(f"scatter cache miss for {sample_sha256}: {path}")
    with np.load(path, allow_pickle=False) as d:
        meta = json.loads(str(d["meta"]))
        if int(meta["g"]) != int(grid):
            raise ValueError(
                f"scatter cache grid {meta['g']} does not match the model's {grid} "
                f"({sample_sha256})")
        st = d["stations_iz_ix"]
        T_src = d["T_src_y"].astype(np.float32)
        T_full = d["T_y_field"].astype(np.float32)
        izmax = int(T_full.shape[1])
        ixmax = int(T_full.shape[2])
    if query_shape != (izmax, ixmax):
        raise ValueError(
            f"scatter T(y->x) is on a {izmax}x{ixmax} grid but the model queries "
            f"{tuple(query_shape)}; resample or re-run the cache at the query grid")
    Tz = np.full((grid, grid), INF, np.float32)
    Tq = np.full((grid, grid, izmax, ixmax), INF, np.float32)
    # The cache orders the lattice z-major (the same order `stations()` built it in).
    for k in range(st.shape[0]):
        iz, ix = int(st[k, 0]), int(st[k, 1])
        row, col = divmod(k, grid)
        Tz[row, col] = T_src[k]
        Tq[row, col] = T_full[k]
    return (torch.as_tensor(Tz, device=device),
            torch.as_tensor(Tq, device=device))


def cache_identity(cache_dir, *, grid: int, records=None) -> dict:
    """Binding record for the run identity: index sha + roster coverage."""
    p = index_path(cache_dir)
    if not p.exists():
        payload = build_index(cache_dir, grid=grid, records=records)
    else:
        payload = json.loads(p.read_text())
    return {
        "dir": str(Path(cache_dir).resolve()),
        "index_sha256": sha256_file(p),
        "grid": int(payload.get("grid", grid)),
        "records": int(payload.get("records", 0)),
        "missing": list(payload.get("missing", [])),
    }


__all__ = ["SCHEMA", "index_path", "build_index", "load_tables", "cache_identity",
           "sha256_file"]
