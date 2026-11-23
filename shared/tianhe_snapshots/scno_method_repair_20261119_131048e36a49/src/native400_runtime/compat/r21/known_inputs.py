"""Known-only IC8 read, followed by separately sealed raw201 future truth read."""
import torch
from protocol92 import FULL_FRAMES, IC_FRAMES

def _contract(view, full):
    if tuple(full) != tuple(FULL_FRAMES) or getattr(view, "GRID_N", None) != 201:
        raise PermissionError("R21 raw201 view and unchanged native100 protocol")

def known_only(loader, sid, view, full):
    _contract(view, full)
    row = loader.byid[sid]
    local = row["local_index"]
    rec = loader.pool.records[local]
    frames = list(IC_FRAMES)
    gt = loader.pool.records.read_wavefield(local, rec.time_s[frames])
    if not bool(gt.exact.all()):
        raise PermissionError("knownIC8 exact native stored201 times")
    physical = view.pressure_view(gt.values)[None]
    data, packet = loader.build_known_inputs(sid, physical)
    packet.update(read_native_frames=frames, no_unknownGT_read=True)
    return data, packet

def truth_after_seal(loader, sid, view, full, seal):
    _contract(view, full)
    from bootstrap import sha
    if not seal.exists():
        raise PermissionError("global forecast seal required before unknownGT")
    row = loader.byid[sid]
    rec = loader.pool.records[row["local_index"]]
    gt = loader.pool.records.read_wavefield(row["local_index"], rec.time_s[list(full)])
    if not bool(gt.exact.all()):
        raise PermissionError("postseal truth exact native100 stored201 only")
    physical = view.pressure_view(gt.values)
    if tuple(physical.shape) != (100,201,201):
        raise AssertionError("postseal actual201 complete pressure")
    return physical.numpy(), dict(sample_id=sid, GT_native_frames=list(full),
                                  forecast_seal_sha256=sha(seal),
                                  unknownGT_opened_after_seal=True, raw_HDF201=True,
                                  spatial_shape=[201,201], units="Pa")
