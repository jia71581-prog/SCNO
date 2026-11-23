"""R21 TRAIN loader: raw HDF201, original IC8+forecast92, pressure encoded once."""
import json
import hashlib
import numpy as np
import torch
from bootstrap import upstream
from protocol92 import FULL_FRAMES, IC_FRAMES, FORECAST_FRAMES
import data_views201 as view

bridge = upstream("model_bridge")
old = upstream("pit_loader")
SOURCE_FIELDS = ("source_x_m", "source_z_m", "source_f0_hz", "source_t0_s", "source_amplitude")

class Forecast92Loader(old.TrainViewLoader):
    def __init__(self, family, code, config):
        super().__init__(family, code, config)
        self.original_view_id = self.view_id
        identity = dict(parent_roster_view=self.original_view_id, protocol=view.protocol(),
                        pressure_normalization="physical IC; forecast encoded exactly once",
                        original_roster_unchanged=True)
        self.view_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    def build_known_inputs(self, sid, physical_ic):
        if sid not in self.byid:
            raise PermissionError("original TRAIN roster only")
        if tuple(physical_ic.shape) != (1, 8, 201, 201):
            raise ValueError("eight actual native201 physical Pa observations")
        row = self.byid[sid]
        local = row["local_index"]
        rec = self.pool.records[local]
        axis = rec.time_s
        h = self.pool.records._file()
        idx = row["source_index"]
        fp = np.asarray([h[k][idx] for k in SOURCE_FIELDS], dtype=np.float64)
        data = dict(sample_id=sid, local_index=local,
                    velocity=view.velocity_view(rec.velocity_mps[None]),
                    source=torch.tensor(fp, dtype=torch.float32)[None],
                    source_map=torch.from_numpy(view.source_map201(fp))[None,None],
                    ic=physical_ic.clone(), anchor=axis[IC_FRAMES[0]:IC_FRAMES[0]+1],
                    x=rec.x_m, z=rec.z_m, time_axis=axis, family=rec.medium_type,
                    future_start=FORECAST_FRAMES[0], sample_sha256=row["sample_sha256"])
        if (tuple(data["velocity"].shape) != (1,1,201,201)
                or len(data["x"]) != 201 or len(data["z"]) != 201
                or abs(float(data["anchor"])-.11) > 1e-7):
            raise AssertionError("native201 velocity/axes and original first-IC anchor")
        for key in ("x", "z"):
            coordinates = torch.as_tensor(data[key], dtype=torch.float64)
            if (abs(float(coordinates[0])) > 1e-9
                    or abs(float(coordinates[-1])-2000.) > 1e-9
                    or not bool(torch.allclose(coordinates[1:]-coordinates[:-1],
                                               torch.full((200,),10.,dtype=torch.float64),
                                               atol=1e-9, rtol=0))):
                raise AssertionError("R21 complete 10m physical coordinate axis")
        if not bool(torch.all(data["ic"][...,0,:] == 0)):
            raise AssertionError("physical IC exact free-surface row")
        for name, fn in (("phase", self.pool.phase_maps), ("scatter", self.pool.scatter_tables)):
            if fn(sid) is not None:
                raise PermissionError("explicit native201 cache contract required: "+name)
        packet = dict(sample_id=sid, source_index=idx, group_id=row["group_id"], split="train",
                      sample_sha256=row["sample_sha256"], view_id=self.view_id,
                      original_view_id=self.original_view_id, spatial_view_schema=view.VIEW_SCHEMA,
                      spatial_shape=[201,201], dx_m=10., dz_m=10.,
                      IC_frames=list(IC_FRAMES), IC_times=axis[list(IC_FRAMES)].tolist(),
                      forecast_frames=list(FORECAST_FRAMES), source_FP64=fp.tolist(),
                      IC_sha256=old.digest_array(data["ic"].numpy()),
                      velocity_sha256=old.digest_array(data["velocity"].numpy()),
                      source_map_sha256=old.digest_array(data["source_map"].numpy()),
                      IC_physical_Pa=True, anchor_first_IC=True,
                      raw_HDF201=True, no_101_interpolation=True)
        return data, packet

    def read(self, sid):
        row = self.byid[sid]
        rec = self.pool.records[row["local_index"]]
        gt = self.pool.records.read_wavefield(row["local_index"], rec.time_s[list(FULL_FRAMES)])
        if not bool(gt.exact.all()):
            raise PermissionError("native100 exact stored pressure times")
        physical = view.pressure_view(gt.values)[None]
        if tuple(physical.shape) != (1,100,201,201):
            raise AssertionError("actual HDF201 full100 pressure")
        data, packet = self.build_known_inputs(sid, physical[:,:8])
        target = self.pool.normalizer.encode_pressure(physical[:,8:], data["source"][:,4])
        packet.update(target_frames=list(FORECAST_FRAMES),
                      target_sha256=old.digest_array(target.numpy()),
                      target_normalized_once=True)
        return data, target, packet
