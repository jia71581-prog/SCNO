"""Original IC8;392 real stored forecast instants including all old92."""
import hashlib,json,importlib.util
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
p=ROOT.parent/'r21'/'protocol92.py'
s=importlib.util.spec_from_file_location('r24_old_times_readonly',p);old=importlib.util.module_from_spec(s);s.loader.exec_module(old)
IC_FRAMES=tuple(old.IC_FRAMES);OLD92_FRAMES=tuple(old.FORECAST_FRAMES)
# The remote CC worker wrote the frame generator. Only its reviewed exact-frame
# construction is used; its separate adaptive-sampler draft is NOT deployed.
ccpath=ROOT/'cc_protocol/protocol400.py'
cs=importlib.util.spec_from_file_location('r24_cc_frame_builder',ccpath)
cc=importlib.util.module_from_spec(cs);cs.loader.exec_module(cc)
assert tuple(cc.IC_FRAMES)==IC_FRAMES and tuple(cc.OLD_FORECAST_92)==OLD92_FRAMES
FORECAST_FRAMES=tuple(cc.FORECAST_FRAMES_392)
FULL_FRAMES=IC_FRAMES+FORECAST_FRAMES
OLD92_POS=tuple(FORECAST_FRAMES.index(i) for i in OLD92_FRAMES)
BINS=((0,int(np.searchsorted(FORECAST_FRAMES,319))),
      (int(np.searchsorted(FORECAST_FRAMES,319)),int(np.searchsorted(FORECAST_FRAMES,549))),
      (int(np.searchsorted(FORECAST_FRAMES,549)),392))
assert len(FULL_FRAMES)==400 and len(FORECAST_FRAMES)==392 and len(set(FULL_FRAMES))==400
assert all(a<b for a,b in zip(FULL_FRAMES,FULL_FRAMES[1:]))
assert set(OLD92_FRAMES)<=set(FORECAST_FRAMES) and min(FORECAST_FRAMES)==146 and max(FORECAST_FRAMES)==800
assert IC_FRAMES==(88,95,102,110,117,124,131,138)
PROTOCOL=dict(schema='R24_REAL201_NATIVE400_IC8_PLUS392_V1',full_frames=FULL_FRAMES,
 forecast_frames=FORECAST_FRAMES,known_frames=IC_FRAMES,old92_positions=OLD92_POS,
 stored_dt_s=.00125,grid=[201,201,400],spacing_m=10.,forecast_seconds=[.1825,1.],
 max_forecast_gap_native=int(np.diff(FORECAST_FRAMES).max()),bins=BINS,
 labels='exact stored frames,no temporal interpolation',old92_metric_preserved=True)
PROTOCOL_SHA=hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()
