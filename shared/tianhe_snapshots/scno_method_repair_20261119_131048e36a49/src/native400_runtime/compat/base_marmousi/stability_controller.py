from __future__ import annotations
import math, statistics
from collections import deque

FAMILIES = ("uniform", "layered")

def _median(xs):
    return statistics.median(xs) if xs else None

def _mad_log(xs):
    if not xs:
        return None
    m = statistics.median(xs)
    return 1.4826 * statistics.median([abs(x-m) for x in xs])

def _p95(xs):
    if not xs:
        return None
    ys=sorted(xs)
    return ys[int(0.95*(len(ys)-1))]

class StabilityController:
    """Deterministic LR multiplier controller for residual-adaptive operator training.

    It never reacts to a single raw loss.  Decisions are based on per-family log
    ras_loss_unbiased windows, robust MAD/trend statistics, and pre-clip gradient norms.
    A separate causal-curriculum gate temporarily reduces LR when the horizon expands.
    """
    def __init__(self, cfg: dict, warmup_updates: int = 0):
        self.cfg=dict(cfg or {})
        self.enabled=bool(self.cfg.get("enabled", False))
        self.warmup_updates=int(warmup_updates)
        self.interval=int(self.cfg.get("control_every",25))
        self.window=int(self.cfg.get("window_per_family",64))
        self.min_points=int(self.cfg.get("min_points_per_family",16))
        self.mad_trigger=float(self.cfg.get("mad_trigger",1.20))
        self.trend_trigger=float(self.cfg.get("trend_trigger_log",math.log(1.50)))
        self.grad_trigger=float(self.cfg.get("grad_p95_trigger",20.0))
        self.grad_trend_trigger=float(self.cfg.get("grad_trend_trigger_log",math.log(1.20)))
        self.trigger_streak_needed=int(self.cfg.get("trigger_streak",2))
        self.recover_mad=float(self.cfg.get("recover_mad",0.65))
        self.recover_grad=float(self.cfg.get("recover_grad_p95",12.0))
        self.recover_streak_needed=int(self.cfg.get("recover_streak",3))
        self.backoff_mult=float(self.cfg.get("backoff_mult",0.5))
        self.recover_mult=float(self.cfg.get("recover_mult",1.1))
        self.min_factor=float(self.cfg.get("min_factor",0.25))
        self.cooldown=int(self.cfg.get("cooldown_updates",100))
        self.grace=int(self.cfg.get("causal_grace_updates",150))
        self.gate_min=float(self.cfg.get("curriculum_gate_min",0.5))
        self.gate_recover=int(self.cfg.get("curriculum_gate_recover_updates",150))
        self.family_logs={f:deque(maxlen=self.window) for f in FAMILIES}
        self.grad=deque(maxlen=max(64,self.window))
        self.pending=[]
        self.current_horizon=None
        self.switch_update=None
        self.gate_start=None
        self.stability_factor=1.0
        self.cooldown_until=-1
        self.bad_streak=0
        self.good_streak=0
        self.backoff_count=0
        self.recover_count=0
        self.last_control_update=-1
        self.epoch_values={f:[] for f in FAMILIES}
        self.epoch_switched=False
        self.prev_comparable_epoch=None
        self.last_epoch_guard={}

    def state_dict(self):
        return {
            "enabled":self.enabled,
            "family_logs":{k:list(v) for k,v in self.family_logs.items()},
            "grad":list(self.grad),"pending":list(self.pending),
            "current_horizon":self.current_horizon,"switch_update":self.switch_update,
            "gate_start":self.gate_start,"stability_factor":self.stability_factor,
            "cooldown_until":self.cooldown_until,"bad_streak":self.bad_streak,
            "good_streak":self.good_streak,"backoff_count":self.backoff_count,
            "recover_count":self.recover_count,"last_control_update":self.last_control_update,
            "epoch_values":self.epoch_values,"epoch_switched":self.epoch_switched,
            "prev_comparable_epoch":self.prev_comparable_epoch,
            "last_epoch_guard":self.last_epoch_guard,
        }

    def load_state_dict(self, s):
        if not s:
            return
        self.family_logs={f:deque(s.get("family_logs",{}).get(f,[]),maxlen=self.window) for f in FAMILIES}
        self.grad=deque(s.get("grad",[]),maxlen=max(64,self.window))
        self.pending=[tuple(x) for x in s.get("pending",[])]
        for k in ("current_horizon","switch_update","gate_start","stability_factor",
                  "cooldown_until","bad_streak","good_streak","backoff_count",
                  "recover_count","last_control_update","epoch_switched",
                  "prev_comparable_epoch","last_epoch_guard"):
            if k in s: setattr(self,k,s[k])
        self.epoch_values={f:list(s.get("epoch_values",{}).get(f,[])) for f in FAMILIES}

    def observe_horizon(self, horizon, completed):
        if not self.enabled or horizon is None:
            return False
        h=float(horizon)
        if self.current_horizon is None:
            self.current_horizon=h
            return False
        if abs(h-float(self.current_horizon)) < 1e-12:
            return False
        self.current_horizon=h
        self.switch_update=int(completed)
        self.gate_start=int(completed)
        for q in self.family_logs.values(): q.clear()
        self.grad.clear()
        # Drop observations accumulated since the last sync under the OLD horizon.
        # Otherwise the first post-switch control window is contaminated by a few
        # samples from a different curriculum distribution.
        self.pending.clear()
        self.bad_streak=self.good_streak=0
        self.epoch_switched=True
        self.prev_comparable_epoch=None
        return True

    def curriculum_gate(self, completed):
        if not self.enabled or self.gate_start is None:
            return 1.0
        d=max(0,int(completed)-int(self.gate_start))
        if d>=self.gate_recover:
            return 1.0
        a=d/max(1,self.gate_recover)
        return self.gate_min+(1.0-self.gate_min)*a

    def effective_factor(self, completed):
        return float(self.stability_factor*self.curriculum_gate(completed)) if self.enabled else 1.0

    def observe_local(self, family, ras_loss_unbiased, grad_norm):
        if not self.enabled or family not in FAMILIES:
            return
        try:
            lv=math.log(max(float(ras_loss_unbiased),1e-12))
            gv=float(grad_norm)
        except Exception:
            return
        if math.isfinite(lv) and math.isfinite(gv):
            self.pending.append((family,lv,gv))

    def _ingest(self, rows):
        for fam,lv,gv in rows:
            if fam not in FAMILIES or not (math.isfinite(lv) and math.isfinite(gv)):
                continue
            self.family_logs[fam].append(float(lv))
            self.epoch_values[fam].append(float(lv))
            self.grad.append(float(gv))

    def stats(self):
        fam={}
        mads=[]; trends=[]
        for f in FAMILIES:
            xs=list(self.family_logs[f])
            if len(xs)<self.min_points:
                fam[f]={"n":len(xs),"median_log":None,"mad_log":None,"trend_log":None}
                continue
            med=_median(xs); mad=_mad_log(xs); h=max(1,len(xs)//2)
            trend=_median(xs[h:])-_median(xs[:h])
            fam[f]={"n":len(xs),"median_log":med,"mad_log":mad,"trend_log":trend}
            mads.append(mad);trends.append(trend)
        return {"families":fam,
                "max_mad_log":max(mads) if mads else None,
                "max_trend_log":max(trends) if trends else None,
                "p95_grad":_p95(list(self.grad))}

    def _can_decide(self, completed):
        if int(completed)<self.warmup_updates: return False
        if self.switch_update is not None and int(completed)-int(self.switch_update)<self.grace: return False
        return True

    def _backoff(self, completed, reason):
        old=self.stability_factor
        self.stability_factor=max(self.min_factor,self.stability_factor*self.backoff_mult)
        changed=self.stability_factor < old-1e-15
        if changed:
            self.backoff_count+=1
            self.cooldown_until=int(completed)+self.cooldown
        self.bad_streak=self.good_streak=0
        return changed,reason

    def _decide(self, completed):
        st=self.stats(); action="hold"; reason="none"
        if not self._can_decide(completed):
            return action,"warmup_or_causal_grace",st
        mad=st["max_mad_log"]; trend=st["max_trend_log"]; pg=st["p95_grad"]
        enough=mad is not None and trend is not None and pg is not None
        if not enough:
            return action,"insufficient_family_points",st
        bad=((mad>self.mad_trigger and trend>self.trend_trigger) or
             (pg>self.grad_trigger and trend>self.grad_trend_trigger))
        good=(mad<self.recover_mad and trend<=0.0 and pg<self.recover_grad)
        if bad:
            self.bad_streak+=1; self.good_streak=0
        elif good:
            self.good_streak+=1; self.bad_streak=0
        else:
            self.bad_streak=self.good_streak=0
        if self.bad_streak>=self.trigger_streak_needed and int(completed)>=self.cooldown_until:
            changed,reason=self._backoff(completed,"sustained_robust_loss_or_gradient_instability")
            action="backoff" if changed else "floor_hold"
        elif self.good_streak>=self.recover_streak_needed and int(completed)>=self.cooldown_until and self.stability_factor<1.0:
            old=self.stability_factor
            self.stability_factor=min(1.0,self.stability_factor*self.recover_mult)
            if self.stability_factor>old+1e-15:
                self.recover_count+=1; self.cooldown_until=int(completed)+self.cooldown
                action="recover";reason="sustained_stability"
            self.bad_streak=self.good_streak=0
        return action,reason,st

    def sync(self, completed, rank=0, world=1, dist_mod=None, force=False, decide=True):
        if not self.enabled:
            return None
        due=force or (int(completed)>0 and int(completed)%self.interval==0)
        if not due:
            return None
        local=list(self.pending); self.pending.clear()
        if world>1:
            gathered=[None]*world
            dist_mod.all_gather_object(gathered,local)
            rows=[x for part in gathered for x in (part or [])] if rank==0 else []
        else:
            rows=local
        event=None
        if rank==0:
            self._ingest(rows)
            action,reason,st=("hold","sync_only",self.stats()) if not decide else self._decide(completed)
            self.last_control_update=int(completed)
            event={"event":"stability_controller","updates_completed":int(completed),
                   "action":action,"reason":reason,"stability_factor":float(self.stability_factor),
                   "curriculum_gate":float(self.curriculum_gate(completed)),
                   "effective_factor":float(self.effective_factor(completed)),
                   "backoff_count":int(self.backoff_count),"recover_count":int(self.recover_count),
                   **st}
            state=self.state_dict()
        else:
            state=None
        if world>1:
            box=[state]
            dist_mod.broadcast_object_list(box,src=0)
            if rank!=0:self.load_state_dict(box[0])
        return event if rank==0 else None

    def epoch_close(self, epoch, completed, rank=0, world=1, dist_mod=None):
        if not self.enabled:
            return {}
        self.sync(completed,rank,world,dist_mod,force=True,decide=False)
        summary=None
        if rank==0:
            cur={}
            allv=[]
            for f in FAMILIES:
                xs=self.epoch_values[f]
                cur[f]={"n":len(xs),"median_log":_median(xs),"mad_log":_mad_log(xs)}
                allv.extend(xs)
            comparable=not self.epoch_switched
            action="hold"; ratios={}
            if comparable and self.prev_comparable_epoch is not None:
                for f in FAMILIES:
                    a=cur[f]["median_log"]; b=self.prev_comparable_epoch["families"][f]["median_log"]
                    ratios[f]=math.exp(a-b) if a is not None and b is not None else None
                om=_median(allv); prev_om=self.prev_comparable_epoch.get("overall_median_log")
                overall_ratio=math.exp(om-prev_om) if om is not None and prev_om is not None else None
                r=[ratios.get(f) for f in FAMILIES]
                bad=((all(x is not None and x>1.15 for x in r)) or
                     (any(x is not None and x>1.30 for x in r) and overall_ratio is not None and overall_ratio>1.15))
                if bad and int(completed)>=self.cooldown_until:
                    changed,_=self._backoff(completed,"epoch_family_median_regression")
                    if changed:action="backoff"
            else:
                overall_ratio=None
            om=_median(allv)
            if comparable:
                self.prev_comparable_epoch={"epoch":int(epoch),"families":cur,"overall_median_log":om}
            else:
                self.prev_comparable_epoch=None
            summary={"epoch":int(epoch),"comparable":bool(comparable),"families":cur,
                     "overall_median_log":om,"ratios_vs_prev":ratios,
                     "overall_ratio_vs_prev":overall_ratio,"action":action,
                     "stability_factor":float(self.stability_factor),
                     "backoff_count":int(self.backoff_count)}
            self.last_epoch_guard=summary
            self.epoch_values={f:[] for f in FAMILIES}
            self.epoch_switched=False
            state=self.state_dict()
        else:
            state=None
        if world>1:
            box=[state]
            dist_mod.broadcast_object_list(box,src=0)
            if rank!=0:self.load_state_dict(box[0])
        return summary if rank==0 else {}

def normalize_stability_controller(raw):
    raw=dict(raw or {})
    if not raw.get("enabled",False):
        return {"enabled":False}
    defaults={
      "enabled":True,"control_every":25,"window_per_family":64,"min_points_per_family":16,
      "mad_trigger":1.20,"trend_trigger_log":math.log(1.50),"grad_p95_trigger":20.0,
      "grad_trend_trigger_log":math.log(1.20),"trigger_streak":2,
      "recover_mad":0.65,"recover_grad_p95":12.0,"recover_streak":3,
      "backoff_mult":0.5,"recover_mult":1.1,"min_factor":0.25,
      "cooldown_updates":100,"causal_grace_updates":150,
      "curriculum_gate_min":0.5,"curriculum_gate_recover_updates":150}
    out={**defaults,**raw}
    for k in ("control_every","window_per_family","min_points_per_family","trigger_streak",
              "recover_streak","cooldown_updates","causal_grace_updates","curriculum_gate_recover_updates"):
        out[k]=int(out[k])
    for k in ("mad_trigger","trend_trigger_log","grad_p95_trigger","grad_trend_trigger_log",
              "recover_mad","recover_grad_p95","backoff_mult","recover_mult","min_factor","curriculum_gate_min"):
        out[k]=float(out[k])
    if out["control_every"]<1 or out["window_per_family"]<8 or out["min_points_per_family"]<4:
        raise ValueError("invalid stability_controller interval/window")
    if not 0<out["min_factor"]<=1 or not 0<out["backoff_mult"]<=1 or out["recover_mult"]<1:
        raise ValueError("invalid stability_controller LR factors")
    if not 0<out["curriculum_gate_min"]<=1:
        raise ValueError("invalid curriculum_gate_min")
    return out
