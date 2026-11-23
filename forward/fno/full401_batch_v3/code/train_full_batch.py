"""Full-state FNO continuation with one complete401-frame prediction per batch."""
from __future__ import annotations
from scno_release_paths import local_asset as _scno_local_asset

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import scno_posix_lock as fcntl
import gc
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'large_batch_v2/code'))
from fast_data import DirectCorpus, base
from full_time import backward_full401

REVISION = Path(__file__).resolve().parents[1]
PARENT_RUNTIME = Path(_scno_local_asset('runtime-fno/fno401_largebatch'))
STOP = False


def valid_progress(progress, records):
    assert 0 <= progress['epoch'] <= 100 and 0 <= progress['offset'] <= records
    assert progress['exposures'] == progress['epoch'] * records + progress['offset'] <= 100 * records
    if progress['epoch'] == 100:
        assert progress['offset'] == 0


class Checkpoints:
    def __init__(self, run, execution):
        self.run, self.execution = run, execution
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.durable = dict(updates=0, exposures=0)

    def flush(self):
        if self.future is not None:
            self.future.result()
            self.future = None

    def persist(self, state, best=False):
        tmp = self.run / 'latest.pt.tmp'
        torch.save(state, tmp)
        loaded = torch.load(tmp, map_location='cpu', weights_only=False)
        assert base.equal(state, loaded), 'Complete asynchronous checkpoint readback differs'
        os.replace(tmp, self.run / 'latest.pt')
        if best:
            # Use exactly the same CPU snapshot and serialized payload.
            import shutil
            shutil.copyfile(self.run / 'latest.pt', self.run / 'best.pt.tmp')
            os.replace(self.run / 'best.pt.tmp', self.run / 'best.pt')
        self.durable = {k: state['progress'][k] for k in ['updates', 'exposures']}
        base.write(self.run / 'CHECKPOINT.json', dict(status='PASS', utc=base.utc(), **self.durable,
                   epoch=state['progress']['epoch'], full_model_Adam_RNG_cursor_readback=True))

    def save(self, model, opt, progress, config, best=False, force=False):
        # At most one immutable snapshot is pending; never queue an unbounded history.
        self.flush()
        state = base.snapshot(model, opt, progress, config)
        state['execution'] = self.execution
        if force:
            self.persist(state, best)
        else:
            self.future = self.pool.submit(self.persist, state, best)
        return state


def run(args):
    out = args.run / args.case
    out.mkdir(parents=True, exist_ok=True)
    owner = (out / 'owner.lock').open('a+')
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    worker = dict(pid=os.getpid(), proc_start=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
                  started_utc=base.utc(), command=sys.argv, cwd=os.getcwd())
    base.write(out / 'WORKER.json', worker)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    parent_path = args.parent_runtime / args.case / 'latest.pt'
    parent = torch.load(parent_path, map_location='cpu', weights_only=False, mmap=True)
    parent_progress = copy.deepcopy(parent['progress'])
    config = parent['config']
    assert config['epochs'] == 100 and config['normalization'] == base.PROFILE
    valid_progress(parent_progress, config['records'])
    execution = dict(revision='FNO401_FULL_TIME_BATCH_V3', requested_batch=args.batch, profiled_max_batch=args.batch,
                     parent_checkpoint=str(parent_path), parent_sha256=base.sha(parent_path),
                     parent_progress=parent_progress, code_sha256={n: base.sha(REVISION / 'code' / n)
                        for n in ['train_full_batch.py', 'full_time.py']},
                     frozen_direct_reader=str(Path(__file__).resolve().parents[2] / 'large_batch_v2/code/fast_data.py'),
                     frozen_direct_reader_sha256=base.sha(Path(__file__).resolve().parents[2] / 'large_batch_v2/code/fast_data.py'),
                     original_model_and_optimizer_retained=True, original_total_exposure_budget_retained=True,
                     output_time_chunk=None, temporal_chunks=1, all401_frames_supervised=True,
                     prediction_calls_per_update=1, backward_calls_per_update=1,
                     target_memory='one complete401 transfer into owned residual buffer; no temporal slicing',
                     checkpoint_every_updates=64, bounded_async_checkpoints=True,
                     data_reader='direct verified source file; exact native stride2; CPU batch prefetch',
                     original_config=config, local_only=True)
    source = out / 'latest.pt'
    if source.exists():
        state = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
        assert state['execution'] == execution
    else:
        state = parent
    model, opt, progress = base.restore(args.case, state, 'cuda:0')
    assert base.equal(base.rng(), state['rng']), 'Complete RNG restoration differs'
    initial_equal = (base.equal(model.state_dict(), state['model'])
                     and base.equal(opt.state_dict(), state['optimizer'])
                     and progress == state['progress'])
    assert initial_equal
    new_run = not source.exists()
    if new_run:
        progress['actual_batch'] = args.batch
    valid_progress(progress, config['records'])
    base.write(out / 'EXECUTION.json', execution)
    base.write(out / 'ZERO_UPDATE_MIGRATION.json', dict(status='PASS', utc=base.utc(),
               full_model_Adam_RNG_cursor_equal=True, inherited_progress=state['progress'],
               runtime_batch=progress['actual_batch'], parent_sha256=execution['parent_sha256'],
               reset_parameters=0, reset_optimizer_slots=0, exact_remaining_record_budget=100 * config['records'] - progress['exposures']))
    data = DirectCorpus(args.case, config['records'])
    weights = data.weights('cuda:0')
    checkpoints = Checkpoints(out, execution)
    if new_run:
        checkpoints.save(model, opt, progress, config, force=True)
        old_best = args.parent_runtime / args.case / 'best.pt'
        if old_best.exists():
            import shutil
            shutil.copyfile(old_best, out / 'inherited_best.pt')
            if not (out / 'best.pt').exists():
                (out / 'best.pt').symlink_to('inherited_best.pt')
    del state, parent
    def stop(signum, frame):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    def status(phase, **extra):
        base.write(out / 'STATUS.json', dict(status=phase, utc=base.utc(), **worker, **progress,
                   runtime_updates=progress['updates'] - parent_progress['updates'],
                   runtime_exposures=progress['exposures'] - parent_progress['exposures'],
                   durable_updates=checkpoints.durable['updates'], durable_exposures=checkpoints.durable['exposures'],
                   target_epochs=100, target_exposures=100 * config['records'],
                   cache_records=len(data.cache), time_frames=401, time_dt_s=.0025,
                   pressure_range=[-1, 1], temporal_chunks=1,
                   prediction_calls_per_update=1, backward_calls_per_update=1, **extra))
    def cpu_batch(rows):
        records = [data.record(row) for row in rows]
        x = torch.from_numpy(np.stack([r[0] for r in records])).pin_memory()
        y = torch.from_numpy(np.stack([r[1] for r in records])).pin_memory()
        return x, y
    def rows_at(order, offset, batch):
        return [data.rows[int(i)] for i in order[offset:min(len(order), offset + batch)]]
    def panel():
        rows = [data.rows[0], data.rows[-1]]
        if args.case == 'ul':
            rows = [next(r for r in data.rows if data.families[r] == n) for n in ['uniform', 'layered']]
        total = 0.
        for row in rows:
            x, y = data.batch([row], 'cuda:0')
            with torch.no_grad():
                prediction = model(x)
                total += float((prediction - y).square().sum(dtype=torch.float64)) * base.PROFILE['pressure_half_range_pa'] ** 2
            del x, y, prediction
        return float(np.sqrt(total / sum(data.energy[r] for r in rows)))
    def step(x, y):
        opt.zero_grad(set_to_none=True)
        try:
            loss = backward_full401(model, x, y, weights)
            assert bool(torch.isfinite(loss))
            assert all(p.grad is not None for p in model.parameters())
            # Finite total gradient norm also rejects any NaN/Inf gradient.
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., foreach=False, error_if_nonfinite=True)
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            return None
        # Populated inherited Adam slots avoid initialization-time storage surprises.
        opt.step()
        torch.cuda.synchronize()
        return float(loss), float(norm)
    pool = ThreadPoolExecutor(max_workers=1)
    future = None
    started = time.monotonic()
    status('RESTORED_PREFETCHING')
    try:
        while progress['epoch'] < 100 and not STOP:
            order = np.random.default_rng(np.random.SeedSequence([20261028, progress['epoch']])).permutation(len(data.rows))
            rows = rows_at(order, progress['offset'], progress['actual_batch'])
            future = pool.submit(cpu_batch, rows)
            while progress['offset'] < len(order) and not STOP:
                request = out / 'BATCH_REQUEST.json'
                if request.exists():
                    wanted = int(json.loads(request.read_text())['batch'])
                    assert 1 <= wanted <= execution['profiled_max_batch']
                    if wanted != progress['actual_batch']:
                        future.result()
                        base.write(out / 'LAST_BATCH_CHANGE.json', dict(utc=base.utc(), old=progress['actual_batch'], new=wanted,
                                   exposures=progress['exposures']))
                        progress['actual_batch'] = wanted
                        rows = rows_at(order, progress['offset'], wanted)
                        future = pool.submit(cpu_batch, rows)
                status('TRAINING', current_rows=rows)
                begin = time.monotonic()
                xc, yc = future.result()
                count = len(rows)
                next_offset = progress['offset'] + count
                next_rows = rows_at(order, next_offset, progress['actual_batch'])
                future = pool.submit(cpu_batch, next_rows) if next_rows else None
                x = xc.to('cuda:0', non_blocking=True)
                y = yc
                del xc, yc
                outcome = step(x, y)
                if outcome is None:
                    if count == 1:
                        raise RuntimeError('Even physical batch1 exceeds currently available GPU memory')
                    if future is not None:
                        future.result()
                    del x, y
                    gc.collect()
                    torch.cuda.empty_cache()
                    progress['actual_batch'] = max(1, count // 2)
                    rows = rows_at(order, progress['offset'], progress['actual_batch'])
                    future = pool.submit(cpu_batch, rows)
                    base.append(out / 'OOM_RETRIES.jsonl', dict(utc=base.utc(), unchanged_offset=progress['offset'],
                                new_batch=progress['actual_batch'], reason='Concurrent GPU memory availability changed'))
                    continue
                value, norm = outcome
                progress['offset'] = next_offset
                progress['updates'] += 1
                progress['exposures'] += count
                valid_progress(progress, config['records'])
                runtime_updates = progress['updates'] - parent_progress['updates']
                record = dict(utc=base.utc(), **progress, runtime_updates=runtime_updates, batch=count,
                              loss=value, gradient_norm=norm, seconds=time.monotonic() - begin,
                              cache_records=len(data.cache), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                              reader='direct_source_prefetch', temporal_chunks=1,
                              prediction_calls=1, backward_calls=1,
                              prediction_shape=[count, 401, 201, 201])
                base.append(out / 'metrics.jsonl', record)
                print(json.dumps(record), flush=True)
                del y
                if runtime_updates == 8 and not (out / 'CUDA_RESTORE8.json').exists():
                    state = checkpoints.save(model, opt, progress, config, force=True)
                    with torch.no_grad():
                        expected = model(x[:1]).detach().cpu()
                    del model, opt
                    gc.collect()
                    torch.cuda.empty_cache()
                    model, opt, progress = base.restore(args.case, state, 'cuda:0')
                    assert base.equal(base.rng(), state['rng'])
                    with torch.no_grad():
                        actual = model(x[:1]).detach().cpu()
                    assert torch.equal(expected, actual)
                    base.write(out / 'CUDA_RESTORE8.json', dict(status='PASS', utc=base.utc(), runtime_updates=8,
                               global_updates=progress['updates'], populated_Adam_slots=len(opt.state),
                               complete_model_Adam_RNG_cursor_exact=True, full401_prediction_bitwise=True))
                    del state, expected, actual
                del x
                if runtime_updates == 9 and not (out / 'LAUNCH_VERIFIED.json').exists():
                    checkpoints.save(model, opt, progress, config, force=True)
                    base.write(out / 'LAUNCH_VERIFIED.json', dict(status='PASS', utc=base.utc(),
                               actual_restored_object_update9=True, global_updates=progress['updates'],
                               total_exposures=progress['exposures'], batch=progress['actual_batch'],
                               temporal_chunks=1, prediction_calls_per_update=1, all401_gradients_finite=True))
                elif progress['updates'] % 64 == 0 and progress['offset'] != len(order):
                    checkpoints.save(model, opt, progress, config)
                rows = next_rows
                status('TRAINING')
            if future is not None:
                future.result()
                future = None
            if progress['offset'] == len(order):
                progress['epoch'] += 1
                progress['offset'] = 0
                valid_progress(progress, config['records'])
                rel = panel()
                improved = (progress['best_train_panel_relative_l2'] is None or rel < progress['best_train_panel_relative_l2'])
                if improved:
                    progress['best_train_panel_relative_l2'] = rel
                checkpoints.save(model, opt, progress, config, best=improved)
                base.append(out / 'epochs.jsonl', dict(utc=base.utc(), **progress, train_panel_relative_l2=rel,
                            train_records=config['records'], exact_once_per_epoch=True))
        checkpoints.save(model, opt, progress, config, force=True)
        phase = 'STOPPED_BY_USER' if STOP else 'COMPLETE'
        status(phase, elapsed_seconds=time.monotonic() - started)
        base.write(out / 'RESULT.json', dict(status=phase, utc=base.utc(), **progress,
                   inherited_progress=parent_progress, original_target_epochs=100,
                   original_target_exposures=config['records'] * 100, heldout_evaluation_executed=False))
    except BaseException as error:
        status('FAILED', error=repr(error))
        base.write(out / 'ERROR.json', dict(error=repr(error), traceback=traceback.format_exc(), utc=base.utc()))
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        checkpoints.flush()
        checkpoints.pool.shutdown(wait=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=['ul'], required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--parent-runtime', type=Path, default=PARENT_RUNTIME)
    run(parser.parse_args())
