"""Bounded CPU reads for the next accepted SCNO training update."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter


@dataclass(frozen=True)
class LocalPressureBatch:
    samples: tuple
    read_s: float
    wait_s: float
    background: bool


class PressureBatchPrefetch:
    """Keep one pending batch per rank while maintaining the committed sampler cursor."""

    def __init__(self, dataset, sampler, rank, world_size, *, enabled):
        self.dataset = dataset
        self.sampler = sampler
        self.rank = rank
        self.world_size = world_size
        self.enabled = enabled
        self._executor = None
        self._pending = None

    def __enter__(self):
        if self.enabled:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scno-pressure-read")
        return self

    def __exit__(self, *_):
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        self._pending = None

    def _read(self, draws):
        start = perf_counter()
        samples = tuple(self.dataset.read(record_id, slots)
                        for record_id, slots in draws[self.rank::self.world_size])
        return samples, perf_counter() - start

    def schedule(self):
        if self._executor is None:
            return
        if self._pending is not None:
            raise RuntimeError("only one pressure batch may be pending")
        before = self.sampler.state_dict()
        try:
            draws = self.sampler.draw()
            after = self.sampler.state_dict()
        finally:
            self.sampler.load_state_dict(before)
        self._pending = (draws, after, self._executor.submit(self._read, draws))

    def take(self):
        if not self.enabled:
            return self.sampler.draw(), None
        start = perf_counter()
        if self._pending is None:
            draws = self.sampler.draw()
            samples, read_s = self._read(draws)
            background = False
        else:
            draws, state, future = self._pending
            if self.sampler.updates_drawn + 1 != state["updates_drawn"]:
                raise RuntimeError("pending pressure batch does not follow the committed sampler cursor")
            samples, read_s = future.result()
            self.sampler.load_state_dict(state)
            self._pending = None
            background = True
        return draws, LocalPressureBatch(samples, read_s, perf_counter() - start, background)
