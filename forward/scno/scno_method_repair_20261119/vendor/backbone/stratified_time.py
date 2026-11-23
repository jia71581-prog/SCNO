"""Time-stratum coverage within an expanding curriculum window."""

import numpy as np


class StratifiedTimeSampler:
    def __init__(self, curriculum, records, strata):
        self.plan = curriculum
        self.records = records
        if type(strata) is not int or strata < 1 or curriculum.queries % strata:
            raise ValueError("time query count must be divisible by the number of strata")
        self.per_stratum = curriculum.queries // strata
        self.groups = [np.array_split(np.arange(size), strata) for size in curriculum.prefix_lengths]
        if min(map(len, self.groups[0])) < self.per_stratum:
            raise ValueError("the initial window cannot supply its distinct time queries")
        self.first_epoch = []
        self.first_epoch_records = []
        for start in curriculum.starts:
            epoch, offset = divmod(int(start), records)
            self.first_epoch.append(epoch)
            order = curriculum.record_order(epoch, records)
            included = np.zeros(records, dtype=bool)
            included[order[offset:]] = True
            self.first_epoch_records.append(included)

    def sample(self, update, record):
        if not 0 <= update < self.plan.total_updates:
            raise ValueError("update is outside the configured curriculum")
        stage = int(np.searchsorted(self.plan.starts, update, side="right") - 1)
        epoch = update // self.records
        first_epoch = self.first_epoch[stage]
        visit = 0
        if epoch > first_epoch:
            visit = epoch - first_epoch - 1 + int(self.first_epoch_records[stage][record])
        selected, weights = [], []
        quota = self.per_stratum
        size = self.plan.prefix_lengths[stage]
        for band, times in enumerate(self.groups[stage]):
            cycle_length = (len(times) + quota - 1) // quota
            cycle, local_visit = divmod(visit, cycle_length)
            seed = np.random.SeedSequence([self.plan.seed, stage, record, band, cycle, len(self.plan.times_s)])
            order = np.random.default_rng(seed).permutation(times)
            indices = (local_visit * quota + np.arange(quota)) % len(times)
            selected.append(np.sort(order[indices]))
            weights.append(np.full(quota, len(times) / (size * quota)))
        return stage, np.concatenate(selected), np.concatenate(weights)
