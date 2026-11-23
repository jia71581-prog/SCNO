"""Synchronous UL data parallelism across explicitly assigned Slurm ranks."""

from datetime import timedelta
from itertools import groupby
import os

import torch
from torch import distributed as dist


class RankRuntime:
    def __init__(self, requested_device):
        self.size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        if self.size < 1 or not 0 <= self.rank < self.size:
            raise ValueError("Invalid RANK or WORLD_SIZE")
        self.primary = self.rank == 0
        self.active = self.size > 1
        self.device = torch.device(requested_device)
        if self.device.type == "cuda":
            local = int(os.environ["LOCAL_RANK"]) if self.active else self.device.index
            self.device = torch.device("cuda", 0 if local is None else local)
            torch.cuda.set_device(self.device)
        if self.active:
            options = {"device_id": self.device} if self.device.type == "cuda" else {}
            dist.init_process_group(backend="nccl" if options else "gloo", init_method="env://",
                                    timeout=timedelta(minutes=25), **options)

    def close(self):
        if self.active:
            dist.destroy_process_group()

    def action(self, value):
        if not self.active:
            return int(value)
        shared = torch.tensor(value, device=self.device, dtype=torch.int32)
        dist.all_reduce(shared, op=dist.ReduceOp.MAX)
        return int(shared.item())

    def stats(self, values, *, maximum=False):
        if not self.active:
            return values
        shared = torch.tensor(values, device=self.device, dtype=torch.float64)
        dist.all_reduce(shared, op=dist.ReduceOp.MAX if maximum else dist.ReduceOp.SUM)
        return shared.tolist()

    def on_primary(self, function, *, share_result=False):
        result, problem = None, None
        if self.primary:
            try:
                result = function()
            except Exception as error:
                problem = error
        if self.action(problem is not None):
            if problem is not None:
                raise problem
            raise RuntimeError("Rank 0 failed during a shared training operation")
        if self.active and share_result:
            shared = [result]
            dist.broadcast_object_list(shared, src=0)
            return shared[0]
        return result

    def on_each(self, function):
        result, problem = None, None
        try:
            result = function()
        except Exception as error:
            problem = error
        if self.action(problem is not None):
            if problem is not None:
                raise problem
            raise RuntimeError("Another rank failed during a local training operation")
        return result

    def gather(self, value):
        if not self.active:
            return [value]
        combined = [None] * self.size if self.primary else None
        dist.gather_object(value, combined, dst=0)
        return combined

    @torch.no_grad()
    def share_initial_parameters(self, model):
        """Establish identical rank replicas before the first forward pass."""
        if not self.active:
            return
        parameters = sorted(model.parameters(), key=lambda parameter: str(parameter.dtype))
        for _, group in groupby(parameters, key=lambda parameter: parameter.dtype):
            bucket = list(group)
            flat = torch.cat([parameter.reshape(-1) for parameter in bucket])
            dist.broadcast(flat, src=0)
            for parameter, synced in zip(bucket, flat.split([p.numel() for p in bucket])):
                parameter.copy_(synced.view_as(parameter))

    @torch.no_grad()
    def reduce_gradients(self, parameters):
        """Sum globally weighted gradients after all ranks finish local backward.

        Delaying communication until the all-rank OOM check lets every rank
        retry the same sampled update without entering a half-formed collective.
        Preserve None gradients when a parameter is unused on all ranks.
        """
        if not self.active:
            return
        used = torch.tensor([parameter.grad is not None for parameter in parameters],
                            device=self.device, dtype=torch.uint8)
        dist.all_reduce(used, op=dist.ReduceOp.MAX)
        selected = [parameter for parameter, keep in zip(parameters, used.tolist()) if keep]
        selected.sort(key=lambda parameter: str(parameter.dtype))
        for _, group in groupby(selected, key=lambda parameter: parameter.dtype):
            bucket = list(group)
            flat = torch.cat([parameter.grad.reshape(-1) if parameter.grad is not None
                              else parameter.new_zeros(parameter.numel()) for parameter in bucket])
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            for parameter, synced in zip(bucket, flat.split([p.numel() for p in bucket])):
                gradient = synced.view_as(parameter)
                if parameter.grad is None:
                    parameter.grad = gradient.clone()
                else:
                    parameter.grad.copy_(gradient)
