"""Execution-only activation checkpointing; no parameter/state-dict changes."""
import torch
from torch.utils.checkpoint import checkpoint


def enable_block_checkpointing(model):
    names = []
    for name, module in model.named_modules():
        if not (name == 'refiner.dg' or
                name.startswith(('refiner.dynamic.', 'refiner.up.')) and name.count('.') == 2):
            continue
        if hasattr(module, '_batch400_original_forward'):
            names.append(name)
            continue
        original = module.forward

        def wrap(function):
            def execute(*args, **kwargs):
                if torch.is_grad_enabled():
                    return checkpoint(function, *args, use_reentrant=False,
                                      preserve_rng_state=True, **kwargs)
                return function(*args, **kwargs)
            return execute

        module._batch400_original_forward = original
        module.forward = wrap(original)
        names.append(name)
    if len(names) != 10:
        raise ValueError(f'Expected 5 dynamic, 4 up and 1 DG block; got {names}')
    return names
