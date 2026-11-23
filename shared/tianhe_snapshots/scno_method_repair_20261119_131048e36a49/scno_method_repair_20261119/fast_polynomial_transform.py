"""Group fixed local polynomial transforms by axis, without broadcasting Q.

The inherited bases, gather/order/masks and SS recursion are unchanged. Each
axis uses gx or gz large GEMMs instead of broadcasting the same tiny basis
over every record/time/channel/block. This is a runtime scheduling policy;
install/remove neither registers state nor replaces any saved buffer.
"""

from types import MethodType

import torch


def _right_x(field, basis):
    batch, channels, gz, gx, mz, mx = field.shape
    rows = field.permute(3, 0, 1, 2, 4, 5).reshape(gx, batch * channels * gz * mz, mx)
    projected = torch.bmm(rows, basis)
    return projected.reshape(gx, batch, channels, gz, mz, mx).permute(1, 2, 3, 0, 4, 5)


def _left_z(basis, field):
    batch, channels, gz, gx, mz, mx = field.shape
    columns = field.permute(2, 4, 0, 1, 3, 5).reshape(gz, mz, batch * channels * gx * mx)
    projected = torch.bmm(basis, columns)
    return projected.reshape(gz, mz, batch, channels, gx, mx).permute(2, 3, 0, 4, 1, 5)


def fast_level_analyze(level, field):
    """Exact fixed analysis layout: right Qx, then left Qz transpose."""
    batch, channels = field.shape[:2]
    blocks = field.flatten(-2).index_select(-1, level.gather)
    blocks = blocks.reshape(batch, channels, level.gz, level.gx, level.gz_modes, level.gx_modes)
    modes = _left_z(level.qz[:, 0].transpose(-2, -1), _right_x(blocks, level.qx[0]))
    modes = modes.flatten(-2).index_select(-1, level.order)
    coarse = modes[..., :16].reshape(batch, channels, level.gz, level.gx, 4, 4)
    coarse = coarse.transpose(-3, -2).reshape(batch, channels, level.gz * 4, level.gx * 4)
    return coarse, modes


def fast_level_synthesize(level, coarse, detail):
    """Exact fixed synthesis layout: right Qx transpose, then left Qz."""
    batch, channels = coarse.shape[:2]
    ss = coarse.reshape(batch, channels, level.gz, 4, level.gx, 4)
    ss = ss.transpose(-3, -2).flatten(-2)
    modes = torch.cat((ss, detail.masked_fill(~level.valid_detail, 0)), dim=-1)
    modes = modes.index_select(-1, level.inverse_order)
    modes = modes.reshape(batch, channels, level.gz, level.gx, level.gz_modes, level.gx_modes)
    blocks = _left_z(level.qz[:, 0], _right_x(modes, level.qx[0].transpose(-2, -1)))
    field = blocks.transpose(-3, -2).flatten(-4)
    return field.index_select(-1, level.unpad).reshape(batch, channels, level.nz, level.nx)


class FastPolynomialPolicy:
    """A removable scheduling policy over the existing level objects/buffers."""

    def __init__(self, pyramid, *, analyze=True, synthesize=True):
        if not analyze and not synthesize:
            raise ValueError("enable at least one transform direction")
        levels = tuple(pyramid.levels)
        required = ("qz", "qx", "gather", "unpad", "order", "inverse_order", "valid_detail",
                    "gz", "gx", "gz_modes", "gx_modes", "modes", "nz", "nx")
        methods = [(name, function) for enabled, name, function in (
            (analyze, "analyze", fast_level_analyze), (synthesize, "synthesize", fast_level_synthesize)) if enabled]
        if not levels or any(not all(hasattr(level, key) for key in required) for level in levels):
            raise TypeError("install only on the inherited local polynomial levels with their complete fixed layout")
        for level in levels:
            for name, function in methods:
                if getattr(getattr(level, name), "__func__", None) is function:
                    raise RuntimeError("a fast policy is already installed; remove it before reinstalling")
        self._changes, self.removed = [], False
        for level in levels:
            for name, function in methods:
                original = level.__dict__.get(name)
                had_instance_value = name in level.__dict__
                self._changes.append((level, name, function, had_instance_value, original))
                setattr(level, name, MethodType(function, level))
        self.receipt = {"schema": "SCNO_FIXED_POLYNOMIAL_GROUPED_AXES_V1",
            "analysis_enabled": bool(analyze), "synthesis_enabled": bool(synthesize),
            "axis_bmm_batch_counts": [{"z": level.gz, "x": level.gx} for level in levels],
            "multiplication_order": "right_x_then_left_z",
            "basis_buffers_gather_padding_order_masks_and_SS_recursion": "inherited_unchanged",
            "state_dict_registration_changed": False, "measured_gpu_speedup": False}

    def remove(self):
        if self.removed:
            return
        if any(getattr(getattr(level, name), "__func__", None) is not function
               for level, name, function, _, _ in self._changes):
            raise RuntimeError("another runtime strategy replaced this policy; refuse to overwrite it")
        for level, name, _, had_instance_value, original in reversed(self._changes):
            if had_instance_value:
                setattr(level, name, original)
            else:
                delattr(level, name)
        self.removed = True

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.remove()


def install_fast_polynomial(pyramid, *, analyze=True, synthesize=True):
    return FastPolynomialPolicy(pyramid, analyze=analyze, synthesize=synthesize)


def remove_fast_polynomial(policy):
    policy.remove()
