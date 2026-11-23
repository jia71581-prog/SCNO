from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FORMULA = "-9.6*f0*(0.6*f0*t-1)*exp(-8*(0.6*f0*t-1)^2)*delta(x-x0,z-z0)"
LATEX_FORMULA = (
    r"-9.6f_0(0.6f_0t-1)\exp[-8(0.6f_0t-1)^2]\delta(x-x_0,z-z_0)"
)


@dataclass(frozen=True)
class BilinearPointSource:
    indices: np.ndarray
    weights: np.ndarray
    source_map: np.ndarray
    delta_h: np.ndarray
    x0_m: float
    z0_m: float


def source_formula_string() -> str:
    return FORMULA


def source_formula_latex() -> str:
    return LATEX_FORMULA


def source_time_function(t_s, f0_hz, f0_weights=None):
    """Ricker 源时间函数(支持单频与多频叠加两种模式)。

    参数
    ----
    t_s : array-like
        采样时刻(秒)。
    f0_hz : float 或 一维序列(list/tuple/ndarray)
        - 标量(旧接口):单频 Ricker,返回值与历史版本逐 bit 一致(向后兼容,
          运算顺序未改动)。
        - 一维序列(新接口,建议 2~3 个不同 f0):逐频率生成 Ricker 后按权重
          求和,得到宽带源(5b 扩频段实验用)。
    f0_weights : None 或 与 f0_hz 序列等长的一维序列
        多频叠加时每个频率分量的权重;None 表示各分量权重均为 1.0(纯叠加)。
        单频(标量 f0_hz)模式下必须为 None,否则抛错——避免静默忽略参数。

    说明
    ----
    - 单个元素的序列 [f0](权重 1)与标量 f0 的结果也逐 bit 相同:逐项公式
      与旧实现完全一致,且 0.0 + x 对有限浮点数是精确运算。
    - 注意:旧实现对"与 t 同形状的 f0 数组"做的是逐元素广播;当前版本把一维
      f0 数组统一解释为"多频列表并求和"。全仓库现有调用方(analytic_halfspace、
      high_order_teacher、generate_acoustic_dataset)均只传标量 float,不受影响。
    """
    t = np.asarray(t_s, dtype=np.float64)
    f0 = np.asarray(f0_hz, dtype=np.float64)
    if f0.ndim == 0:
        # ---- 旧行为:单频 Ricker(与原实现逐 bit 一致,勿改运算顺序) ----
        if f0_weights is not None:
            raise ValueError("f0_weights 仅在 f0_hz 为频率列表(多频叠加)时有效")
        a = 0.6 * f0 * t - 1.0
        return -9.6 * f0 * a * np.exp(-8.0 * a**2)
    # ---- 新行为:多频 Ricker 叠加(宽带源) ----
    if f0.ndim != 1 or f0.size < 1:
        raise ValueError(f"f0_hz 多频模式要求一维非空序列,实际 shape={f0.shape}")
    if f0_weights is None:
        weights = np.ones(f0.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(f0_weights, dtype=np.float64)
        if weights.shape != f0.shape:
            raise ValueError(
                f"f0_weights 形状 {weights.shape} 必须与 f0_hz 形状 {f0.shape} 一致"
            )
    out = np.zeros_like(t, dtype=np.float64)
    for f0_k, w_k in zip(f0.tolist(), weights.tolist()):
        a = 0.6 * f0_k * t - 1.0
        out = out + w_k * (-9.6 * f0_k * a * np.exp(-8.0 * a**2))
    return out


def _bracketing_index(coord_m: float, *, n: int, d_m: float, centering: str) -> tuple[int, int, float]:
    if centering not in {"cell", "node"}:
        raise ValueError("centering must be 'cell' or 'node'")
    offset = 0.5 if centering == "cell" else 0.0
    coordinates = (np.arange(n, dtype=np.float64) + offset) * float(d_m)
    if not (coordinates[0] <= coord_m <= coordinates[-1]):
        raise ValueError(
            f"source coordinate {coord_m} is outside {centering} range "
            f"[{coordinates[0]}, {coordinates[-1]}]"
        )
    upper = int(np.searchsorted(coordinates, coord_m, side="right"))
    upper = min(max(upper, 1), n - 1)
    lower = upper - 1
    frac = float((coord_m - coordinates[lower]) / (coordinates[upper] - coordinates[lower]))
    return lower, upper, frac


def bilinear_point_source(
    x0_m: float,
    z0_m: float,
    *,
    nx: int,
    nz: int,
    dx_m: float,
    dz_m: float,
    centering: str = "cell",
) -> BilinearPointSource:
    x0 = float(x0_m)
    z0 = float(z0_m)
    ix0, ix1, fx = _bracketing_index(x0, n=int(nx), d_m=float(dx_m), centering=centering)
    iz0, iz1, fz = _bracketing_index(z0, n=int(nz), d_m=float(dz_m), centering=centering)

    indices = np.asarray([[iz0, ix0], [iz0, ix1], [iz1, ix0], [iz1, ix1]], dtype=np.int32)
    weights = np.asarray(
        [(1.0 - fz) * (1.0 - fx), (1.0 - fz) * fx, fz * (1.0 - fx), fz * fx],
        dtype=np.float64,
    )
    weights = np.maximum(weights, 0.0)
    weights = weights / weights.sum()
    source_map = np.zeros((int(nz), int(nx)), dtype=np.float32)
    delta_h = np.zeros_like(source_map, dtype=np.float64)
    for (z_idx, x_idx), weight in zip(indices, weights):
        source_map[int(z_idx), int(x_idx)] += np.float32(weight)
        delta_h[int(z_idx), int(x_idx)] += float(weight) / (float(dx_m) * float(dz_m))
    return BilinearPointSource(
        indices=indices,
        weights=weights,
        source_map=source_map,
        delta_h=delta_h,
        x0_m=x0,
        z0_m=z0,
    )
