"""Hard zero initial pressure and rate on the native UL output axis."""

FIRST_RAMP_ROW = 2
LAST_RAMP_ROW = 10


def apply_hard_initial_pressure(pressure, rows):
    """Apply a C² startup gate to summed pressure on the registered time grid.

    The second-order forward derivative uses rows 0, 1 and 2. A zero
    gate on those rows enforces it for independently queried raw fields.
    The quintic ramp reaches one at row 10 and stays there.
    """
    fraction = ((rows - FIRST_RAMP_ROW).to(dtype=pressure.dtype)
                / (LAST_RAMP_ROW - FIRST_RAMP_ROW)).clamp(0., 1.)
    gate = fraction.pow(3) * (10. + fraction * (-15. + 6. * fraction))
    return pressure * gate[:, None, None]
