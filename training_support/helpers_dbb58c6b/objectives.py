"""Helpers used by the public training entry points."""
import torch
from scipy.fft import next_fast_len

def require(condition, message):
    if not condition:
        raise ValueError(message)

def lowpass_time(traces, response):
    """Filter [shot,time,receiver] by zero padding the right end and cropping it back."""
    if response is None:
        return traces
    require(traces.ndim == 3 and traces.is_floating_point(), "Expected a real 3-D shot gather")
    samples = traces.shape[1]
    nfft = next_fast_len(2 * samples)
    require(response.shape == (nfft // 2 + 1,), "Wrong one-sided time-frequency response")
    spectrum = torch.fft.rfft(traces.transpose(1, 2).contiguous(), n=nfft, dim=-1)
    result = torch.fft.irfft(spectrum * response, n=nfft, dim=-1)[..., :samples]
    if result.requires_grad and result.device.type == "cpu":
        result.register_hook(lambda grad: grad.contiguous())
    return result.transpose(1, 2)
