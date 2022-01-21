import torch
import scipy.fft
import numpy as np
from functools import lru_cache


@lru_cache()
def compute_dct_mat(n: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    m = scipy.fft.dct(np.eye(n), norm="ortho")
    return torch.tensor(m, device=device, dtype=dtype)


@lru_cache()
def compute_idct_mat(n: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    m = scipy.fft.idct(np.eye(n), norm="ortho")
    return torch.tensor(m, device=device, dtype=dtype)


def dct2(t: torch.Tensor) -> torch.Tensor:
    h, w = t.shape[-2:]
    mh = compute_dct_mat(h, device=t.device, dtype=t.dtype)
    mw = compute_dct_mat(w, device=t.device, dtype=t.dtype)
    return torch.einsum("...hw,hi,wj->...ij", t, mh, mw)


def idct2(t: torch.Tensor) -> torch.Tensor:
    h, w = t.shape[-2:]
    mh = compute_idct_mat(h, device=t.device, dtype=t.dtype)
    mw = compute_idct_mat(w, device=t.device, dtype=t.dtype)
    return torch.einsum("...hw,hi,wj->...ij", t, mh, mw)