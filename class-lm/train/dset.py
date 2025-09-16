from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedBinDataset(Dataset):
    """Memory-map a flat binary token file into fixed-length sequences.

    The binary file is expected to contain unsigned 16-bit token IDs (a common
    format for GPT-style datasets). Adjust ``dtype`` when initialising if your
    preprocessing stored a different numeric type.
    """

    def __init__(
        self,
        bin_path: str | Path,
        *,
        seq_len: int,
        dtype: np.dtype | str = np.uint16,
    ) -> None:
        bin_path = Path(bin_path)
        if not bin_path.exists():
            raise FileNotFoundError(f"binary dataset not found: {bin_path}")
        self.seq_len = int(seq_len)
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")

        np_dtype = np.dtype(dtype)
        self._tokens = np.memmap(bin_path, dtype=np_dtype, mode="r")
        if self._tokens.size <= self.seq_len:
            raise ValueError(
                "binary file too small for the requested sequence length"
            )
        # We need one extra token per sequence to create the next-token target.
        usable = self._tokens.size - 1
        self._num_sequences = usable // self.seq_len
        self._usable_tokens = self._num_sequences * self.seq_len + 1

    def __len__(self) -> int:
        return self._num_sequences

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self._num_sequences:
            raise IndexError(idx)
        start = idx * self.seq_len
        end = start + self.seq_len
        x = np.array(self._tokens[start:end], copy=False)
        y = np.array(self._tokens[start + 1 : end + 1], copy=False)
        return (
            torch.from_numpy(x.astype(np.int64)),
            torch.from_numpy(y.astype(np.int64)),
        )

    def iter_tokens(self) -> Iterator[int]:
        """Yield the truncated token stream used to build sequences."""
        for token in self._tokens[: self._usable_tokens]:
            yield int(token)
