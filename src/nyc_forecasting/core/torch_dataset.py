import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, data_array: np.ndarray, target_array: np.ndarray, input_len: int = 24, output_len: int = 1):
        self.data = data_array.astype(np.float32)
        self.target = target_array.astype(np.float32)
        self.input_len = input_len
        self.output_len = output_len

    def __len__(self) -> int:
        return len(self.data) - self.input_len - self.output_len + 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.input_len]
        y = self.target[idx + self.input_len : idx + self.input_len + self.output_len]
        return torch.from_numpy(x), torch.from_numpy(y)


