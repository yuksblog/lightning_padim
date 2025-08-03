import torch
from torch import Tensor, nn


class MinMaxNormalizer(nn.Module):

    def __init__(self):
        super().__init__()

        self.register_buffer("min", torch.tensor(0.0))
        self.register_buffer("max", torch.tensor(0.0))

        self.min: torch.Tensor
        self.max: torch.Tensor

    def forward(self, anomaly_map: Tensor) -> list[Tensor]:
        anomaly_map = (anomaly_map - self.min) / (self.max - self.min)
        anomaly_map = anomaly_map.clamp(0, 1)
        return anomaly_map

    def calculate(self, embedding: Tensor) -> None:
        self.min = torch.min(embedding)
        self.max = torch.max(embedding)
