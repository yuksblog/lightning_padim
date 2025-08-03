import torch
from torch import nn, Tensor

__all__ = [
    "AnomalyMapGenerator",
]


class AnomalyMapGenerator(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def compute_distance(embedding: Tensor, stats: list[Tensor]) -> Tensor:
        batch, channel, height, width = embedding.shape
        embedding = embedding.reshape(batch, channel, height * width)

        # calculate mahalanobis distances
        mean, inv_covariance = stats
        delta = (embedding - mean).permute(2, 0, 1)

        distances = (torch.matmul(delta, inv_covariance) * delta).sum(2).permute(1, 0)
        distances = distances.reshape(batch, 1, height, width)
        distances = distances.clamp(0).sqrt()

        return distances

    def compute_anomaly_map(
        self,
        embedding: torch.Tensor,
        mean: torch.Tensor,
        inv_covariance: torch.Tensor,
    ) -> torch.Tensor:

        # Compute the score map using the Mahalanobis distance
        score_map = self.compute_distance(
            embedding=embedding,
            stats=[
                mean.to(embedding.device),
                inv_covariance.to(embedding.device),
            ],
        )

        return score_map

    def forward(
        self, embedding: Tensor, mean: Tensor, inv_covariance: Tensor
    ) -> Tensor:

        # return self.compute_anomaly_map(embedding, mean, inv_covariance)

        batch, channel, height, width = embedding.shape
        embedding = embedding.reshape(batch, channel, height * width)

        # calculate mahalanobis distances
        delta = (embedding - mean).permute(2, 0, 1)
        score_map = (torch.matmul(delta, inv_covariance) * delta).sum(2).permute(1, 0)
        score_map = score_map.reshape(batch, 1, height, width)
        score_map = score_map.clamp(0).sqrt()

        return score_map
