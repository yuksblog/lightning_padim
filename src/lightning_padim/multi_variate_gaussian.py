from typing import Any
import torch
from torch import Tensor, nn


class MultiVariateGaussian(nn.Module):

    def __init__(self):
        super().__init__()

        # NOTE mean and inv_covariance are reshaped on losd_state_dict
        self.param_names = ["mean", "inv_covariance"]
        for param_name in self.param_names:
            self.register_buffer(param_name, torch.empty(0))

        self.mean: torch.Tensor
        self.inv_covariance: torch.Tensor

    @staticmethod
    def _cov(
        observations: Tensor,
        rowvar: bool = False,
        bias: bool = False,
        ddof: int | None = None,
        aweights: Tensor = None,
    ) -> Tensor:

        # ensure at least 2D
        if observations.dim() == 1:
            observations = observations.view(-1, 1)

        # treat each column as a data point, each row as a variable
        if rowvar and observations.shape[0] != 1:
            observations = observations.t()

        if ddof is None:
            ddof = 1 if bias == 0 else 0

        weights = aweights
        weights_sum: Any

        if weights is not None:
            if not torch.is_tensor(weights):
                weights = torch.tensor(weights, dtype=torch.float)
            weights_sum = torch.sum(weights)
            avg = torch.sum(observations * (weights / weights_sum)[:, None], 0)
        else:
            avg = torch.mean(observations, 0)

        # Determine the normalization
        if weights is None:
            fact = observations.shape[0] - ddof
        elif ddof == 0:
            fact = weights_sum
        elif aweights is None:
            fact = weights_sum - ddof
        else:
            fact = weights_sum - ddof * torch.sum(weights * weights) / weights_sum

        observations_m = observations.sub(avg.expand_as(observations))

        x_transposed = (
            observations_m.t()
            if weights is None
            else torch.mm(torch.diag(weights), observations_m).t()
        )

        covariance = torch.mm(x_transposed, observations_m)
        covariance = covariance / fact

        return covariance.squeeze()

    def forward(self, embedding: Tensor) -> list[Tensor]:

        device = embedding.device

        batch, channel, height, width = embedding.size()
        embedding_vectors = embedding.view(batch, channel, height * width)
        self.mean = torch.mean(embedding_vectors, dim=0)
        covariance = torch.zeros(size=(channel, channel, height * width), device=device)
        identity = torch.eye(channel).to(device)
        for i in range(height * width):
            covariance[:, :, i] = (
                self._cov(embedding_vectors[:, :, i], rowvar=False) + 0.01 * identity
            )

        # Stabilize the covariance matrix by adding a small regularization term
        stabilized_covariance = covariance.permute(2, 0, 1) + 1e-5 * identity

        # Check if the device is MPS and fallback to CPU if necessary
        if device.type == "mps":
            # Move stabilized covariance to CPU for inversion
            self.inv_covariance = torch.linalg.inv(stabilized_covariance.cpu()).to(
                device
            )
        else:
            # Calculate inverse covariance as we need only the inverse
            self.inv_covariance = torch.linalg.inv(stabilized_covariance)

        return [self.mean, self.inv_covariance]

    def calculate(self, embedding: Tensor) -> None:
        self.forward(embedding)

    def _load_from_state_dict(self, state_dict: dict, prefix: str, *args) -> None:

        # # Resize the tensors in the state_dict to match the expected shapes
        self.mean.resize_(state_dict[f"{prefix}mean"].shape)
        self.inv_covariance.resize_(state_dict[f"{prefix}inv_covariance"].shape)

        # Call the parent method to load the state_dict
        super()._load_from_state_dict(state_dict, prefix, *args)
