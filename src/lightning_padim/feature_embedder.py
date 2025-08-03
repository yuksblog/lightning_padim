import logging
import random
import torch
from torch import nn, Tensor
from torch.nn import functional as F_torch

from lightning_padim.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


class FeatureEmbedder(nn.Module):

    def __init__(
        self,
        backbone: str,
        layers: list[str],
        n_features: int | None = None,
    ) -> None:
        super().__init__()

        self.layers = layers
        self.n_features = n_features

        # Feature extractor
        self.feature_extractor = FeatureExtractor(backbone, layers)

        # check n_features
        self.n_features_max = sum(self.feature_extractor.out_dims)
        if n_features is None:
            # If n_features is not specified, use all features
            self.n_features = self.n_features_max
        else:
            # If n_features is specified, check if it is valid
            if n_features > self.n_features_max:
                msg = (
                    f"n_features ({n_features}) cannot be greater than "
                    f"the number of features in the model "
                    "({self.n_features_max})."
                )
                raise ValueError(msg)

        logger.info(
            f"Using {self.n_features} features out of "
            "{self.n_features_max} available features."
        )

        # Index for subsampling features
        self.register_buffer(
            "index",
            torch.tensor(random.sample(range(self.n_features_max), self.n_features)),
        )
        self.index: Tensor

    def forward(self, x: Tensor) -> Tensor:

        with torch.no_grad():
            features = self.feature_extractor(x)
            embeddings = self.generate_embedding(features)

        return embeddings

    def generate_embedding(self, features: dict[str, Tensor]) -> Tensor:

        # Concatenate features
        embeddings = features[self.layers[0]]
        for layer in self.layers[1:]:
            layer_embedding = features[layer]
            layer_embedding = F_torch.interpolate(
                layer_embedding,
                size=embeddings.shape[-2:],
                mode="nearest",
            )
            embeddings = torch.cat((embeddings, layer_embedding), 1)

        # subsample features
        index = self.index.to(embeddings.device)
        embeddings = torch.index_select(embeddings, 1, index)

        return embeddings
