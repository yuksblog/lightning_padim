import logging
import torch
from torch.nn import functional as F
from torchvision.transforms import GaussianBlur
import lightning as L

from lightning_padim.multi_variate_gaussian import MultiVariateGaussian
from lightning_padim.padim_callback import PadimCallback
from lightning_padim.min_max_normalizer import MinMaxNormalizer
from lightning_padim.feature_embedder import FeatureEmbedder
from lightning_padim.anomaly_map_generator import AnomalyMapGenerator


logger = logging.getLogger(__name__)


class Padim(L.LightningModule):

    def __init__(
        self,
        backbone: str = "resnet18",
        layers: list[str] = ["layer1", "layer2", "layer3"],  # noqa: B006
        n_features: int | None = None,
        blur_sigma: float = None,
    ) -> None:
        super().__init__()

        # Feature embedding
        self.feature_embedder = FeatureEmbedder(
            backbone=backbone, layers=layers, n_features=n_features
        )

        # Multi Variate Gaussian for computing Mahalanobis distance
        self.gaussian = MultiVariateGaussian()

        # Optional Gaussian blur for anomaly map
        self.blur = None
        if blur_sigma is not None:
            kernel_size = 2 * int(4.0 * blur_sigma + 0.5) + 1
            self.blur = GaussianBlur(
                kernel_size=kernel_size, sigma=(blur_sigma, blur_sigma)
            )

        # Anomaly map generator
        self.anomaly_map = AnomalyMapGenerator()

        # Normalizer for anomaly map
        self.normalizer = MinMaxNormalizer()

        self.embeddings: list[torch.Tensor] = []
        self.val_anomaly_maps: list[torch.Tensor] = []

    def training_step(self, batch, batch_idx):
        embedding = self.feature_embedder(batch["image"])
        self.embeddings.append(embedding)

    def validation_step(self, batch, batch_idx):
        anomalymaps = self._generate_anomaly_map(batch["image"])
        self.val_anomaly_maps.append(anomalymaps)

    def on_train_end(self) -> None:
        # On training end, compute the MultiVariateGaussian parameters
        embeddings = torch.vstack(self.embeddings)
        self.gaussian.calculate(embeddings)

    def on_validation_end(self) -> None:
        # On validation end, compute the MinMaxNormalizer parameters
        val_anomalymaps = torch.vstack(self.val_anomaly_maps)
        self.normalizer.calculate(val_anomalymaps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Generate anomaly map
        image_size = x.shape[-2:]
        anomaly_map = self._generate_anomaly_map(x, image_size)

        # Normalize the anomaly map
        anomaly_map = self.normalizer(anomaly_map)

        return anomaly_map

    def _generate_anomaly_map(self, x: torch.Tensor, image_size=None) -> torch.Tensor:
        # Generate embeddings
        embeddings = self.feature_embedder(x)

        # Compute anomaly map
        anomaly_map = self.anomaly_map(
            embeddings,
            self.gaussian.mean,
            self.gaussian.inv_covariance,
        )

        # Resize anomaly map to original image size if provided
        if image_size is not None:
            anomaly_map = F.interpolate(
                anomaly_map,
                size=image_size,
                mode="bilinear",
                align_corners=False,
            )

        # Apply Gaussian blur if specified
        if self.blur is not None:
            anomaly_map = self.blur(anomaly_map)

        return anomaly_map

    def configure_optimizers(self):
        # Padim doesn't require optimization.
        return None
