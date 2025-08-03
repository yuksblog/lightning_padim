import logging
from collections.abc import Sequence

import timm
import torch
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor

logger = logging.getLogger(__name__)


class FeatureExtractor(nn.Module):

    def __init__(
        self,
        backbone: str | nn.Module,
        layers: Sequence[str],
        pre_trained: bool = True,
        requires_grad: bool = False,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.layers = list(layers)
        self.requires_grad = requires_grad

        # Create feature_extractor
        if isinstance(backbone, nn.Module):
            # If backbone is a pre-trained model, use it directly
            self.feature_extractor = create_feature_extractor(
                backbone,
                return_nodes={layer: layer for layer in self.layers},
            )
            layer_metadata = self._dryrun_find_featuremap_dims()
            self.out_dims = [
                feature_info["num_features"]
                for layer_name, feature_info in layer_metadata.items()
            ]

        elif isinstance(backbone, str):
            # If backbone is a string, create a model using timm
            self.idx = self._map_layer_to_idx()
            self.feature_extractor = timm.create_model(
                backbone,
                pretrained=pre_trained,
                pretrained_cfg=None,
                features_only=True,
                exportable=True,
                out_indices=self.idx,
            )
            self.out_dims = self.feature_extractor.feature_info.channels()

        else:
            # Raise an error if backbone is neither a string nor a nn.Module
            msg = f"Backbone of type {type(backbone)} must be of type str or nn.Module."
            raise TypeError(msg)

        # Ensure the output is a dictionary with layer names as keys
        self._features = {layer: torch.empty(0) for layer in self.layers}

    def _map_layer_to_idx(self) -> list[int]:

        # Create a model to find indices of layers
        model = timm.create_model(
            self.backbone,
            pretrained=False,
            features_only=True,
            exportable=True,
        )

        # model.feature_info.info returns list of dicts containing info,
        # inside which "module" contains layer name
        idx = []
        layer_names = [info["module"] for info in model.feature_info.info]
        for layer in self.layers:
            try:
                idx.append(layer_names.index(layer))
            except ValueError:  # noqa: PERF203
                msg = f"Layer {layer} not found in model {self.backbone}. "
                "Available layers: {layer_names}"
                logger.warning(msg)
                # Remove unfound key from layer dict
                self.layers.remove(layer)

        return idx

    def _dryrun_find_featuremap_dims(
        self,
    ) -> dict[str, dict[str, int | tuple[int, int]]]:

        device = next(self.feature_extractor.parameters()).device
        dryrun_input = torch.empty(1, 3, *(256, 256)).to(device)
        dryrun_features = self.feature_extractor(dryrun_input)

        return {
            layer: {
                "num_features": dryrun_features[layer].shape[1],
                "resolution": dryrun_features[layer].shape[2:],
            }
            for layer in self.layers
        }

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:

        # Extract features from the model
        if self.requires_grad:
            features = self.feature_extractor(inputs)
        else:
            self.feature_extractor.eval()
            with torch.no_grad():
                features = self.feature_extractor(inputs)

        # Ensure the output is a dictionary
        if not isinstance(features, dict):
            features = dict(zip(self.layers, features, strict=True))

        return features
