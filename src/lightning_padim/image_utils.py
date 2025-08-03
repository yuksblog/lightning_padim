import cv2
import numpy as np
import torch


def superimpose_anomaly_map(
    anomaly_map: torch.Tensor,
    image: np.ndarray,
    alpha: float = 0.4,
    gamma: int = 0,
) -> np.ndarray:

    # Normalize the anomaly map to the range [0, 255]
    anomaly_map = anomaly_map.squeeze().detach().cpu().numpy()
    anomaly_map = anomaly_map * 255
    anomaly_map = anomaly_map.astype("uint8")

    # Convert the anomaly map to a color map
    color_map = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)

    # Resize the original image to match the color map size
    height, width = color_map.shape[:2]
    image = cv2.resize(image, (width, height))

    # Superimpose the color map on the original image
    heat_map = cv2.addWeighted(color_map, alpha, image, (1 - alpha), gamma)

    return heat_map


def read_rgb_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image
