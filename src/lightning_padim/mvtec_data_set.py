from pathlib import Path
from typing import Any
import torch
import torch.utils.data
import albumentations as A

from lightning_padim.image_utils import read_rgb_image


class MVTecDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        target_dir: str | Path,
        transform: A.Compose = None,
        is_train: bool = False,
    ) -> None:
        super().__init__()

        self.transform = transform
        self.is_train = is_train

        if isinstance(target_dir, str):
            target_dir = Path(target_dir)
        pattern = "train/*/*" if is_train else "test/good/*"
        self.image_path_list = list(target_dir.glob(pattern))

    def __getitem__(self, index: int) -> dict[str, str | None | Any]:
        # image_path
        image_path = str(self.image_path_list[index])

        # image
        image = read_rgb_image(image_path)
        image = self.transform(image)

        # label
        label = True
        if not self.is_train:
            index = True if image_path.find("test/good") >= 0 else False

        return {
            "image_path": image_path,
            "image": image,
            "label": label,
        }

    def __len__(self):
        return len(self.image_path_list)
