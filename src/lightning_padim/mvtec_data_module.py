import os
import hashlib
import tarfile

from urllib.request import urlretrieve
import lightning as L
import torch
from torch.utils.data import DataLoader
from lightning_padim.download_tqdm import DownloadTqdm
from lightning_padim.mvtec_data_set import MVTecDataset


class MVTecDataModule(L.LightningDataModule):

    def __init__(self, data_dir: str, target: str, transform, batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.target = target
        self.transform = transform
        self.batch_size = batch_size
        self.val_ratio = 0.2

        self.train_set = None
        self.val_set = None

        self.url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz"
        self.hashsum = (
            "cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d"
        )
        self.tarxz_name = "mvtec_anomaly_detection.tar.xz"

    def prepare_data(self):

        # Download tar.xz file if it does not exist
        tarxz_path = os.path.join(self.data_dir, self.tarxz_name)
        if not os.path.exists(tarxz_path):
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)

            # Download the file with progress bar
            with DownloadTqdm(
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                miniters=1,
                desc=self.tarxz_name,
            ) as t:
                urlretrieve(
                    url=self.url,
                    filename=os.path.join(self.data_dir, self.tarxz_name),
                    reporthook=t.update_to,
                )

            # Check the hash of the downloaded file
            with open(tarxz_path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
                if file_hash != self.hashsum:
                    raise ValueError(
                        f"Downloaded file hash does not match. Expected {self.hashsum}, "
                        f"but got {file_hash}."
                    )

            # Extract the tar.xz file
            with tarfile.open(tarxz_path, "r:xz") as tar:
                tar.extractall(path=self.data_dir)

    def setup(self, stage: str = None):
        if stage == "fit":
            # train_set
            self.train_set = MVTecDataset(
                f"{self.data_dir}/{self.target}",
                transform=self.transform,
                is_train=True,
            )

        if stage == "validate":
            # dataset of /test folder
            test_set_all = MVTecDataset(
                f"{self.data_dir}/{self.target}",
                transform=self.transform,
                is_train=False,
            )

            # val_set
            val_size = int(len(test_set_all) * self.val_ratio)
            rest_size = len(test_set_all) - val_size
            self.val_set, _ = torch.utils.data.random_split(
                test_set_all, [val_size, rest_size]
            )

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)
