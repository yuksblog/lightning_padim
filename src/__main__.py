import logging
import logging.config
import os
import cv2
import torch
from torchvision.transforms.v2 import (
    Compose,
    Resize,
    Normalize,
    ToImage,
    ToDtype,
)
import lightning as L
import yaml

from lightning_padim.image_utils import superimpose_anomaly_map
from lightning_padim.mvtec_data_module import MVTecDataModule
from lightning_padim.padim import Padim

data_root = "./datas"
target = "bottle"
image_size = (256, 256)
batch_size = 32

backbone = "resnet18"
layers = ["layer1", "layer2", "layer3"]
n_features = 100
# backbone = "vit_tiny_patch16_224"
# layers = ["blocks.1", "blocks.2", "blocks.3"]
# n_features = 100
blur_sigma = 4

model_path = "./ckpt/model.ckpt"

image_path = "./datas/bottle/test/broken_large/000.png"
# image_path = "./datas/bottle/test/good/000.png"

if __name__ == "__main__":

    if os.path.isfile("logging.yaml"):
        logging.config.dictConfig(yaml.safe_load(open("logging.yaml").read()))

    # Transform
    transform = Compose(
        [
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Resize(image_size, antialias=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # LightningDataModule
    data_module = MVTecDataModule(
        data_dir=data_root,
        target="bottle",
        transform=transform,
        batch_size=batch_size,
    )
    data_module.prepare_data()

    # LightningModel
    model = Padim(
        backbone=backbone,
        layers=layers,
        n_features=n_features,
        blur_sigma=blur_sigma,
    )

    # Train
    trainer = L.Trainer(max_epochs=1)
    data_module.setup("fit")
    trainer.fit(model=model, train_dataloaders=data_module.train_dataloader())

    # Validation
    data_module.setup("validate")
    trainer.validate(model=model, dataloaders=data_module.val_dataloader())

    # Save the model
    trainer.save_checkpoint(model_path)

    # Load the model
    model = Padim.load_from_checkpoint(
        model_path,
        backbone=backbone,
        layers=layers,
        n_features=n_features,
        blur_sigma=blur_sigma,
    )

    # Read an image
    image = cv2.imread(image_path)
    input_tensor = transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    input_tensor = torch.unsqueeze(input_tensor, 0)

    # Predict an image
    model.eval()
    anomaly_map = model(input_tensor)

    # Print anomaly value
    print(f"Anomaly value: {anomaly_map.max()}")

    # Create and save the heatmap
    heatmap = superimpose_anomaly_map(anomaly_map, image)
    cv2.imwrite("heatmap.png", heatmap)
