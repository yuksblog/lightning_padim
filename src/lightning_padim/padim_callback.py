from lightning import Callback, LightningModule, Trainer


class PadimCallback(Callback):

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Call the Padim.on_train_end() to compute the MultiVariateGausian parameters
        pl_module.on_train_end()

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Call the Padim.on_validation_end() to compute the normalization parameters
        pl_module.on_validation_end()
