from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import cuda
from optuna import Trial, create_study
from optuna.pruners import MedianPruner

from VisionTransformer import VisionTransformer
from MnistDatamodule import MnistDatamodule

data_module = MnistDatamodule()


def objective(trial: Trial):
    patch_size = trial.suggest_categorical('patch_size', [1, 2, 4])
    dropout = trial.suggest_float('dropout', 0.1, 0.4)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2)

    model = VisionTransformer(
        image_size=28,
        patch_size=patch_size,
        num_encoder_layers=6,
        num_classes=data_module.num_classes,
        channels=1,
        dropout=dropout,
        learning_rate=learning_rate
    )

    model.load()

    trainer = Trainer(
        gpus=(-1 if cuda.is_available() else 0),
        max_epochs=100,
        callbacks=[EarlyStopping(monitor="val_loss")]
    )

    trainer.fit(model, data_module)

    trainer.test(model, data_module)

    return trainer.callback_metrics["val_acc"].item()


if __name__ == '__main__':
    pruner = MedianPruner()
    study = create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=100)