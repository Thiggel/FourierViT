from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import cuda
from optuna import Trial, create_study
from optuna.pruners import MedianPruner
from os.path import exists
from os import mkdir
from json import dumps

from VisionTransformer import VisionTransformer
from MnistDatamodule import MnistDatamodule

data_module = MnistDatamodule()

SAVED_DIR = 'saved'

def objective(trial: Trial):
    patch_size = trial.suggest_categorical('patch_size', [1, 2, 4])
    dropout = trial.suggest_float('dropout', 0.1, 0.4)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2)

    hyperparams = dumps({
        "patch_size": patch_size,
        "dropout": dropout,
        "learning_rate": learning_rate
    })

    print("Hyper-parameters: ", hyperparams)

    model = VisionTransformer(
        image_size=28,
        patch_size=patch_size,
        num_encoder_layers=6,
        num_classes=data_module.num_classes,
        channels=1,
        dropout=dropout,
        learning_rate=learning_rate,
        filename=f'{SAVED_DIR}/{hyperparams}.pt'
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

def create_saved_dir():
    # create directory for saved transformer if not exists
    if not exists(SAVED_DIR):
        mkdir(SAVED_DIR)


if __name__ == '__main__':
    create_saved_dir()

    pruner = MedianPruner()
    study = create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=100)