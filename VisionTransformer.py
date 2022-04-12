from PatchEmbedding import PatchEmbedding
from torch import zeros, cat, Tensor, argmax, save, load
from pytorch_lightning import LightningModule
from torch.nn import \
    Parameter, \
    Linear
from torch.nn import functional as F
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.utilities.types import LRSchedulerType
from torchmetrics import Accuracy
from os.path import isfile
from typing import Tuple, List

from transformer_encoder import TransformerEncoder


class VisionTransformer(LightningModule):
    def __init__(
            self,
            image_size: int,
            num_encoder_layers: int,
            num_classes: int,
            channels: int = 3,
            patch_size: int = 16,
            num_heads: int = 12,
            embed_dim: int = 768,
            dropout: float = 0.1,
            learning_rate: float = 1e-3
    ) -> None:
        super().__init__()
        self.model_type = 'Transformer'

        self.filename = 'model.pt'

        # the embedding dimension is used throughout the entire model
        # it is a hyperparameter of the transformer encoder layer
        # and hence the image patches have to be projected to this dimension.
        self.embed_dim = embed_dim

        # The patch embedding module takes a sequence of images and for each image,
        # it splits it into 16x16 patches, flattens them and projects them to
        # the embedding dimension `embed_dim`
        # hence, the resulting vector will have the shape (num_images, num_patches, embed_dim)
        self.patch_embed = PatchEmbedding(image_size, channels=channels, patch_size=patch_size, embed_dim=embed_dim)

        # We prepend a class token `[class]` to an image patch sequence
        # this class token is a learnable parameter and is at the end fit into
        # the MLP head for the eventual classification.
        # It thus has the same dimensions as a single image patch (embed_dim)
        self.class_token = Parameter(zeros(1, 1, embed_dim))

        # We concatenate each image patch with its positional embedding
        # so that this information is not lost in the transformer
        # (as the order of tokens fed into a transformer normally does
        # not make a difference) it also is a parameter the model learns
        # Its second dimension is larger than the number of patches by exactly 1,
        # because we prepend the class token to the patch embedding before
        # adding the positional embedding.
        # Therefore, the shape is (1, num_patches + 1, embed_dim) as it is added to
        # one patch embedding (hence the 1 in front), and each token in the embedding
        # has embed_dim dimensions.

        self.positional_embedding = Parameter(
            zeros(
                1,
                self.patch_embed.num_patches + 1,
                embed_dim
            )
        )

        # we use a transformer encoder as the main part of the network.
        # There are num_encoder_layers in this encoder.
        self.transformer_encoder = TransformerEncoder(
            d_model=embed_dim,
            n_heads=num_heads,
            n_layers=num_encoder_layers,
            d_ff=2048,
            dropout=dropout
        )

        self.MLP_head = Linear(embed_dim, num_classes)

        self.learning_rate = learning_rate

        self.accuracy = Accuracy()

    def forward(self, x: Tensor) -> Tensor:
        # create patch embeddings
        x = self.patch_embed(x)

        # replicate class token as many times as there
        # are images in the tensor
        n_class_tokens = self.class_token.expand(x.shape[0], -1, -1)

        # prepend class tokens to embeddings
        x = cat((n_class_tokens, x), dim=1)

        # add positional embedding
        x += self.positional_embedding

        x = self.transformer_encoder(x, mask=None)

        # get the class embeddings out of all embeddings
        final_class_token = x[:, 0]

        # lastly, feed it into the MLP
        return self.MLP_head(final_class_token)

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        predictions = argmax(logits, dim=1)
        self.accuracy(predictions, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy, prog_bar=True)
        return loss

    def test_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[LRSchedulerType]]:
        """
        Decide what optimizer and scheduler Pytorch Lightning should use
        :return: a Tuple of a list of optimizers and a list of schedulers,
        in our case we just use one of each
        """
        optimizer = Adam(self.parameters(), lr=self.learning_rate)

        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, patience=3),
            'monitor': 'train_loss'
        }

        return [optimizer], [scheduler]

    def training_epoch_end(self, _) -> None:
        """
        Hook that fires after each epoch. Saves the model
        to a file.
        """
        # save the model after every epoch
        self.save()

    def save(self) -> None:
        """
        Save model to given filename
        """
        save(self.state_dict(), self.filename)

    def load(self) -> None:
        """
        Load model from given filename
        """
        if isfile(self.filename):
            self.load_state_dict(load(self.filename))
