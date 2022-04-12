from torch import Tensor, view_as_real
from torch.fft import fftn
from torch.nn import Module, Conv2d


class PatchEmbedding(Module):

    def __init__(self, image_size: int, channels: int = 3, patch_size: int = 16, embed_dim: int = 768) -> None:
        """
        Split a tensor of at least two dimensions
        into patches of a specified size
        within the first two dimensions
        (To input a PIL image, use torchvision.transforms.PILToTensor())
        ----------------------------------------------------------------
        :param image_size: image size (square)
        :param patch_size: patch size (square)
        """
        super().__init__()

        self.image_size = image_size
        # two times channels because we have two complex components
        # that will be flattened into channel dimension
        self.channels = 2 * channels
        self.patch_size = patch_size

        self.linear_projection = Conv2d(self.channels, embed_dim, (patch_size, patch_size), (patch_size, patch_size))

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    def forward(self, x: Tensor) -> Tensor:
        """
        Call the class to create the patches
        :param x: the input tensor
        :return: an array of patches
        """

        # 1. convert into fourier domain
        # 2. convert complex numbers into arrays of two numbers
        # 3. transpose last dimension (complex components) and x dimension
        #    such that (n, 3, 128, 128, 2) -> (n, 3, 2, 128, 128)
        # 4. combine channel and complex dimension:
        #    (n, 3, 2, 128, 128) -> (n, 6, 128, 128)
        fourier = view_as_real(fftn(x)).transpose(2, 4).flatten(1, 2)

        patches = self.linear_projection(fourier) \
            .flatten(2) \
            .transpose(1, 2)

        return patches