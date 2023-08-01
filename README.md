# FourierViT

This repository contains the implementation of a vision transformer that processes images in fourier space instead of its pixel values directly. This tweak intends to improve image understanding since attention uses a set of multiplications of patch embeddings, whereas a multiplication in fourier space is the same as a convolution in normal space. Thus, the power of convolutions is fused with the power of attention without explicitly using convolutions.
