# StyleGAN2 From Scratch in PyTorch
A complete implementation of a StyleGAN2 architecture built entirely from scratch using PyTorch. The model is trained on the CelebA dataset using a style-modulated generator with runtime weight demodulation, equalized learning rates, style mixing, and lazy R1/Path Length regularization.

## Generated Image


## Training GIF


## Project Structure
```text
stylegan2-from-scratch/
│
├── generated_images/
│   ├── epoch_005.png
│   ├── epoch_010.png
│   └── ...
│
├── data/
│   └── celeba/
│
├── checkpoint.pth
├── stylegan2_celeba.py
├── requirements.txt
├── README.md
└── training_progress.gif
```

## Requirements
```text
torch
torchvision
matplotlib
imageio
numpy
```

## References
- StyleGAN2 Paper: https://arxiv.org/abs/1912.04958
    *Analyzing and Improving the Image Quality of StyleGAN* (Karras et al., 2019)

- ProGAN Paper: https://arxiv.org/abs/1710.10196
    *Progressive Growing of GANs for Improved Quality, Stability, and Variation* (Karras et al., 2017)
