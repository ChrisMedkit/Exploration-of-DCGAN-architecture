"""DCGAN networks.

Milestone 0 (IMPLEMENTED, do not change while running ablations):
    Generator, Discriminator, weights_init
    = DCGAN paper (Radford et al., 2016) / class notebook code/L14/04_02_dcgan-celeba.ipynb
      / PyTorch DCGAN tutorial (Inkawhich). Shapes for IMAGE_SIZE = 64:

    G:  z (B,100,1,1) -> 4x4x512 -> 8x8x256 -> 16x16x128 -> 32x32x64 -> 64x64x3 (tanh)
    D:  x (B,3,64,64) -> 32x32x64 -> 16x16x128 -> 8x8x256 -> 4x4x512 -> logit (B,)

Milestone 1+ (TODO): CondGenerator, CondDiscriminator
    = GAN-CLS text conditioning (Reed et al., 2016, Fig. 2) on top of the SAME blocks.
    The only new parts are a small text projection and one concatenation in each network.
"""

import torch
from torch import nn

from src import config


def weights_init(m: nn.Module) -> None:
    """DCGAN init: conv weights ~ N(0, 0.02); BatchNorm weight ~ N(1, 0.02), bias 0. Use with model.apply()."""
    name = m.__class__.__name__
    if "Conv" in name:
        nn.init.normal_(m.weight, 0.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)
    elif "BatchNorm" in name:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


def up_block(c_in: int, c_out: int) -> nn.Sequential:
    """ConvTranspose 4x4 stride 2 (doubles H, W) + BatchNorm + ReLU."""
    return nn.Sequential(
        nn.ConvTranspose2d(c_in, c_out, 4, 2, 1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


def down_block(c_in: int, c_out: int, batchnorm: bool = True) -> nn.Sequential:
    """Conv 4x4 stride 2 (halves H, W) + optional BatchNorm + LeakyReLU(0.2)."""
    layers = [nn.Conv2d(c_in, c_out, 4, 2, 1, bias=not batchnorm)]
    if batchnorm:
        layers.append(nn.BatchNorm2d(c_out))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


# ============================================================ Milestone 0: unconditional DCGAN
class Generator(nn.Module):
    """z (B, latent_dim, 1, 1) -> image (B, 3, 64, 64) in [-1, 1]."""

    def __init__(self, latent_dim: int = config.LATENT_DIM, ngf: int = config.NGF,
                 channels: int = config.IMAGE_CHANNELS):
        super().__init__()
        self.latent_dim = latent_dim
        self.stem = nn.Sequential(  # 1x1 -> 4x4
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            up_block(ngf * 8, ngf * 4),  # 8x8
            up_block(ngf * 4, ngf * 2),  # 16x16
            up_block(ngf * 2, ngf),      # 32x32
        )
        self.head = nn.Sequential(nn.ConvTranspose2d(ngf, channels, 4, 2, 1, bias=False), nn.Tanh())  # 64x64

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(self.stem(z)))


class Discriminator(nn.Module):
    """image (B, 3, 64, 64) -> raw logit (B,). Use BCEWithLogits (no sigmoid here)."""

    def __init__(self, ndf: int = config.NDF, channels: int = config.IMAGE_CHANNELS):
        super().__init__()
        self.features = nn.Sequential(
            down_block(channels, ndf, batchnorm=False),  # 32x32  (no BN on the first layer)
            down_block(ndf, ndf * 2),                     # 16x16
            down_block(ndf * 2, ndf * 4),                 # 8x8
            down_block(ndf * 4, ndf * 8),                 # 4x4
        )
        self.head = nn.Conv2d(ndf * 8, 1, 4, 1, 0)        # 1x1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).view(-1)


# ============================================================ Milestone 1+: text-conditional (GAN-CLS)
class TextProjection(nn.Module):
    """Compress a frozen CLIP text embedding (B, 512) to (B, 128): Linear + BatchNorm1d + LeakyReLU(0.2).

    Reed et al. (2016) Sec. 4.1: "compressed using a fully-connected layer to a small dimension (128)
    followed by leaky-ReLU". One instance in G and a separate one in D.
    """

    def __init__(self, text_dim: int = config.TEXT_DIM, proj_dim: int = config.TEXT_PROJ_DIM):
        super().__init__()
        self.text_dim, self.proj_dim = text_dim, proj_dim
        self.net = nn.Sequential(nn.Linear(text_dim, proj_dim), nn.BatchNorm1d(proj_dim),
                                 nn.LeakyReLU(0.2, inplace=True))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)  # (B, 512) -> (B, 128)


class CondGenerator(Generator):
    """(z, text embedding) -> image. Same blocks as Generator; only the stem input grows by proj_dim.

    Reed et al. Fig. 2 (left): concat [z ; proj(t)] -> (B, 100+128, 1, 1) -> the DCGAN stack.
    """

    def __init__(self, latent_dim: int = config.LATENT_DIM, ngf: int = config.NGF,
                 text_dim: int = config.TEXT_DIM, proj_dim: int = config.TEXT_PROJ_DIM):
        super().__init__(latent_dim=latent_dim + proj_dim, ngf=ngf)
        self.latent_dim = latent_dim  # noise size only; the stem was built for latent_dim + proj_dim
        self.text_proj = TextProjection(text_dim, proj_dim)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.text_proj(t)[:, :, None, None]           # (B, 128, 1, 1)
        x = torch.cat([z, h], dim=1)                      # (B, 228, 1, 1)
        return self.head(self.body(self.stem(x)))         # (B, 3, 64, 64)


class CondDiscriminator(Discriminator):
    """(image, text embedding) -> logit "real image AND matches the caption".

    Reed et al. Fig. 2 (right): run the DCGAN conv stack to 4x4x512, replicate proj(t) spatially to
    (B, 128, 4, 4), concat -> (B, 640, 4, 4), 1x1 conv + BN + LeakyReLU back to 512, then the 4x4 head.
    """

    def __init__(self, ndf: int = config.NDF, text_dim: int = config.TEXT_DIM,
                 proj_dim: int = config.TEXT_PROJ_DIM):
        super().__init__(ndf=ndf)
        self.text_proj = TextProjection(text_dim, proj_dim)
        self.joint = nn.Sequential(nn.Conv2d(ndf * 8 + proj_dim, ndf * 8, 1, 1, 0, bias=False),
                                   nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True))
        # self.head from Discriminator is reused unchanged

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        f = self.features(x)                                                   # (B, 512, 4, 4)
        h = self.text_proj(t)[:, :, None, None].expand(-1, -1, *f.shape[-2:])  # (B, 128, 4, 4)
        return self.head(self.joint(torch.cat([f, h], 1))).view(-1)           # (B,)


def build_models(conditional: bool, device: torch.device | str = config.DEVICE):
    """Return (G, D) for an experiment, initialised with weights_init and moved to device."""
    if conditional:
        g, d = CondGenerator(), CondDiscriminator()
    else:
        g, d = Generator(), Discriminator()
    g.apply(weights_init)
    d.apply(weights_init)
    return g.to(device), d.to(device)


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
