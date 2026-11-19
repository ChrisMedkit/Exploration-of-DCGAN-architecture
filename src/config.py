"""Paths, hyperparameters and experiment definitions shared by the whole project.

Hyperparameters follow the DCGAN paper (Radford et al., 2016) and the class
notebook `code/L14/04_02_dcgan-celeba.ipynb`. Change one thing at a time, and
only through an entry in EXPERIMENTS, so every run can be compared to E0.
"""

from pathlib import Path

import torch

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"              # torchvision downloads flowers-102/ here
DATA_PROCESSED = ROOT / "data" / "processed"  # 64x64 image cache + CLIP embeddings
MODELS = ROOT / "models"                      # checkpoints: models/<run>/last.pt
RUNS = ROOT / "runs"                          # per-run CSV logs
FIGURES = ROOT / "reports" / "figures"        # figures that go in the report

FLOWERS_DIR = DATA_RAW / "flowers-102"        # created by torchvision.datasets.Flowers102
CAPTIONS_DIR = DATA_RAW / "text_c10"          # Reed et al. captions: one .txt per image
IMAGE_CACHE = DATA_PROCESSED / "flowers64.pt"
TEXT_CACHE = DATA_PROCESSED / "clip_text_vitb32.pt"

# ---------------------------------------------------------------- reproducibility
RANDOM_SEED = 42
SEEDS = (42, 43, 44)  # 3 seeds per experiment for significance (report §3.2)

# ---------------------------------------------------------------- device (this Mac: M2 Pro)
def get_device() -> torch.device:
    """mps on Apple silicon, cuda if present, else cpu."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()

# ---------------------------------------------------------------- DCGAN (paper defaults)
IMAGE_SIZE = 64        # DCGAN generates 64x64; do NOT go higher until E0 works
IMAGE_CHANNELS = 3
LATENT_DIM = 100       # z ~ N(0, I), shape (B, 100, 1, 1)
NGF = 64               # generator feature maps
NDF = 64               # discriminator feature maps
BATCH_SIZE = 128
NUM_EPOCHS = 100       # ~8k images -> 64 batches/epoch; check samples every epoch
LR_G = 2e-4
LR_D = 2e-4
BETAS = (0.5, 0.999)   # beta1 = 0.5 is part of the DCGAN recipe
NUM_WORKERS = 0        # images are pre-cached as one tensor; workers are not needed
SAMPLE_EVERY = 1       # save a fixed-noise grid every N epochs

# ---------------------------------------------------------------- text conditioning (Reed et al., 2016)
CLIP_MODEL = "ViT-B/32"  # frozen text encoder, 512-d embedding
TEXT_DIM = 512
TEXT_PROJ_DIM = 128      # Reed et al. compress the text embedding to 128-d
CAPTIONS_PER_IMAGE = 10
INTERP_BETA = 0.5        # GAN-INT: mix two caption embeddings with this weight

# ---------------------------------------------------------------- evaluation
NUM_EVAL_SAMPLES = 5000  # images generated for FID / CLIP score
EVAL_BATCH_SIZE = 250

# ---------------------------------------------------------------- experiments (report §3.1)
# One independent variable changes per row. E0 is the tested baseline.
EXPERIMENTS = {
    "E0_dcgan": dict(mode="dcgan", conditional=False, mismatch=False, interpolate=False,
                     note="Unconditional DCGAN, class L14 architecture. Must produce flowers first."),
    "E1_cond": dict(mode="cond", conditional=True, mismatch=False, interpolate=False,
                    note="+ CLIP text embedding concatenated in G and D (plain conditional GAN)."),
    "E2_gan_cls": dict(mode="gan_cls", conditional=True, mismatch=True, interpolate=False,
                       note="+ matching-aware D: real image with wrong caption is a negative."),
    "E3_gan_int_cls": dict(mode="gan_cls", conditional=True, mismatch=True, interpolate=True,
                           note="+ GAN-INT: G also trained on interpolated caption embeddings."),
}
