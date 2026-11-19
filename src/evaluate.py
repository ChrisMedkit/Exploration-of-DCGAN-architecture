"""Figures and metrics for report §3.

IMPLEMENTED (Milestone 0): save_grid, plot_losses, plot_sample_progress.
TODO (Milestone 3): FID, CLIP score, discriminator pair accuracy, seed summary + significance.

Why these metrics: Project 1 reported only discriminator accuracy, which says nothing about whether
images look real or match the caption. FID measures realism/diversity against real images; CLIP
score measures text-image agreement. Both are standard for text-to-image GANs.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures without a display (training runs from the terminal)
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torchvision.utils import make_grid, save_image  # noqa: E402

from src import config  # noqa: E402
from src.data import to_image_range  # noqa: E402


# ============================================================ Milestone 0: figures
def save_grid(images: torch.Tensor, path: Path, nrow: int = 8) -> Path:
    """Save a (B, 3, H, W) batch in [-1, 1] as one PNG grid."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(to_image_range(images.detach().cpu()), path, nrow=nrow, padding=2)
    return path


def plot_losses(log_csv: Path, out: Path | None = None) -> Path:
    """Plot per-epoch D/G loss and D(x) vs D(G(z)) from a run's log.csv (report figure)."""
    df = pd.read_csv(log_csv)
    out = Path(out or config.FIGURES / f"{Path(log_csv).parent.name}_losses.png")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(df["epoch"], df["d_loss"], label="Discriminator")
    ax[0].plot(df["epoch"], df["g_loss"], label="Generator")
    ax[0].set(xlabel="Epoch", ylabel="Loss", title="Training loss")
    ax[0].legend()
    ax[1].plot(df["epoch"], df["d_real"], label="D(x)  real")
    ax[1].plot(df["epoch"], df["d_fake"], label="D(G(z))  fake")
    ax[1].axhline(0.5, color="gray", lw=0.8, ls="--")
    ax[1].set(xlabel="Epoch", ylabel="Mean D output (sigmoid)", ylim=(0, 1), title="Discriminator output")
    ax[1].legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_sample_progress(sample_dir: Path, epochs: list[int], out: Path | None = None) -> Path:
    """Put the saved fixed-noise grids of several epochs side by side (report Fig.: samples over training)."""
    sample_dir = Path(sample_dir)
    out = Path(out or config.FIGURES / f"{sample_dir.name}_progress.png")
    fig, axes = plt.subplots(1, len(epochs), figsize=(4 * len(epochs), 4.3))
    for ax, e in zip(axes if len(epochs) > 1 else [axes], epochs):
        ax.imshow(plt.imread(sample_dir / f"epoch_{e:03d}.png"))
        ax.set_title(f"Epoch {e}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ============================================================ Milestone 3: metrics
@torch.no_grad()
def generate_for_eval(G, text_emb: torch.Tensor | None, n: int = config.NUM_EVAL_SAMPLES,
                      seed: int = config.RANDOM_SEED, device=config.DEVICE) -> torch.Tensor:
    """Generate n images in [-1, 1] on CPU (uint8-safe batches of EVAL_BATCH_SIZE).

    TODO (Plan step 10):
      - unconditional (text_emb None): G(z)
      - conditional: cycle through the held-out TEST-class caption embeddings text_emb (M, 512) so
        every model is scored on the SAME captions; z from torch.Generator().manual_seed(seed)
      - G.eval(); return torch.cat(batches)
    """
    raise NotImplementedError


def compute_fid(real: torch.Tensor, fake: torch.Tensor, device=config.DEVICE) -> float:
    """Frechet Inception Distance between real and generated images (both (N, 3, 64, 64) in [-1, 1]). Lower is better.

    TODO (Plan step 10): needs `torchmetrics[image]` (see OUTLINE.md, Environment).
      from torchmetrics.image.fid import FrechetInceptionDistance
      fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)   # normalize: floats in [0, 1]
      update with to_image_range(real) (real=True) and to_image_range(fake) (real=False) in batches
      return fid.compute().item()
    Use the SAME real set (test-class images) and the same N for every experiment; FID is biased by N.
    """
    raise NotImplementedError


def clip_score(images: torch.Tensor, text_emb: torch.Tensor, clip_model=None, device=config.DEVICE) -> float:
    """Mean cosine similarity between CLIP image embeddings of `images` and their caption embeddings. Higher is better.

    TODO (Plan step 10):
      - resize images to 224 and normalise with CLIP's mean/std (or use the preprocess from load_clip)
      - img = F.normalize(clip_model.encode_image(x).float(), dim=-1); return (img * text_emb).sum(-1).mean()
      - Baselines for the table: real test images with their true captions (upper bound) and E0
        images with random captions (chance level).
    Caveat for §3.3: CLIP is also the conditioning encoder, so this score can flatter the model;
    cross-check with a second encoder (e.g. CLIP ViT-B/16) if time allows.
    """
    raise NotImplementedError


@torch.no_grad()
def discriminator_pair_accuracy(D, loader, G, device=config.DEVICE) -> dict:
    """Accuracy of a conditional D on (real, match), (real, wrong), (fake, match) pairs (Project 1's metric, kept for comparison).

    TODO (Plan step 10): sign(logit) > 0 means "real and matching"; return dict(acc_real, acc_wrong, acc_fake).
    """
    raise NotImplementedError


def summarize_seeds(results: pd.DataFrame, metric: str, baseline: str = "E1_cond") -> pd.DataFrame:
    """Mean +- std of `metric` per experiment over SEEDS, and Welch's t-test vs `baseline` (report §3.2).

    TODO (Plan step 11):
      results columns: experiment, seed, fid, clip_score
      from scipy.stats import ttest_ind
      for each experiment: mean, std, p = ttest_ind(exp_vals, base_vals, equal_var=False).pvalue
    With 3 seeds the test has little power: report the effect size and say so in §3.3.
    """
    raise NotImplementedError
