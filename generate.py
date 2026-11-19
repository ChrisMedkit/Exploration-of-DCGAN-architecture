"""Text -> image with a trained conditional generator (used by app.py and for report figures).

    python generate.py "this flower has bright yellow petals and a dark center" --n 16 --seed 0
"""

import argparse
from pathlib import Path

import torch

from src import config


def load_generator(checkpoint: Path, device=config.DEVICE):
    """Rebuild the right G for the checkpoint's experiment and load its weights in eval mode.

    TODO (Plan step 12):
      ck = utils.load_checkpoint(checkpoint, device)
      G, _ = model.build_models(config.EXPERIMENTS[ck["experiment"]]["conditional"], device)
      G.load_state_dict(ck["G"]); return G.eval()
    """
    raise NotImplementedError


@torch.no_grad()
def generate(prompt: str, G, clip_model, n: int = 16, seed: int = 0, device=config.DEVICE) -> torch.Tensor:
    """Return n images (n, 3, 64, 64) in [-1, 1] for one caption.

    TODO (Plan step 12):
      t = text_encoder.encode_text([prompt], clip_model, device).to(device).expand(n, -1)
      z = torch.randn(n, G.latent_dim, 1, 1, generator=torch.Generator().manual_seed(seed)).to(device)
      return G(z, t).cpu()
    """
    raise NotImplementedError


def interpolate(prompt_a: str, prompt_b: str, G, clip_model, steps: int = 8, seed: int = 0) -> torch.Tensor:
    """Fixed z, caption embedding moved from prompt_a to prompt_b (report figure: latent/text interpolation).

    TODO (Plan step 12): t = (1-a)*t_a + a*t_b for a in linspace(0, 1, steps), normalise, same z for all.
    """
    raise NotImplementedError


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("prompt")
    p.add_argument("--checkpoint", type=Path, default=config.MODELS / "E3_gan_int_cls_s42" / "last.pt")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=config.FIGURES / "generated.png")
    a = p.parse_args()
    # TODO (Plan step 12): G = load_generator(a.checkpoint); clip_model, _ = text_encoder.load_clip()
    #                      evaluate.save_grid(generate(a.prompt, G, clip_model, a.n, a.seed), a.out, nrow=4)
    raise NotImplementedError


if __name__ == "__main__":
    main()
