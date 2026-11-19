"""Frozen CLIP ViT-B/32 text encoder (Milestone 1, TODO).

CLIP is used ONLY as a fixed feature extractor: encode every caption once, cache to disk, never
backprop into it. This replaces Reed et al.'s char-CNN-RNN encoder (the one change to their recipe,
justified in report §2.2). `import clip` is already available in Anaconda base.
"""

from pathlib import Path

import torch

from src import config


def load_clip(device: torch.device | str = config.DEVICE):
    """Return (model, preprocess) for config.CLIP_MODEL in eval mode with requires_grad False.

    TODO (Plan step 5):
      import clip
      model, preprocess = clip.load(config.CLIP_MODEL, device=device, jit=False)
      model.eval().requires_grad_(False); return model, preprocess
    """
    raise NotImplementedError


@torch.no_grad()
def encode_text(texts: list[str], model=None, device: torch.device | str = config.DEVICE) -> torch.Tensor:
    """List of captions -> L2-normalised CLIP embeddings (len(texts), 512) float32 on CPU.

    TODO (Plan step 5):
      tokens = clip.tokenize(texts, truncate=True).to(device)   # 77-token limit
      z = model.encode_text(tokens).float()
      return torch.nn.functional.normalize(z, dim=-1).cpu()
    """
    raise NotImplementedError


def build_text_cache(captions: dict[str, list[str]], ids: list[str], out: Path = config.TEXT_CACHE,
                     batch_size: int = 512, force: bool = False) -> Path:
    """Encode the first CAPTIONS_PER_IMAGE captions of every image; save (N, 10, 512) float16 row-aligned with ids.

    TODO (Plan step 5):
      1. skip if out exists and not force
      2. flat = [captions[i][k] for i in ids for k in range(config.CAPTIONS_PER_IMAGE)]
      3. encode in batches of batch_size with encode_text, cat, .view(len(ids), 10, 512).half()
      4. torch.save(dict(emb=..., ids=ids, model=config.CLIP_MODEL), out)
    Size check: 8,189 x 10 x 512 x 2 bytes ~ 84 MB. Takes a few minutes on MPS.
    """
    raise NotImplementedError


def load_text_cache(ids: list[str], path: Path = config.TEXT_CACHE) -> torch.Tensor:
    """Load the cached embeddings and check they are in the same row order as the image cache.

    TODO (Plan step 5): d = torch.load(path); assert d["ids"] == ids; return d["emb"]
    """
    raise NotImplementedError
