"""Frozen CLIP ViT-B/32 text encoder (Milestone 1).

CLIP is used ONLY as a fixed feature extractor: encode every caption once, cache to disk, never
backprop into it. This replaces Reed et al.'s char-CNN-RNN encoder (the one change to their recipe,
justified in report §2.2). OpenAI's `clip` package is already available in Anaconda base.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src import config


def load_clip(device: torch.device | str = config.DEVICE):
    """Return (model, preprocess) for config.CLIP_MODEL in eval mode with gradients off.

    First call downloads the ViT-B/32 weights (~340 MB) to ~/.cache/clip.
    """
    import clip

    model, preprocess = clip.load(config.CLIP_MODEL, device=device, jit=False)
    model.eval().requires_grad_(False)
    return model, preprocess


@torch.no_grad()
def encode_text(texts: list[str], model, device: torch.device | str = config.DEVICE) -> torch.Tensor:
    """List of captions -> L2-normalised CLIP embeddings (len(texts), 512) float32 on CPU."""
    import clip

    tokens = clip.tokenize(texts, truncate=True).to(device)  # 77-token limit; longer captions are cut
    z = model.encode_text(tokens).float()
    return F.normalize(z, dim=-1).cpu()


def build_text_cache(captions: dict[str, list[str]], ids: list[str], out: Path = config.TEXT_CACHE,
                     batch_size: int = 512, force: bool = False, device=config.DEVICE) -> Path:
    """Encode the first CAPTIONS_PER_IMAGE captions of every image; save (N, 10, 512) float16 row-aligned with ids.

    Size: 8,189 x 10 x 512 x 2 bytes ~ 84 MB.
    """
    out = Path(out)
    if out.exists() and not force:
        return out
    k = config.CAPTIONS_PER_IMAGE
    flat = [captions[i][j] for i in ids for j in range(k)]
    model, _ = load_clip(device)
    chunks = [encode_text(flat[s:s + batch_size], model, device)
              for s in tqdm(range(0, len(flat), batch_size), desc="Encoding captions with CLIP")]
    emb = torch.cat(chunks).view(len(ids), k, -1).half()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(emb=emb, ids=list(ids), model=config.CLIP_MODEL), out)
    return out


def load_text_cache(ids: list[str], path: Path = config.TEXT_CACHE) -> torch.Tensor:
    """Load cached embeddings (N, 10, 512) float16, checking row order matches the image cache.

    Builds the cache on first use (reads data/raw/text_c10, runs CLIP once).
    """
    path = Path(path)
    if not path.exists():
        from src.data import check_captions, load_captions

        caps = load_captions()
        check_captions(caps, ids)
        build_text_cache(caps, ids, path)
    d = torch.load(path, weights_only=False)
    if d["ids"] != list(ids):
        raise ValueError("text cache rows do not match the image cache; rebuild with force=True")
    if d["model"] != config.CLIP_MODEL:
        raise ValueError(f"text cache was built with {d['model']}, config wants {config.CLIP_MODEL}")
    return d["emb"]
