"""Oxford-102 Flowers images (+ Reed et al. captions) -> 64x64 tensors -> DataLoaders.

Milestone 0 (IMPLEMENTED): download images with torchvision, resize/crop ONCE to 64x64 and cache
them as a single uint8 tensor (~100 MB). Training then never decodes a JPEG, which is what made the
Project 1 loader slow and error-prone.

Milestone 1+ (TODO): load the 10 captions per image, split by class (Reed et al.: 82 train/val
classes, 20 held-out test classes) and return (image, caption embedding, mismatched embedding).
"""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm

from src import config


# ============================================================ Milestone 0: images
def image_transform(size: int = config.IMAGE_SIZE) -> transforms.Compose:
    """PIL image -> uint8 tensor (3, size, size): resize the short side, then center crop."""
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(size),
        transforms.PILToTensor(),
    ])


def build_image_cache(size: int = config.IMAGE_SIZE, root: Path = config.DATA_RAW,
                      out: Path = config.IMAGE_CACHE, force: bool = False) -> Path:
    """Download Oxford-102 (all 8,189 images, ~330 MB) and save {images, labels, ids} to `out`.

    images: uint8 (N, 3, size, size); labels: int64 (N,) in 0..101; ids: ["image_00001", ...]
    sorted by id so row i always means the same image (captions are matched by id later).
    """
    out = Path(out)
    if out.exists() and not force:
        return out
    tf = image_transform(size)
    rows = []
    for split in ("train", "val", "test"):  # torchvision's splits are tiny; a GAN trains on all images
        ds = datasets.Flowers102(root=root, split=split, download=True)
        rows += list(zip(ds._image_files, ds._labels))
    rows.sort(key=lambda r: r[0].stem)
    images = torch.empty(len(rows), 3, size, size, dtype=torch.uint8)
    for i, (path, _) in enumerate(tqdm(rows, desc=f"Caching {size}x{size} images")):
        with Image.open(path) as im:
            images[i] = tf(im.convert("RGB"))
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(images=images, labels=torch.tensor([r[1] for r in rows]),
                    ids=[r[0].stem for r in rows], size=size), out)
    return out


def load_image_cache(path: Path = config.IMAGE_CACHE) -> dict:
    """Load the dict written by build_image_cache (build it first if missing)."""
    if not Path(path).exists():
        build_image_cache(out=path)
    return torch.load(path, weights_only=False)


def to_model_range(x_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 [0, 255] -> float [-1, 1] (same as Normalize((0.5,)*3, (0.5,)*3) in the class notebook)."""
    return x_uint8.float().div(127.5).sub(1.0)


def to_image_range(x: torch.Tensor) -> torch.Tensor:
    """float [-1, 1] -> float [0, 1] for saving/plotting."""
    return x.add(1.0).div(2.0).clamp(0.0, 1.0)


class FlowersImages(Dataset):
    """Unconditional dataset for E0: returns (image in [-1, 1], class label). Random horizontal flip."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, flip: bool = True):
        assert images.dtype == torch.uint8 and images.ndim == 4
        self.images, self.labels, self.flip = images, labels, flip

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        x = to_model_range(self.images[i])
        if self.flip and torch.rand(()) < 0.5:
            x = x.flip(-1)
        return x, self.labels[i]


# ============================================================ Milestone 1+: captions
def load_captions(captions_dir: Path = config.CAPTIONS_DIR) -> dict[str, list[str]]:
    """Return {"image_00001": [10 captions], ...} from Reed et al.'s text_c10 folder.

    Source: Reed et al. (2016) "cvpr2016_flowers" release (github.com/reedscot/icml2016, README links);
    unpack so that data/raw/text_c10/class_00001/image_06734.txt exists. Each .txt has one caption
    per line. Record the exact download link and license in data/README.md.

    TODO (Plan step 5):
      1. for p in sorted(captions_dir.glob("class_*/image_*.txt")):
      2.     lines = [" ".join(l.split()) for l in p.read_text().splitlines() if l.strip()]
      3.     out[p.stem] = lines
      4. assert len(out) == 8189 and every entry has >= config.CAPTIONS_PER_IMAGE captions
    """
    raise NotImplementedError


def class_split(labels: torch.Tensor, captions_dir: Path = config.CAPTIONS_DIR) -> dict[str, torch.Tensor]:
    """Return {"train": row indices, "test": row indices} split BY CLASS (zero-shot, as in Reed et al.).

    TODO (Plan step 5):
      1. Read captions_dir.parent / "trainvalclasses.txt" and "testclasses.txt" from the Reed release
         (lines like "class_00001" -> label 0).
      2. If those files are missing, fall back to a fixed split: torch.randperm(102, generator seeded
         with config.RANDOM_SEED), first 82 classes = train, last 20 = test. Note which one you used.
      3. Return index tensors of the rows whose label is in each set. No class may be in both.
    """
    raise NotImplementedError


class FlowersCaptions(Dataset):
    """Conditional dataset for E1-E3: returns (image, t_match, t_wrong, row index).

    t_match: CLIP embedding (512,) of one of the image's 10 captions, picked at random each call.
    t_wrong: CLIP embedding of a caption from a DIFFERENT class (the GAN-CLS "real image, wrong
             text" negative). Picking from another class avoids "wrong" captions that still fit.
    """

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, text_emb: torch.Tensor,
                 rows: torch.Tensor, flip: bool = True):
        # text_emb: (N_images, 10, 512) float16 from src.text_encoder.load_text_cache(), row-aligned with images
        # TODO (Plan step 6): store args; build self.by_class = {label: tensor of rows} for wrong-caption sampling
        raise NotImplementedError

    def __len__(self) -> int:
        # TODO: number of rows in this split
        raise NotImplementedError

    def __getitem__(self, i: int):
        # TODO (Plan step 6):
        #   1. r = self.rows[i]; x = to_model_range(self.images[r]) (+ random flip like FlowersImages)
        #   2. t_match = self.text_emb[r, random caption index].float()
        #   3. pick a row r2 with labels[r2] != labels[r]; t_wrong = self.text_emb[r2, random index].float()
        #   4. return x, t_match, t_wrong, r
        raise NotImplementedError


# ============================================================ loaders
def get_dataloader(conditional: bool, batch_size: int = config.BATCH_SIZE, split: str = "train",
                   shuffle: bool = True) -> DataLoader:
    """DataLoader for an experiment. drop_last=True keeps BatchNorm statistics stable.

    E0 (conditional=False) trains on ALL images (no captions, so no zero-shot split is needed).
    E1-E3 (conditional=True) TODO: use class_split(...)[split] rows and FlowersCaptions.
    """
    cache = load_image_cache()
    if not conditional:
        ds = FlowersImages(cache["images"], cache["labels"])
    else:
        # TODO (Plan step 6):
        #   from src.text_encoder import load_text_cache
        #   rows = class_split(cache["labels"])[split]
        #   ds = FlowersCaptions(cache["images"], cache["labels"], load_text_cache(cache["ids"]), rows)
        raise NotImplementedError("Conditional data comes in milestone 1 (Plan step 6)")
    g = torch.Generator().manual_seed(config.RANDOM_SEED)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle,
                      num_workers=config.NUM_WORKERS, generator=g)
