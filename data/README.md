# Data

| Folder | Contents | Made by |
|---|---|---|
| `raw/flowers-102/` | `jpg/image_00001.jpg … image_08189.jpg`, `imagelabels.mat`, `setid.mat` (~330 MB) | `src.data.build_image_cache()` via `torchvision.datasets.Flowers102(download=True)` |
| `raw/text_c10/` | `class_00001/image_06734.txt` … — 10 captions per image | manual download (below) |
| `raw/trainvalclasses.txt`, `raw/testclasses.txt` | Reed et al. class split (82 / 20) | same download |
| `processed/flowers64.pt` | `images` uint8 (8189, 3, 64, 64), `labels`, `ids` (~100 MB) | `build_image_cache()` |
| `processed/clip_text_vitb32.pt` | CLIP ViT-B/32 caption embeddings (8189, 10, 512) float16 (~84 MB) | `src.text_encoder.build_text_cache()` |

None of this is committed (see `.gitignore`).

## Sources
- **Images:** Nilsback & Zisserman (2008), *Automated Flower Classification over a Large Number of Classes*. https://www.robots.ox.ac.uk/~vgg/data/flowers/102/ — 8,189 images, 102 classes (40–258 images per class).
- **Captions:** Reed, Akata, Lee & Schiele (2016), *Learning Deep Representations of Fine-Grained Visual Descriptions*. Their `cvpr2016_flowers` release, linked from the README of https://github.com/reedscot/icml2016. Unpack `text_c10/` (and the class split files) into `data/raw/`.
  TODO: record the exact link you downloaded from and its license/terms here.

## Facts to fill in after `notebooks/01_exploration.ipynb`
- Images per class (min / median / max): TODO
- Caption length in words (mean, 95th percentile), share over CLIP's 77-token limit: TODO
- Train / test classes and images under the class split: TODO
