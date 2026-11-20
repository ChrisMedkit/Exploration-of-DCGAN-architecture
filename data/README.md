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
- **Captions:** Reed, Akata, Lee & Schiele (2016), *Learning Deep Representations of Fine-Grained Visual Descriptions*. Downloaded 2026-11-20 as `cvpr2016_flowers.tar.gz` (588 MB) from the Google Drive link in the README of https://github.com/reedscot/cvpr2016 (`https://drive.google.com/open?id=0B0ywwgffWnLLcms2WWJQRFNSWXM`). Only `text_c10/class_*/image_*.txt` and the `*classes.txt` split files were extracted into `data/raw/`. The bundled Torch embeddings and images were not used. No license is stated on the README; cite the paper.

## Dataset facts
- Images per class: min 40, median 66, max 258.
- Captions: 10 per image, 81,890 in total; mean 13.5 words, max 65 words. One caption exceeds CLIP's 77-token limit and is truncated.
- Class split (Reed's `trainvalclasses.txt` / `testclasses.txt`): 82 train classes = 7,034 images; 20 held-out test classes = 1,155 images.
- CLIP ViT-B/32 cosine similarity: about 0.81 between two captions of the same image and about 0.78 for a caption from another class. Flower captions sit close together in CLIP space, so the learned projection has to separate them.
