# Project 1 (rebuild): Text-to-Image DCGAN — MSCS 435/635 Advanced Machine Learning
**Due:** not stated · **Type:** ML (Python 3, PyTorch, MPS) · **Points:** not stated
See [DIAGRAM.md](DIAGRAM.md) for diagrams and [report/REPORT_OUTLINE.md](report/REPORT_OUTLINE.md) for the write-up plan.

## Summary
This rebuilds the Project 1 text-to-image DCGAN so that it actually generates images. The end goal is unchanged: a caption goes in, a matching RGB image comes out, and there is a front end. The rebuild drops the custom architecture (complex EML layers, ColorFocus, a DINOv3 discriminator) and the uncurated YFCC100M data. Instead it builds step by step on proven pipelines: the class L14 DCGAN, then a conditional GAN, then GAN-CLS (Reed et al., 2016). Each step adds one change, and the work is written up in the course Project Report Format.

## Why v1 failed (the fixes this rebuild makes)
| v1 problem (from code + report) | v2 fix |
|---|---|
| A new, untested architecture: complex EML convs, ColorFocus, ConvEmbedding, DynoEmbed, EMLSkip | The L14 / Radford et al. DCGAN, unchanged (`src/model.py` Generator, Discriminator) |
| ~10 techniques at once (FiLM, projection, spectral norm, hinge, mismatch, 4 caption levels, frozen DINOv3) with no baseline | E0 → E1 → E2 → E3, one change per experiment (`config.EXPERIMENTS`) |
| Frozen DINOv3 discriminator rejected 94–100% of fakes from epoch 1, so G got no useful gradient | A plain DCGAN discriminator trained from scratch; D(x) and D(G(z)) logged every epoch |
| Uncurated YFCC100M (8.5 TiB, every kind of scene) at 256–512 px | Oxford-102 Flowers (8,189 images, one domain, human captions) at 64×64 |
| Captions from a local LLM, never checked | Human-written captions from Reed et al. (10 per image) |
| Only metric was D accuracy | FID + CLIP score + D pair accuracy, 3 seeds, a significance test |
| Monolithic, AI-generated, CUDA-only scripts, no tests | Small `src/` modules, pytest, MPS/CUDA/CPU, a 100 MB tensor cache instead of a JPEG loader |

## Direction & class methods
- **Vision:** The Target tier. An unconditional DCGAN (E0) is verified first. Then CLIP conditioning (E1), the GAN-CLS matching-aware loss (E2) and GAN-INT (E3) are compared on FID and CLIP score over 3 seeds, and a Gradio demo is built. The end goal of the Project 1 proposal is kept.
- **Process:** Small working milestones. Milestone 0 is **fully implemented** and must produce flowers before any new piece is added. Later milestones are detailed TODO stubs.
- **In-class methods & notation to follow:**
  - DCGAN architecture and training loop from `MSCS435-AML-main/code/L14/04_02_dcgan-celeba.ipynb` and `helper_train.py::train_gan_v1`: one D step then one G step, BCE-with-logits, flipped labels for G, Adam (lr 2e-4, β = (0.5, 0.999)), fixed-noise grid every epoch.
  - Seeding from `helper_utils.py` (`set_all_seeds`, `set_deterministic`).
  - L14 slides: the generator/discriminator minimax game.
  - Report structure from `assignment/project_report_format.png`.
- **Constraints:** An individual project per the course page (Project 1 was done as a group; confirm with the instructor). Compute is an Apple M2 Pro with 16 GB (MPS), about 17 s per epoch for E0, measured. Allowed topic: §3 Generative Modeling (GANs, conditional generation).

## Requirements
- [ ] Topic from a course section: §3 Generative Modeling (GANs, conditional generation)
- [ ] Working text → image generator (the Project 1 end goal)
- [ ] **M0:** unconditional DCGAN produces recognizable flowers at 64×64 (E0)
- [ ] **M1:** CLIP-conditioned DCGAN (E1)
- [ ] **M2:** GAN-CLS matching-aware D (E2) and GAN-INT (E3)
- [ ] **M3:** FID + CLIP score for E0–E3 × 3 seeds, with a significance test
- [ ] **M4:** front end (text box → images), `app.py`
- [ ] Instructor checkpoint review (be ready to explain GAN loss, DCGAN rules, GAN-CLS)
- [ ] Brief write-up in the Project Report Format (§1–6 + bibliography): `report/REPORT_OUTLINE.md`
- [ ] Demonstration (live `app.py` + sample figures)

## Dataset
- **Oxford-102 Flowers** (Nilsback & Zisserman, 2008): 8,189 images, 102 classes (40–258 per class), variable size (~500 px). Downloaded by `torchvision.datasets.Flowers102`.
- **Captions:** Reed et al. (2016) `text_c10`, 10 human captions per image, downloaded manually (see `data/README.md`).
- **Split:** by class, as in Reed et al.: 82 train/val classes, 20 held-out test classes (zero-shot). E0 is unconditional and trains on all images.
- **Preprocessing:** resize the short side to 64 and center-crop once into a uint8 cache (`data/processed/flowers64.pt`, ~100 MB); random horizontal flip during training; scale to [−1, 1].
- **Known issues:** class imbalance (40–258 images per class); captions describe color and shape but not background; a few captions exceed CLIP's 77 tokens (truncated).

## Approach
1. **E0 (baseline, the tested pipeline):** DCGAN, z ∈ ℝ¹⁰⁰ → 64×64×3. This shows the data, loop and hardware work. Done when the fixed-noise grids show flowers.
2. **E1 (+ text):** frozen CLIP ViT-B/32 text embedding (512) → Linear + BN + LeakyReLU → 128, concatenated to z in G. In D the 128-d embedding is replicated to 4×4 and concatenated before the final conv (Reed et al. Fig. 2). Plain conditional GAN loss.
3. **E2 (GAN-CLS):** D also sees (real image, wrong caption) as fake, so it must check the text, not just realism. This is the published version of v1's "mismatched captions" idea.
4. **E3 (GAN-INT-CLS):** G is also trained on interpolated caption embeddings (β = 0.5). There is no real image for these, so they are used only in the G step.
- **Why this fits:** a DCGAN works on narrow domains at 64×64, and GAN-CLS is the standard first text-to-image GAN, with published flower results. CLIP replacing char-CNN-RNN is the one modern change, and the report states and justifies it.

## Evaluation
- **FID** (`torchmetrics`): 5,000 generated images vs the real test-class images. Lower is better; it measures realism and diversity.
- **CLIP score:** cosine(CLIP image embedding, caption embedding). Higher is better; it measures text alignment. Report it alongside a real-image upper bound and a shuffled-caption chance level.
- **D pair accuracy** on (real, match) / (real, wrong) / (fake, match), the Project 1 metric, kept for comparison.
- **Validation:** fixed held-out test classes; the same captions, z seeds and N for every model; 3 training seeds; Welch t-test plus effect size.
- **What "good" looks like:** E0 shows flowers by about epoch 25. D(x) and D(G(z)) stay apart without saturating (D(G(z)) not pinned at 0, unlike v1). E1–E3 change color and shape when the caption changes.

## Experiments
| # | Change | Metric | Result | Notes |
|---|---|---|---|---|
| E0 | DCGAN, unconditional | FID, samples | | must work before E1 |
| E1 | + CLIP text conditioning | FID, CLIP score | | H1 |
| E2 | + matching-aware D (GAN-CLS) | FID, CLIP score, D pair acc | | H2 |
| E3 | + caption interpolation (GAN-INT) | FID, CLIP score | | H3 |
| (opt.) | + one Project 1 idea, e.g. DINOv3 features in D | FID, CLIP score | | stretch only, after E3 |

## Write-up / deliverables
- Report: `report/REPORT_OUTLINE.md` → final doc (Intro, Problem & Algorithm, Experimental Evaluation, Related Work, Future Work, Conclusion, Bibliography).
- Figures: produced by `notebooks/02_results.ipynb` into `reports/figures/` (loss curves, sample progress, FID/CLIP bars, caption grid, interpolation).
- Demo: `python app.py`.

## Plan
**Milestone 0: proven baseline (IMPLEMENTED; run and verify)**
1. Run the tests (`pytest -q`): the 10 milestone-0 tests pass.
2. `python train.py --epochs 5`: downloads the images, builds the cache, trains 5 epochs, and saves `reports/figures/samples/E0_dcgan_s42/epoch_005.png`. Blobs with flower colors should already be visible.
3. `caffeinate -i python train.py`: full 100 epochs (~30 min). Check the loss plot and samples in `notebooks/02_results.ipynb` §1–2. **Do not continue until the grid shows flowers.**
4. `notebooks/01_exploration.ipynb` §1–3: class balance and a real sample grid (report §3.1).

**Milestone 1: text conditioning (E1)**
5. Download the Reed captions → `data.load_captions`, `data.class_split`; `text_encoder.load_clip`, `encode_text`, `build_text_cache`, `load_text_cache`. Notebook 01 §4–5.
6. `data.FlowersCaptions` and the conditional branch of `data.get_dataloader`.
7. `model.TextProjection`, `CondGenerator`, `CondDiscriminator`; the `mode == "cond"` branch in `trainer.train_one_epoch`; `fixed_t` in `trainer.train`. Run E1 for 5 epochs and check that the samples change with the caption.

**Milestone 2: GAN-CLS (E2, E3)**
8. `trainer.d_loss_gan_cls` and the `gan_cls` branch (wrong-caption pairs).
9. `trainer.interpolate_text` and the GAN-INT term in the G step.

**Milestone 3: evaluation**
10. `evaluate.generate_for_eval`, `compute_fid`, `clip_score`, `discriminator_pair_accuracy`; `evaluate_all.py`.
11. Train E0–E3 × seeds 42/43/44 (~12 runs × 30 min); `summarize_seeds`; notebook 02 §3 and §6.

**Milestone 4: demo + report**
12. `generate.py` (`load_generator`, `generate`, `interpolate`), then `app.py` (Gradio); notebook 02 §4–5.
13. Write the report from `report/REPORT_OUTLINE.md`; rehearse the demo.

## Files
| File | Purpose |
|---|---|
| `src/config.py` | Paths, device (mps), DCGAN hyperparameters, `EXPERIMENTS` E0–E3 |
| `src/utils.py` | `set_all_seeds`, `set_deterministic` (class helper_utils), checkpoint save/load |
| `src/data.py` | Flowers102 download → 64×64 uint8 cache; `FlowersImages` (M0); captions, class split, `FlowersCaptions` (TODO) |
| `src/text_encoder.py` | Frozen CLIP ViT-B/32: encode and cache caption embeddings (TODO) |
| `src/model.py` | `Generator`, `Discriminator`, `weights_init` (M0); `TextProjection`, `CondGenerator`, `CondDiscriminator` (TODO) |
| `src/trainer.py` | Losses, `train_one_epoch` (dcgan done; cond / gan_cls TODO), `train` (logging, grids, resumable checkpoints) |
| `src/evaluate.py` | `save_grid`, `plot_losses`, `plot_sample_progress` (done); FID, CLIP score, D pair accuracy, seed stats (TODO) |
| `train.py` | CLI: `--experiment --seed --epochs`, resumes automatically |
| `evaluate_all.py` | Score all runs → `reports/results.csv` (TODO) |
| `generate.py` | Caption → image grid, caption interpolation (TODO) |
| `app.py` | Gradio front end (TODO) |
| `notebooks/01_exploration.ipynb` | Dataset facts and figures (report §2.1, §3.1) |
| `notebooks/02_results.ipynb` | All report figures (§3.2) |
| `report/REPORT_OUTLINE.md` | Section-by-section plan of the write-up, from the course format |
| `tests/test_*.py` | pytest: M0 tests pass now; M1+ tests fail until their TODOs are done |
| `assignment/` | Course project page, report format, v1 proposal and v1 report |

## Environment
- Python: `~/.venvs/aml-dcgan-v2` (uv venv over Anaconda base with `torchmetrics[image]` and `gradio`; already created, and VS Code points to it). Notebook kernel: **Python (aml-dcgan-v2)**.
- `clip` (OpenAI) already works from Anaconda base. The first `clip.load` downloads ViT-B/32 weights (~340 MB).
- The first `compute_fid` downloads Inception weights (~100 MB).
- The v1 repo (`../AML-DCGAN-main 2/`) is untouched. Nothing from it is imported.

## Key concepts
- **GAN minimax:** D learns real vs fake and G learns to fool D. In practice G minimizes BCE(D(G(z)), 1) (non-saturating), because log(1 − D(G(z))) has no gradient when D wins easily. That is exactly what happened in v1.
- **DCGAN rules (Radford et al.):** strided convs instead of pooling; BatchNorm in both networks (not on D's input or G's output); ReLU in G with tanh output; LeakyReLU(0.2) in D; init N(0, 0.02); Adam with β₁ = 0.5.
- **Conditional GAN:** give the same side information (here text) to both G and D.
- **GAN-CLS:** a plain conditional D can ignore the text and judge only realism. Adding (real image, wrong text) as negatives forces it to judge the match.
- **GAN-INT:** interpolated text embeddings are plausible captions with no image. Training G on them fills the gaps between captions.
- **Frozen encoder:** CLIP's text features are precomputed once, which costs nothing per batch and cannot be "unlearned".
- **FID:** distance between Gaussian fits of Inception features of real and fake images. It depends on N, so always use the same N.

## Testing
- **Normal:** G(z) has shape (B, 3, 64, 64) in [−1, 1]; D gives one logit per image; one epoch updates both networks with finite losses; the PNG grid and loss plot are written.
- **Edge:** a non-square image becomes 64×64 after the transform; an undecided D (logit 0) gives loss log 2 for DCGAN and 2·log 2 for GAN-CLS; a perfect D gives ≈ 0.
- **Data integrity:** captions parse to the Reed layout; the class split is disjoint (82/20); the wrong caption always comes from a different class; text-cache row order equals image-cache row order (assert in `load_text_cache`).
- **Error cases:** an unknown experiment or mode raises; a missing cache is built automatically; a run resumes from `last.pt`.
- **Metric sanity:** FID(x, x) ≈ 0; the CLIP score for real images with true captions is above the shuffled-caption score.

## Questions / ambiguities
- Course page says "two **individual** projects", but v1 was a group project. Is a group rebuild allowed, and does it count as Project 1 or 2?
- Is a resubmission of Project 1 accepted, and is there a new due date?
- Is switching from YFCC100M to Oxford-102 acceptable given the original proposal? (The end goal, text → image, is unchanged.)
- Is a 64×64 output acceptable for the demonstration?
