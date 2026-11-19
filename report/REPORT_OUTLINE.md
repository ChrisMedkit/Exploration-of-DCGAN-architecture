# Report outline: Text-to-Image DCGAN (GAN-CLS) on Oxford-102 Flowers

This follows the course **Project Report Format** (`assignment/project_report_format.png`), one heading per required section. Each section lists what it must cover, which project file or figure supplies it, and a TODO for your group's prose. The bullets are notes, not text to paste in.

---

## 1. Introduction
*Format asks:* motivate the problem, the basic approach, related work in brief, and a summary of the results.
- Problem: generate an image from a text description (same end goal as the Project 1 proposal).
- What changed from Project 1, and why (one short paragraph, honest):
  - v1 invented a new architecture (complex EML layers, ColorFocus, a DINOv3 discriminator) and trained it on uncurated YFCC100M at 256–512 px. It produced textures, not images, and with no baseline the cause could not be isolated.
  - v2 starts from a **proven pipeline** (class L14 DCGAN → Reed et al. GAN-CLS) on a **narrow, captioned dataset** at **64×64**, and adds one change per experiment.
- Approach in one sentence: a DCGAN conditioned on frozen CLIP text embeddings, with a matching-aware discriminator.
- Results summary: TODO after §3.2 (one sentence per hypothesis).

## 2. Problem Definition and Algorithm
### 2.1 Task Definition
*Format asks:* inputs and outputs, stated formally, and why the problem matters.
- Input: caption *c* (English, ≤77 CLIP tokens) and noise **z** ∈ ℝ¹⁰⁰, **z** ~ N(0, I).
- Output: image **x̂** = G(**z**, φ(*c*)) ∈ [−1, 1]^{3×64×64}, where φ is the frozen CLIP ViT-B/32 text encoder (512-d).
- Goal: **x̂** should look like a real flower (realism) and match *c* (text alignment), and different **z** should give different flowers (diversity).
- Zero-shot setting: test captions come from 20 flower classes never seen in training.
- Why it matters: TODO (creative tools, data augmentation; Project 1's motivation still applies).

### 2.2 Algorithm Definition
*Format asks:* the algorithm in reasonable detail, pseudocode, and a traced concrete example.
- Architecture: the DIAGRAM.md "Model architecture" figure (G and D with shapes).
- Pseudocode: Reed et al. Algorithm 1 (GAN-CLS) in your own notation, matching `src/trainer.py::train_one_epoch`:
  - D sees three kinds of pair: (real image, right text) → 1; (real image, wrong text) → 0; (fake image, right text) → 0.
  - G loss: BCE(D(G(z, t), t), 1); with GAN-INT, add the same loss on t_int = β·t + (1−β)·t′.
- Worked trace (the format asks for an intuitive example): take one caption, e.g. *"this flower has five pink petals and a yellow center"*:
  1. CLIP gives t ∈ ℝ⁵¹² → projection → 128-d.
  2. Concatenate with z → (228, 1, 1) → 4×4×512 → … → 64×64×3.
  3. D's features for the fake → 4×4×512; t replicated to 4×4×128 → concat → logit.
  4. Show which of the three pairs D gets wrong early in training and late in training.
  Use a real batch from notebooks/02_results.ipynb.
- Design choices, each with a citation: DCGAN rules (Radford et al.: strided convs, BatchNorm, ReLU/LeakyReLU, Adam with β₁ = 0.5); CLIP in place of char-CNN-RNN (the one deliberate change to Reed et al.).

## 3. Experimental Evaluation
### 3.1 Methodology
*Format asks:* criteria, hypotheses, independent and dependent variables, data, what was collected, and comparisons with competing methods.
- Hypotheses:
  - **H1**: text conditioning (E1) gives higher CLIP score than chance (E0 images with shuffled captions), without worse FID.
  - **H2**: the matching-aware loss (E2) improves CLIP score over plain conditioning (E1).
  - **H3**: GAN-INT (E3) improves FID over E2 (smoother text manifold, as Reed et al. report).
- Independent variable: experiment E0–E3 (`config.EXPERIMENTS`; exactly one change per step). Everything else is fixed: architecture, learning rates, epochs, batch size and data.
- Dependent variables: **FID** (realism/diversity, lower is better); **CLIP score** (text–image cosine, higher is better); D pair accuracy (kept for comparison with Project 1); training losses.
- Data: Oxford-102 train/val classes (82) for training; 20 held-out test classes for evaluation (5,000 generated images, the same captions and seeds for every model). Why it is realistic: TODO.
- Repeats: 3 seeds per experiment (`config.SEEDS`).
- Reference rows: real test images with their true captions (CLIP upper bound); shuffled captions (chance); published GAN-INT-CLS flower results (cite the paper's own figures or numbers).

### 3.2 Results
*Format asks:* quantitative results, graphs rather than tables where possible, and statistical significance.
- Fig. A: E0 loss curves and D(x) vs D(G(z)) — `reports/figures/E0_dcgan_s42_losses.png`.
- Fig. B: E0 samples at epochs 1/10/25/50/100 — `…_progress.png`.
- Fig. C: FID and CLIP score per experiment, mean ± std over seeds (bar chart with error bars).
- Table: `reports/results.csv` summarised (mean ± std, Welch t-test p-value vs E1).
- Fig. D: 4 held-out captions × 8 noise vectors for the best model (text control and diversity).
- Fig. E: caption interpolation strip.
- Significance: TODO. With n = 3 seeds, report effect sizes and state the low power.

### 3.3 Discussion
*Format asks:* are the hypotheses supported, strengths and weaknesses, and explanations rooted in the algorithm and the data.
- One paragraph per hypothesis: supported or not, and the number that shows it.
- Contrast with Project 1: v2's D does not saturate (D(G(z)) stays well above 0); point to Fig. A against v1's 94–100% rejection rate.
- Caveat: CLIP is both the conditioning encoder and the judge, so the CLIP score may flatter the model. Cross-check with another encoder if time allows.
- Weaknesses: 64×64 resolution, a single domain, possible mode collapse inside classes (memorization check, 02_results §6).

## 4. Related Work
*Format asks, for each work:* their problem and method, how ours differs, and why ours is better (or what we trade off).
| Work | Their problem / method | How ours differs |
|---|---|---|
| Goodfellow et al. 2014, GAN | Adversarial training, MLPs | — (foundation) |
| Radford et al. 2016, DCGAN | Convolutional GAN design rules | Our G/D are exactly this, plus text conditioning |
| Mirza & Osindero 2014, cGAN | Condition G and D on labels | Text embeddings instead of class labels |
| Reed et al. 2016, GAN-INT-CLS | Text-to-image with char-CNN-RNN text, matching-aware D | Frozen CLIP text encoder; ablation of each piece; zero-shot split |
| Zhang et al. 2017, StackGAN | Two-stage 256×256 text-to-image | We stay single-stage at 64×64 (compute); future work |
| Radford et al. 2021, CLIP | Joint image–text embedding | Used as a frozen encoder and as an evaluation metric |
| Heusel et al. 2017, FID | GAN evaluation metric | Our main realism metric |
| Project 1 (our v1) | EML/complex DCGAN on YFCC100M | Why v2 abandons it (one paragraph) |

## 5. Future Work
*Format asks:* each major shortcoming paired with a proposed fix.
- Low resolution → a StackGAN-style second stage, or progressive growing to 128×128.
- One domain → CUB-200 birds next (same caption format), then a filtered YFCC100M subset using the Project 1 Qwen captions.
- GAN instability → spectral norm or a hinge loss as **single controlled ablations** (Project 1 used them all at once).
- Test Project 1's DINOv3-feature discriminator idea as one ablation on top of E3.

## 6. Conclusion
*Format asks:* the important results and what they mean for future work.
- TODO: 3–4 sentences: what worked, which hypothesis held, and the lesson about building on proven pipelines.

## Bibliography
*Format asks:* standard, well-formatted, comprehensive.
- Keep the Project 1 citations that still apply (Radford 2016 DCGAN, Mirza & Osindero, Radford 2021 CLIP, Inkawhich DCGAN tutorial).
- Add: Reed et al. 2016 (ICML, *Generative Adversarial Text to Image Synthesis*); Reed et al. 2016 (CVPR, fine-grained visual descriptions: the captions); Nilsback & Zisserman 2008 (Flowers-102); Heusel et al. 2017 (FID); Goodfellow et al. 2014; Zhang et al. 2017 (StackGAN); Hessel et al. 2021 (CLIPScore).
- Drop the citations used only for the removed parts (EML / Odrzywołek, complex RMSNorm, DINOv3, YFCC100M), unless they appear in §1 or §4 when you explain v1.
