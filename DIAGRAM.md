# Text-to-Image DCGAN (rebuild) — Diagrams
See [OUTLINE.md](OUTLINE.md)

## 1. Plan
```mermaid
flowchart TD
  subgraph M0["Milestone 0: proven baseline (implemented)"]
    S1["Step 1: pytest -q<br>tests/"] --> S2["Step 2: train.py --epochs 5<br>data.py, trainer.py"]
    S2 --> S3["Step 3: full E0 run, check flowers<br>02_results.ipynb"]
    S3 --> S4["Step 4: dataset facts<br>01_exploration.ipynb"]
  end
  subgraph M1["Milestone 1: text conditioning"]
    S5["Step 5: captions, split, CLIP cache<br>data.py, text_encoder.py"] --> S6["Step 6: FlowersCaptions<br>data.py"]
    S6 --> S7["Step 7: CondG, CondD, cond loop<br>model.py, trainer.py"]
  end
  subgraph M2["Milestone 2: GAN-CLS"]
    S8["Step 8: matching-aware D loss<br>trainer.py"] --> S9["Step 9: GAN-INT<br>trainer.py"]
  end
  subgraph M3["Milestone 3-4: evaluate, demo, report"]
    S10["Step 10: FID, CLIP score<br>evaluate.py, evaluate_all.py"] --> S11["Step 11: E0-E3 x 3 seeds, stats"]
    S11 --> S12["Step 12: generate.py, app.py"] --> S13["Step 13: write report"]
  end
  S3 -- "only if E0 shows flowers" --> S5
  S7 --> S8
  S9 --> S10
```
The build order. Each milestone adds one piece on top of a verified one, and E0 must work before anything else.

## 2. Why v1 failed vs what v2 does
```mermaid
flowchart LR
  subgraph V1["v1 (Project 1)"]
    A1["YFCC100M, uncurated, 256-512 px"] --> B1["Custom EML complex G<br>ColorFocus, FiLM"]
    B1 --> C1["Frozen DINOv3 D<br>hinge + spectral norm"]
    C1 --> D1["D wins from epoch 1<br>textures, no images"]
  end
  subgraph V2["v2 (this rebuild)"]
    A2["Oxford-102 Flowers, 64 px<br>human captions"] --> B2["L14 DCGAN G and D<br>unchanged"]
    B2 --> C2["+ one change per experiment<br>E0, E1, E2, E3"]
    C2 --> D2["FID + CLIP score<br>3 seeds"]
  end
```
The structural difference: v1 changed everything at once on hard data; v2 starts from the class pipeline on easy data and changes one variable at a time.

## 3. Code structure
```mermaid
classDiagram
  class Generator {
    +int latent_dim
    +stem: ConvT 1-4
    +body: 3 x up_block
    +head: ConvT + Tanh
    +forward(z) Tensor
  }
  class Discriminator {
    +features: 4 x down_block
    +head: Conv 4x4
    +forward(x) Tensor
  }
  class TextProjection {
    +forward(t) Tensor
  }
  class CondGenerator {
    +text_proj: TextProjection
    +forward(z, t) Tensor
  }
  class CondDiscriminator {
    +text_proj: TextProjection
    +joint: Conv1x1 BN LReLU
    +forward(x, t) Tensor
  }
  class FlowersImages {
    +getitem(i) tuple
  }
  class FlowersCaptions {
    +by_class: dict
    +getitem(i) tuple
  }
  Generator <|-- CondGenerator
  Discriminator <|-- CondDiscriminator
  CondGenerator *-- TextProjection
  CondDiscriminator *-- TextProjection
```
The conditional networks inherit the tested DCGAN blocks and add only the text projection. Datasets live in `src/data.py`.

```mermaid
flowchart LR
  T["train.py main"] --> TR["trainer.train"]
  TR --> DL["data.get_dataloader"]
  DL --> IC["data.load_image_cache / build_image_cache"]
  DL -.-> TC["text_encoder.load_text_cache"]
  TR --> BM["model.build_models"]
  TR --> E1["trainer.train_one_epoch"]
  E1 --> L["d_loss_dcgan / d_loss_gan_cls / g_loss / interpolate_text"]
  TR --> SG["evaluate.save_grid"]
  T --> PL["evaluate.plot_losses"]
  EA["evaluate_all.py"] -.-> F["evaluate.compute_fid / clip_score / summarize_seeds"]
  AP["app.py"] -.-> G["generate.load_generator / generate"]
```
Files → functions with "calls" arrows. Dashed arrows are milestone 1+ TODO paths.

## 4. ML pipeline
```mermaid
flowchart LR
  R["Flowers102 jpg<br>torchvision download"] --> C["64x64 uint8 cache<br>data.build_image_cache"]
  CAP["Reed text_c10 captions<br>data.load_captions"] --> EMB["CLIP ViT-B/32 cache<br>text_encoder.build_text_cache"]
  C --> SP["class split 82 / 20<br>data.class_split"]
  EMB --> SP
  SP --> DS["FlowersImages / FlowersCaptions<br>data.get_dataloader"]
  DS --> TRN["train E0-E3 x 3 seeds<br>trainer.train"]
  TRN --> CK["models/run/last.pt<br>runs/run/log.csv"]
  CK --> EV["FID, CLIP score<br>evaluate_all.py"]
  EV --> FIG["reports/figures<br>02_results.ipynb"]
  CK --> APP["app.py demo"]
```
From raw files to report figures and the demo, with the function responsible for each stage.

## 5. Model architecture: generator
```mermaid
flowchart TD
  Z["z: 100 x 1 x 1"] --> CAT["concat (E1+ only)<br>228 x 1 x 1"]
  T["CLIP text t: 512"] --> P["Linear 512-128 + BN + LeakyReLU<br>(E1+ only)"]
  P --> CAT
  CAT --> G1["ConvT 4x4 s1 + BN + ReLU<br>512 x 4 x 4"]
  G1 --> G2["ConvT 4x4 s2 + BN + ReLU<br>256 x 8 x 8"]
  G2 --> G3["ConvT 4x4 s2 + BN + ReLU<br>128 x 16 x 16"]
  G3 --> G4["ConvT 4x4 s2 + BN + ReLU<br>64 x 32 x 32"]
  G4 --> G5["ConvT 4x4 s2 + Tanh<br>3 x 64 x 64"]
```
DCGAN generator (3.58M parameters for E0). E0 feeds z straight into the first ConvT (100 channels); E1+ concatenates the projected text first.

## 6. Model architecture: discriminator
```mermaid
flowchart TD
  X["image: 3 x 64 x 64"] --> D1["Conv 4x4 s2 + LeakyReLU 0.2<br>64 x 32 x 32"]
  D1 --> D2["Conv 4x4 s2 + BN + LeakyReLU<br>128 x 16 x 16"]
  D2 --> D3["Conv 4x4 s2 + BN + LeakyReLU<br>256 x 8 x 8"]
  D3 --> D4["Conv 4x4 s2 + BN + LeakyReLU<br>512 x 4 x 4"]
  T["CLIP text t: 512"] --> P["Linear 512-128 + BN + LeakyReLU<br>replicate to 128 x 4 x 4 (E1+)"]
  D4 --> J["concat 640 x 4 x 4<br>Conv 1x1 + BN + LeakyReLU to 512 (E1+)"]
  P --> J
  J --> H["Conv 4x4 s1<br>logit, 1 per image"]
```
DCGAN discriminator (2.77M parameters for E0). E0 goes from the 4×4×512 features straight to the head. E1+ joins the text at 4×4 (Reed et al. Fig. 2).

## 7. Training step (GAN-CLS, Reed et al. Algorithm 1)
```mermaid
flowchart TD
  B["batch: x, t, t_wrong"] --> F["fake = G(z, t)"]
  F --> DS["D step"]
  DS --> R1["D(x, t) to 1"]
  DS --> R2["D(x, t_wrong) to 0<br>(E2, E3 only)"]
  DS --> R3["D(fake.detach(), t) to 0"]
  R1 --> UD["update D"]
  R2 --> UD
  R3 --> UD
  UD --> GS["G step: D(fake, t) to 1"]
  GS --> INT{"interpolate? (E3)"}
  INT -- yes --> GI["+ D(G(z, t_int), t_int) to 1"]
  INT -- no --> UG["update G"]
  GI --> UG
```
One batch of `trainer.train_one_epoch`. E0 is the same loop without t, and E1 drops the wrong-caption term.

## 8. Requirements → code
```mermaid
flowchart LR
  R0["M0: DCGAN makes flowers"] --> C0["model.Generator, Discriminator<br>trainer mode dcgan"] --> T0["test_model, test_trainer (pass)"]
  R1["M1: text conditioning"] --> C1["text_encoder, FlowersCaptions<br>CondGenerator, CondDiscriminator"] --> T1["test_cond_*, test_load_captions"]
  R2["M2: GAN-CLS + INT"] --> C2["d_loss_gan_cls, interpolate_text"] --> T2["test_gan_cls_loss_values"]
  R3["M3: FID, CLIP score, stats"] --> C3["evaluate.compute_fid, clip_score<br>evaluate_all.py"] --> T3["test_fid_near_zero"]
  R4["M4: front end"] --> C4["generate.py, app.py"]
  R5["Report sections 1-6"] --> C5["report/REPORT_OUTLINE.md<br>02_results.ipynb figures"]
```
Each requirement linked to the code that meets it and the test that checks it.
