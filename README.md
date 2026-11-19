# Text-to-Image DCGAN (v2), MSCS 435/635 AML

A rebuild of Project 1 (`../AML-DCGAN-main 2`). The goal is the same: caption in, image out. This time it is built on proven pipelines: the class L14 DCGAN, then a conditional GAN, then GAN-CLS (Reed et al., 2016), trained on Oxford-102 Flowers at 64×64 on Apple silicon (MPS).

```bash
~/.venvs/aml-dcgan-v2/bin/python -m pytest -q        # milestone-0 tests pass; later milestones are TODO
~/.venvs/aml-dcgan-v2/bin/python train.py --epochs 5 # downloads Flowers-102 (~330 MB), trains E0 briefly
caffeinate -i ~/.venvs/aml-dcgan-v2/bin/python train.py   # full E0 baseline (~30 min)
```

Start with [OUTLINE.md](OUTLINE.md) for the plan, [DIAGRAM.md](DIAGRAM.md) for the diagrams, and [report/REPORT_OUTLINE.md](report/REPORT_OUTLINE.md) for the write-up.
