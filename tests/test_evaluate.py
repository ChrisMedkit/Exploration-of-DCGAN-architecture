"""Evaluation helper tests."""

import pandas as pd
import torch

from src import evaluate


def test_save_grid_writes_png(tmp_path):
    p = evaluate.save_grid(torch.rand(16, 3, 64, 64) * 2 - 1, tmp_path / "g.png", nrow=4)
    assert p.exists() and p.stat().st_size > 0


def test_plot_losses_from_log(tmp_path):
    log = tmp_path / "run" / "log.csv"
    log.parent.mkdir()
    pd.DataFrame(dict(epoch=[1, 2], d_loss=[0.7, 0.6], g_loss=[1.0, 1.2],
                      d_real=[0.6, 0.6], d_fake=[0.4, 0.3])).to_csv(log, index=False)
    assert evaluate.plot_losses(log, tmp_path / "loss.png").exists()


# ------------------------------------------------------------ Milestone 3 (fail until Plan step 10)


def test_fid_is_near_zero_for_identical_sets():
    x = torch.rand(64, 3, 64, 64) * 2 - 1
    assert evaluate.compute_fid(x, x, device="cpu") < 1.0
