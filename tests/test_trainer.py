"""Training-loop tests on random tensors with tiny networks (seconds on CPU)."""

import math

import torch
from torch.utils.data import DataLoader, TensorDataset

from src import model, trainer

# ------------------------------------------------------------ Milestone 0 (should pass now)


def _tiny_gan():
    g, d = model.Generator(ngf=8), model.Discriminator(ndf=8)
    g.apply(model.weights_init)
    d.apply(model.weights_init)
    return g, d


def test_dcgan_epoch_runs_and_updates_both_networks():
    torch.manual_seed(0)
    g, d = _tiny_gan()
    g0, d0 = g.stem[0].weight.clone(), d.head.weight.clone()
    loader = DataLoader(TensorDataset(torch.rand(16, 3, 64, 64) * 2 - 1, torch.zeros(16)), batch_size=8)
    opt_g = torch.optim.Adam(g.parameters(), 2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(d.parameters(), 2e-4, betas=(0.5, 0.999))
    stats = trainer.train_one_epoch(g, d, opt_g, opt_d, loader, "cpu", mode="dcgan")
    assert all(math.isfinite(v) for v in stats.values())
    assert {"d_loss", "g_loss", "d_real", "d_fake"} <= stats.keys()
    assert not torch.equal(g0, g.stem[0].weight) and not torch.equal(d0, d.head.weight)


def test_dcgan_d_loss_is_log2_for_an_undecided_discriminator():
    zero = torch.zeros(4)
    assert math.isclose(trainer.d_loss_dcgan(zero, zero).item(), math.log(2), rel_tol=1e-6)


# ------------------------------------------------------------ Milestone 1


def test_cond_epoch_runs_and_text_branch_learns():
    torch.manual_seed(0)
    g, d = model.CondGenerator(ngf=8), model.CondDiscriminator(ndf=8)
    g.apply(model.weights_init)
    d.apply(model.weights_init)
    tg0, td0 = g.text_proj.net[0].weight.clone(), d.text_proj.net[0].weight.clone()
    t = torch.nn.functional.normalize(torch.randn(16, 512), dim=-1)
    ds = TensorDataset(torch.rand(16, 3, 64, 64) * 2 - 1, t, t.roll(1, 0), torch.arange(16))
    opt_g = torch.optim.Adam(g.parameters(), 2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(d.parameters(), 2e-4, betas=(0.5, 0.999))
    stats = trainer.train_one_epoch(g, d, opt_g, opt_d, DataLoader(ds, batch_size=8), "cpu", mode="cond")
    assert all(math.isfinite(v) for v in stats.values())
    assert not torch.equal(tg0, g.text_proj.net[0].weight)  # gradients reach the text projection
    assert not torch.equal(td0, d.text_proj.net[0].weight)


# ------------------------------------------------------------ Milestone 2 (fail until Plan steps 8-9)


def test_gan_cls_loss_values():
    big = torch.full((4,), 20.0)
    # perfect D: real-match high, wrong & fake low -> loss ~ 0
    assert trainer.d_loss_gan_cls(big, -big, -big).item() < 1e-6
    # undecided D: log2 + 0.5*(log2 + log2) = 2*log2
    z = torch.zeros(4)
    assert math.isclose(trainer.d_loss_gan_cls(z, z, z).item(), 2 * math.log(2), rel_tol=1e-6)


def test_interpolated_text_is_unit_norm_and_between():
    t = torch.nn.functional.normalize(torch.randn(4, 512), dim=-1)
    ti = trainer.interpolate_text(t, 0.5)
    assert torch.allclose(ti.norm(dim=-1), torch.ones(4), atol=1e-5)
    assert (ti * t).sum(-1).min() > 0  # still points toward its own caption
