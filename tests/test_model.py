"""Shape and init tests for the networks (CPU, small batch)."""

import torch

from src import model

# ------------------------------------------------------------ Milestone 0 (should pass now)


def test_generator_output_shape_and_range():
    g = model.Generator()
    g.apply(model.weights_init)
    x = g(torch.randn(4, 100, 1, 1))
    assert x.shape == (4, 3, 64, 64)
    assert x.min() >= -1 and x.max() <= 1  # tanh


def test_discriminator_returns_one_logit_per_image():
    d = model.Discriminator()
    assert d(torch.randn(4, 3, 64, 64)).shape == (4,)


def test_weights_init_matches_dcgan_paper():
    g = model.Generator()
    g.apply(model.weights_init)
    w = g.stem[0].weight
    assert abs(w.mean().item()) < 0.005 and abs(w.std().item() - 0.02) < 0.005


# ------------------------------------------------------------ Milestone 1 (fail until Plan step 7)


def test_cond_generator_shape():
    g = model.CondGenerator()
    assert g(torch.randn(4, 100, 1, 1), torch.randn(4, 512)).shape == (4, 3, 64, 64)


def test_cond_discriminator_shape_and_text_matters():
    d = model.CondDiscriminator().eval()
    x = torch.randn(2, 3, 64, 64)
    a, b = d(x, torch.randn(2, 512)), d(x, torch.randn(2, 512))
    assert a.shape == (2,)
    assert not torch.allclose(a, b)  # the caption must change D's output
