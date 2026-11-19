"""GAN training loop, following class `code/L14/helper_train.py::train_gan_v1`.

One D step then one G step per batch, BCE-with-logits, non-saturating G loss (G wants D to say
"real"). That loop is the tested part; GAN-CLS only changes WHICH pairs D sees (Reed et al. Alg. 1).

    mode "dcgan"   (E0, IMPLEMENTED)  D: real x -> 1, G(z) -> 0
    mode "cond"    (E1, TODO)         D: (x, t) -> 1, (G(z,t), t) -> 0
    mode "gan_cls" (E2/E3, TODO)      D: (x, t) -> 1, (x, t_wrong) -> 0, (G(z,t), t) -> 0
    interpolate    (E3, TODO)         G also trained on t_int = beta*t + (1-beta)*t_other
"""

import csv
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src import config, evaluate, model
from src.data import get_dataloader
from src.utils import load_checkpoint, save_checkpoint, set_all_seeds, set_deterministic


# ============================================================ losses
def bce(logits: torch.Tensor, target: float) -> torch.Tensor:
    """BCE-with-logits against a constant label (1.0 real, 0.0 fake)."""
    return F.binary_cross_entropy_with_logits(logits, torch.full_like(logits, target))


def d_loss_dcgan(s_real: torch.Tensor, s_fake: torch.Tensor) -> torch.Tensor:
    """Class helper_train: 0.5 * (BCE(D(x), 1) + BCE(D(G(z)), 0)). Also used for mode "cond"."""
    return 0.5 * (bce(s_real, 1.0) + bce(s_fake, 0.0))


def d_loss_gan_cls(s_real: torch.Tensor, s_wrong: torch.Tensor, s_fake: torch.Tensor) -> torch.Tensor:
    """Matching-aware D loss (Reed et al. 2016, Algorithm 1, line 9).

    L_D = log(s_r) + (log(1 - s_w) + log(1 - s_f)) / 2, written as BCE terms:
        BCE(s_real, 1) + 0.5 * (BCE(s_wrong, 0) + BCE(s_fake, 0))

    TODO (Plan step 8): return the expression above. Unit test: tests/test_trainer.py.
    """
    raise NotImplementedError


def g_loss(s_fake: torch.Tensor) -> torch.Tensor:
    """Non-saturating generator loss: BCE(D(G(z)), 1)."""
    return bce(s_fake, 1.0)


def interpolate_text(t: torch.Tensor, beta: float = config.INTERP_BETA) -> torch.Tensor:
    """GAN-INT (Reed et al. Sec. 4.3): mix each caption embedding with another one in the batch.

    TODO (Plan step 9): t_other = t.roll(1, dims=0); return F.normalize(beta*t + (1-beta)*t_other, dim=-1)
    There is no real image for t_int, so it is used ONLY in the generator step (D sees G(z, t_int)).
    """
    raise NotImplementedError


# ============================================================ one epoch
def _accuracy(logits: torch.Tensor, target: float) -> float:
    pred = (logits.detach() > 0).float()
    return (pred == target).float().mean().item()


def train_one_epoch(G, D, opt_g, opt_d, loader, device, mode: str = "dcgan",
                    interpolate: bool = False) -> dict:
    """Train for one pass over `loader`; return the per-epoch MEAN of each logged metric.

    Logged (report §3.2): d_loss, g_loss, d_real (mean sigmoid D on real), d_fake (on G(z)),
    acc_real, acc_fake, and for gan_cls also d_wrong / acc_wrong.
    """
    G.train()
    D.train()
    totals: dict[str, float] = {}
    n = 0
    for batch in loader:
        if mode == "dcgan":
            x = batch[0].to(device)
            b = x.size(0)
            z = torch.randn(b, G.latent_dim, 1, 1, device=device)
            fake = G(z)

            # ---- D step: real -> 1, fake -> 0
            opt_d.zero_grad(set_to_none=True)
            s_real = D(x)
            s_fake = D(fake.detach())
            loss_d = d_loss_dcgan(s_real, s_fake)
            loss_d.backward()
            opt_d.step()

            # ---- G step: make D call the fakes real
            opt_g.zero_grad(set_to_none=True)
            s_fake_g = D(fake)
            loss_g = g_loss(s_fake_g)
            loss_g.backward()
            opt_g.step()

            stats = dict(d_loss=loss_d.item(), g_loss=loss_g.item(),
                         d_real=torch.sigmoid(s_real).mean().item(),
                         d_fake=torch.sigmoid(s_fake).mean().item(),
                         acc_real=_accuracy(s_real, 1.0), acc_fake=_accuracy(s_fake, 0.0))

        elif mode in ("cond", "gan_cls"):
            # TODO (Plan steps 7-9), same structure as the dcgan branch:
            #   x, t, t_wrong, _ = (v.to(device) for v in batch); z = randn(b, G.latent_dim, 1, 1)
            #   fake = G(z, t)
            #   D step:
            #     s_real = D(x, t); s_fake = D(fake.detach(), t)
            #     cond:    loss_d = d_loss_dcgan(s_real, s_fake)
            #     gan_cls: s_wrong = D(x, t_wrong); loss_d = d_loss_gan_cls(s_real, s_wrong, s_fake)
            #   G step:
            #     loss_g = g_loss(D(fake, t))
            #     if interpolate: t_int = interpolate_text(t); loss_g = loss_g + g_loss(D(G(z, t_int), t_int))
            #   stats: as above, plus d_wrong / acc_wrong for gan_cls
            raise NotImplementedError(f"mode {mode!r} comes in milestone 1+ (Plan steps 7-9)")
        else:
            raise ValueError(f"unknown mode {mode!r}")

        for k, v in stats.items():
            totals[k] = totals.get(k, 0.0) + v * b
        n += b
    return {k: v / max(n, 1) for k, v in totals.items()}


# ============================================================ full run
def train(experiment: str = "E0_dcgan", seed: int = config.RANDOM_SEED, epochs: int = config.NUM_EPOCHS,
          run_name: str | None = None, resume: bool = True, device=config.DEVICE) -> Path:
    """Train one (experiment, seed) run. Returns the run's CSV log path.

    Every epoch: append metrics to runs/<run>/log.csv, save a fixed-noise 8x8 grid to
    reports/figures/samples/<run>/epoch_XXX.png and overwrite models/<run>/last.pt (resumable).
    """
    exp = config.EXPERIMENTS[experiment]
    run = run_name or f"{experiment}_s{seed}"
    log_path = config.RUNS / run / "log.csv"
    ckpt_path = config.MODELS / run / "last.pt"
    sample_dir = config.FIGURES / "samples" / run

    set_all_seeds(seed)
    set_deterministic()
    G, D = model.build_models(exp["conditional"], device)
    opt_g = torch.optim.Adam(G.parameters(), lr=config.LR_G, betas=config.BETAS)
    opt_d = torch.optim.Adam(D.parameters(), lr=config.LR_D, betas=config.BETAS)
    loader = get_dataloader(exp["conditional"])

    # Fixed noise (made on CPU so it is identical on mps/cuda/cpu) to compare epochs fairly.
    fixed_z = torch.randn(64, G.latent_dim, 1, 1, generator=torch.Generator().manual_seed(seed))
    fixed_t = None  # TODO (Plan step 7): 8 held-out TEST captions x 8 noise vectors for conditional runs

    start = 0
    if resume and ckpt_path.exists():
        ck = load_checkpoint(ckpt_path, device)
        G.load_state_dict(ck["G"]); D.load_state_dict(ck["D"])
        opt_g.load_state_dict(ck["opt_g"]); opt_d.load_state_dict(ck["opt_d"])
        start, fixed_z = ck["epoch"], ck["fixed_z"]
        print(f"Resuming {run} after epoch {start}")

    print(f"{run}: {exp['note']}\n  device={device}  G={model.count_parameters(G):,}  "
          f"D={model.count_parameters(D):,} params  batches/epoch={len(loader)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(start + 1, epochs + 1):
        t0 = time.time()
        stats = train_one_epoch(G, D, opt_g, opt_d, loader, device, exp["mode"], exp["interpolate"])
        stats.update(epoch=epoch, seconds=round(time.time() - t0, 1))
        _append_csv(log_path, stats)
        print(f"epoch {epoch:03d}/{epochs} | D {stats['d_loss']:.3f}  G {stats['g_loss']:.3f} | "
              f"D(x) {stats['d_real']:.2f}  D(G(z)) {stats['d_fake']:.2f} | {stats['seconds']}s")

        if epoch % config.SAMPLE_EVERY == 0 or epoch == epochs:
            G.eval()
            with torch.no_grad():
                imgs = G(fixed_z.to(device)) if fixed_t is None else G(fixed_z.to(device), fixed_t.to(device))
            evaluate.save_grid(imgs, sample_dir / f"epoch_{epoch:03d}.png")
        save_checkpoint(ckpt_path, G=G.state_dict(), D=D.state_dict(), opt_g=opt_g.state_dict(),
                        opt_d=opt_d.state_dict(), epoch=epoch, fixed_z=fixed_z, experiment=experiment,
                        seed=seed, config={k: v for k, v in exp.items()})
    return log_path


def _append_csv(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
