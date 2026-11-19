"""Train one experiment end to end.

    python train.py                                  # E0: unconditional DCGAN, seed 42, 100 epochs
    python train.py --epochs 5                       # quick check that flowers start to appear
    python train.py --experiment E2_gan_cls --seed 43

Long runs on the Mac: `caffeinate -i python train.py ...` and keep it plugged in. Re-running the
same command resumes from models/<run>/last.pt.
"""

import argparse

from src import config, evaluate, trainer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", default="E0_dcgan", choices=list(config.EXPERIMENTS))
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    p.add_argument("--run", default=None, help="run name (default: <experiment>_s<seed>)")
    p.add_argument("--no-resume", action="store_true")
    a = p.parse_args()

    log = trainer.train(a.experiment, a.seed, a.epochs, a.run, resume=not a.no_resume)
    print("Loss plot:", evaluate.plot_losses(log))
    # Next (Plan step 10): once runs finish, `python evaluate_all.py` scores them (FID, CLIP score).


if __name__ == "__main__":
    main()
