#!/usr/bin/env python3
"""
Plot training / validation curves straight from the SLURM logs.

Reads the same "Epoch NNN | Train <loss> ... | Val <loss> ..." lines that
parse_training_curves.py reads, so it needs nothing added to the training scripts and
can be run WHILE a job is still training -- it just sees fewer epochs.

Examples
--------
  # every tinyUnet run, one figure
  python plot_training_curves.py --match tinyUnet

  # three specific runs, compared on shape rather than absolute value
  python plot_training_curves.py --match "tinyUnet_version_(8|9|10)" --normalize

  # one run in detail: train vs val, LR, and the overfit gap shaded
  python plot_training_curves.py --match tinyUnet_version_10 --detail

  # watch a job that is still running
  python plot_training_curves.py --match tinyUnet_version_10 --detail --out live.png

NOTE ON COMPARING RUNS: loss VALUES are not comparable across versions. v2-v8 trained
on raw Mg/ha, v9-v14 on log1p, and v15/v6/v7/v8+ on sqrt, with the Huber delta rescaled
1.0 -> 5.0 at the sqrt switch. Only the SHAPE of a curve and the train-vs-val gap within
one run mean anything across versions -- use --normalize for an honest comparison.
"""

import re
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_DIR = Path(__file__).resolve().parent / "out_and_err"
OUT_DIR = Path(__file__).resolve().parent / "curves"

EPOCH_RE = re.compile(
    r"^Epoch\s+(\d+)\s*\|\s*"
    r"Train\s+(MSE|Huber)\s+([0-9.eE+-]+)\s+MAE\s+([0-9.eE+-]+)\s*\|\s*"
    r"Val\s+(?:MSE|Huber)\s+([0-9.eE+-]+)\s+MAE\s+([0-9.eE+-]+)"
    r"(?:\s*\|\s*LR\s+([0-9.eE+-]+))?"
)

PALETTE = ["#2e8b85", "#c9853a", "#8b6fc4", "#c4534a",
           "#4f8f48", "#3d7fb5", "#b25d92", "#93801f"]


def parse_log(path):
    rows = {}
    loss_name = None
    for line in path.read_text(errors="replace").splitlines():
        m = EPOCH_RE.match(line)
        if not m:
            continue
        loss_name = m.group(2)
        # dict keyed by epoch: a resumed run repeats epoch numbers, keep the last
        rows[int(m.group(1))] = dict(
            epoch=int(m.group(1)),
            train_loss=float(m.group(3)), train_mae=float(m.group(4)),
            val_loss=float(m.group(5)),  val_mae=float(m.group(6)),
            lr=float(m.group(7)) if m.group(7) else None,
        )
    if not rows:
        return None
    eps = [rows[k] for k in sorted(rows)]
    val = [e["val_loss"] for e in eps]
    bi = min(range(len(val)), key=lambda i: val[i])
    return dict(run=path.stem, loss_name=loss_name, epochs=eps,
                best_epoch=eps[bi]["epoch"], best_val=val[bi], final_val=val[-1])


def collect(log_dir, match, min_epochs):
    pat = re.compile(match) if match else None
    out = []
    for p in sorted(Path(log_dir).glob("*.out")):
        if pat and not pat.search(p.stem):
            continue
        r = parse_log(p)
        if r and len(r["epochs"]) >= min_epochs:
            out.append(r)
    return out


def plot_compare(runs, metric, normalize, logy, out_path):
    key_t = "train_loss" if metric == "loss" else "train_mae"
    key_v = "val_loss" if metric == "loss" else "val_mae"
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for i, r in enumerate(runs):
        c = PALETTE[i % len(PALETTE)]
        x = [e["epoch"] for e in r["epochs"]]
        tv = [e[key_t] for e in r["epochs"]]
        vv = [e[key_v] for e in r["epochs"]]
        if normalize:                       # divide by each run's own epoch-1 value,
            t0, v0 = tv[0] or 1, vv[0] or 1  # the only honest cross-version comparison
            tv = [v / t0 for v in tv]
            vv = [v / v0 for v in vv]
        lbl = r["run"].replace("_train", "").replace("tinyUnet_", "tinyUnet ")
        ax.plot(x, vv, color=c, lw=2.0, label=f"{lbl}  (best@{r['best_epoch']})")
        ax.plot(x, tv, color=c, lw=1.2, ls="--", alpha=0.55)
        bi = [e["epoch"] for e in r["epochs"]].index(r["best_epoch"])
        ax.plot(r["best_epoch"], vv[bi], "o", ms=7, mfc="white", mec="#b3382f", mew=2, zorder=5)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(("normalised " if normalize else "") +
                  ("loss" if metric == "loss" else "MAE") + " (÷ epoch 1)" * normalize)
    ax.set_title("Validation (solid) vs train (dashed) — red ring = best epoch",
                 fontsize=11)
    ax.grid(alpha=0.25, lw=0.7)
    ax.legend(fontsize=8.5, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"wrote {out_path}")


def plot_detail(r, out_path):
    eps = r["epochs"]
    x = [e["epoch"] for e in eps]
    tv = [e["train_loss"] for e in eps]
    vv = [e["val_loss"] for e in eps]
    has_lr = eps[-1]["lr"] is not None
    fig, axes = plt.subplots(3 if has_lr else 2, 1, figsize=(10, 9 if has_lr else 6.5),
                             sharex=True, gridspec_kw=dict(height_ratios=[3, 2, 1][:3 if has_lr else 2]))
    ax = axes[0]
    ax.plot(x, tv, color="#c9853a", lw=1.4, ls="--", label=f"train {r['loss_name']}")
    ax.plot(x, vv, color="#2e8b85", lw=2.1, label=f"val {r['loss_name']}")
    ax.plot(r["best_epoch"], r["best_val"], "o", ms=8, mfc="white", mec="#b3382f", mew=2)
    # shade from the best epoch to the end: everything here is overfitting
    if r["best_epoch"] < x[-1]:
        ax.axvspan(r["best_epoch"], x[-1], color="#b3382f", alpha=0.07)
        ax.annotate(f"best @ {r['best_epoch']}\nval {r['best_val']:.4f}\n"
                    f"final {r['final_val']:.4f}  ({r['final_val']-r['best_val']:+.4f})",
                    xy=(r["best_epoch"], r["best_val"]),
                    xytext=(0.62, 0.72), textcoords="axes fraction", fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="#b3382f", lw=1.2))
    ax.set_ylabel(r["loss_name"]); ax.grid(alpha=0.25, lw=0.7); ax.legend(fontsize=9)
    ax.set_title(r["run"], fontsize=11, fontweight="bold")

    ax = axes[1]
    ax.plot(x, [e["train_mae"] for e in eps], color="#c9853a", lw=1.4, ls="--", label="train MAE")
    ax.plot(x, [e["val_mae"] for e in eps], color="#2e8b85", lw=2.0, label="val MAE")
    ax.set_ylabel("MAE"); ax.grid(alpha=0.25, lw=0.7); ax.legend(fontsize=9)

    if has_lr:
        ax = axes[2]
        ax.plot(x, [e["lr"] for e in eps], color="#8b6fc4", lw=1.6)
        ax.set_yscale("log"); ax.set_ylabel("LR"); ax.grid(alpha=0.25, lw=0.7)
    axes[-1].set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match", default="", help="regex against the .out filename")
    ap.add_argument("--log-dir", default=str(LOG_DIR))
    ap.add_argument("--out", default="", help="output PNG (default: curves/<name>.png)")
    ap.add_argument("--metric", choices=["loss", "mae"], default="loss")
    ap.add_argument("--normalize", action="store_true",
                    help="divide each curve by its own epoch-1 value; use this whenever "
                         "comparing runs that trained on different target spaces")
    ap.add_argument("--log", action="store_true", help="log y axis")
    ap.add_argument("--detail", action="store_true",
                    help="one run only: loss, MAE and LR stacked, overfit region shaded")
    ap.add_argument("--min-epochs", type=int, default=2)
    a = ap.parse_args()

    runs = collect(a.log_dir, a.match, a.min_epochs)
    if not runs:
        raise SystemExit(f"no runs matched {a.match!r} in {a.log_dir}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{'run':46s} {'ep':>4} {'best@':>6} {'best val':>10} {'final':>10} {'overfit':>9}")
    print("-" * 90)
    for r in runs:
        print(f"{r['run'][:46]:46s} {len(r['epochs']):4d} {r['best_epoch']:6d} "
              f"{r['best_val']:10.4f} {r['final_val']:10.4f} "
              f"{r['final_val']-r['best_val']:+9.4f}")
    print()

    if a.detail:
        # newest run wins when the pattern matches several
        r = max(runs, key=lambda r: len(r["epochs"]))
        out = Path(a.out) if a.out else OUT_DIR / f"{r['run']}_detail.png"
        plot_detail(r, out)
    else:
        name = (a.match or "all").replace("|", "-").replace("(", "").replace(")", "")
        out = Path(a.out) if a.out else OUT_DIR / f"curves_{name}.png"
        plot_compare(runs, a.metric, a.normalize, a.log, out)


if __name__ == "__main__":
    main()
