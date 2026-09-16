# Parses per-epoch training curves straight out of the SLURM .out logs.
#
# Nothing new has to be added to the training scripts: they already print one line per
# epoch. This just reads them back. Safe to run WHILE a job is training -- it simply
# sees fewer epochs -- so it doubles as a live progress check.
#
# Two log formats exist across the project's history:
#   old (v2-v9)   Epoch 001 | Train MSE 3732.0653 MAE 36.8649 | Val MSE 3357.3695 MAE 35.4483
#   new (v10+)    Epoch 001 | Train Huber 5.8436 MAE 2.7061 | Val Huber 4.9977 MAE 2.4416 | LR 1.00e-04
#
# NOTE ON COMPARING CURVES ACROSS VERSIONS: the loss VALUES are not comparable.
# v2-v8 trained on raw Mg/ha, v9-v14 on log1p, v15/v6/v7/v8 on sqrt, and the Huber
# delta was rescaled 1.0 -> 5.0 at the sqrt switch. Only the SHAPE of a curve, and the
# train-vs-val gap within one run, carry meaning across versions.

import re
import json
import argparse
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent / "out_and_err"
OUT_DIR = Path(__file__).resolve().parent / "curves"

EPOCH_RE = re.compile(
    r"^Epoch\s+(\d+)\s*\|\s*"
    r"Train\s+(MSE|Huber)\s+([0-9.eE+-]+)\s+MAE\s+([0-9.eE+-]+)\s*\|\s*"
    r"Val\s+(?:MSE|Huber)\s+([0-9.eE+-]+)\s+MAE\s+([0-9.eE+-]+)"
    r"(?:\s*\|\s*LR\s+([0-9.eE+-]+))?"
)
BEST_RE = re.compile(r"^Best epoch:\s*(\d+)\s*\(Val\s+\w+\s+([0-9.eE+-]+)\)")


def parse_log(path: Path) -> dict | None:
    epochs = []
    loss_name = None
    best_epoch = None
    for line in path.read_text(errors="replace").splitlines():
        m = EPOCH_RE.match(line)
        if m:
            loss_name = m.group(2)
            epochs.append({
                "epoch": int(m.group(1)),
                "train_loss": float(m.group(3)),
                "train_mae": float(m.group(4)),
                "val_loss": float(m.group(5)),
                "val_mae": float(m.group(6)),
                "lr": float(m.group(7)) if m.group(7) else None,
            })
            continue
        b = BEST_RE.match(line)
        if b:
            best_epoch = int(b.group(1))
    if not epochs:
        return None

    # The script may have been resumed; keep the last value seen per epoch number.
    dedup = {e["epoch"]: e for e in epochs}
    epochs = [dedup[k] for k in sorted(dedup)]

    val = [e["val_loss"] for e in epochs]
    argmin = min(range(len(val)), key=lambda i: val[i])
    return {
        "run": path.stem,
        "job_id": path.stem.split("_")[-1],
        "loss_name": loss_name,
        "n_epochs": len(epochs),
        "epochs": epochs,
        # best_epoch from the log if the script printed it, else recomputed
        "best_epoch": best_epoch if best_epoch is not None else epochs[argmin]["epoch"],
        "best_val_loss": val[argmin],
        "final_val_loss": val[-1],
        # how much worse the last epoch is than the best one: the overfit gap
        "overfit_gap": val[-1] - val[argmin],
        "final_train_val_gap": epochs[-1]["val_loss"] - epochs[-1]["train_loss"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-epochs", type=int, default=5,
                    help="ignore crashed runs with fewer epochs than this")
    ap.add_argument("--log-dir", default=str(LOG_DIR))
    args = ap.parse_args()

    runs = []
    for p in sorted(Path(args.log_dir).glob("*.out")):
        r = parse_log(p)
        if r and r["n_epochs"] >= args.min_epochs:
            runs.append(r)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "curves.json").write_text(json.dumps(runs, indent=1))

    with (OUT_DIR / "curves.csv").open("w") as f:
        f.write("run,epoch,train_loss,train_mae,val_loss,val_mae,lr\n")
        for r in runs:
            for e in r["epochs"]:
                f.write(f"{r['run']},{e['epoch']},{e['train_loss']},{e['train_mae']},"
                        f"{e['val_loss']},{e['val_mae']},{e['lr'] if e['lr'] else ''}\n")

    print(f"parsed {len(runs)} runs -> {OUT_DIR}/curves.json + curves.csv\n")
    hdr = f"{'run':42s} {'loss':6s} {'ep':>4} {'best@':>6} {'best val':>9} {'final val':>9} {'overfit':>8}"
    print(hdr); print("-" * len(hdr))
    for r in sorted(runs, key=lambda r: r["run"]):
        print(f"{r['run'][:42]:42s} {r['loss_name']:6s} {r['n_epochs']:4d} "
              f"{r['best_epoch']:6d} {r['best_val_loss']:9.4f} {r['final_val_loss']:9.4f} "
              f"{r['overfit_gap']:+8.4f}")


if __name__ == "__main__":
    main()
