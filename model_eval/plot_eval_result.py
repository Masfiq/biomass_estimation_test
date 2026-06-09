import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def save_bar(labels, values, title, ylabel, out_path, rotate=False):
    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)

    if rotate:
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_horizontal_bar(labels, values, title, xlabel, out_path):
    plt.figure(figsize=(10, max(5, 0.35 * len(labels))))
    y_pos = np.arange(len(labels))
    plt.barh(y_pos, values)
    plt.yticks(y_pos, labels)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_heatmap(matrix, title, out_path, xticklabels=None, yticklabels=None):
    plt.figure(figsize=(7, 5))
    plt.imshow(matrix, aspect="auto")
    plt.colorbar(label="Value")
    plt.title(title)

    if xticklabels is not None:
        plt.xticks(np.arange(len(xticklabels)), xticklabels, rotation=45, ha="right")

    if yticklabels is not None:
        plt.yticks(np.arange(len(yticklabels)), yticklabels)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, help="Path to eval summary JSON")
    parser.add_argument("--top-n", type=int, default=15)
    args = parser.parse_args()

    summary_path = Path(args.summary)
    out_dir = summary_path.with_name(summary_path.stem + "_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)


    # 1. Overall metrics
 
    metrics = summary.get("metrics", {})
    if metrics:
        labels = list(metrics.keys())
        values = [float(metrics[k]) for k in labels]

        save_bar(
            labels,
            values,
            "Overall Validation Metrics",
            "Value",
            out_dir / "overall_metrics.png",
            rotate=False,
        )


    # 2. Channel attention
  
    channel_attention = summary.get("channel_attention_mean", None)
    if channel_attention:
        bands = list(channel_attention.keys())
        weights = [float(channel_attention[b]) for b in bands]

        save_bar(
            bands,
            weights,
            "Mean Channel Attention by Band",
            "Mean attention weight",
            out_dir / "channel_attention_by_band.png",
            rotate=True,
        )

  
    # 3. Spatial attention 3x3

    spatial = summary.get("spatial_attention_mean_3x3", None)
    if spatial:
        spatial_matrix = np.array(spatial, dtype=float)

        save_heatmap(
            spatial_matrix,
            "Mean Spatial Attention Map",
            out_dir / "spatial_attention_3x3.png",
            xticklabels=["col 0", "col 1", "col 2"],
            yticklabels=["row 0", "row 1", "row 2"],
        )

 
    # 4. Fusion weights
  
    fusion = summary.get("fusion_weights_mean", None)
    if fusion:
        labels = list(fusion.keys())
        values = [float(fusion[k]) for k in labels]

        save_bar(
            labels,
            values,
            "Mean Fusion Weights",
            "Mean weight",
            out_dir / "fusion_weights.png",
            rotate=False,
        )

  
    # 5. Land-cover metrics CSV

    lc_csv = summary_path.with_name(summary_path.stem + "_landcover_metrics.csv")

    if lc_csv.exists():
        lc_df = pd.read_csv(lc_csv)

        if "nlcd_name" in lc_df.columns:
            name_col = "nlcd_name"
        else:
            name_col = "nlcd_class"

        lc_df = lc_df.sort_values("n", ascending=False).head(args.top_n)

        labels = lc_df[name_col].astype(str).tolist()

        if "mae" in lc_df.columns:
            save_horizontal_bar(
                labels,
                lc_df["mae"].astype(float).tolist(),
                f"MAE by Land-Cover Class, Top {args.top_n} by Sample Count",
                "MAE",
                out_dir / "landcover_mae.png",
            )

        if "rmse" in lc_df.columns:
            save_horizontal_bar(
                labels,
                lc_df["rmse"].astype(float).tolist(),
                f"RMSE by Land-Cover Class, Top {args.top_n} by Sample Count",
                "RMSE",
                out_dir / "landcover_rmse.png",
            )

        if "bias" in lc_df.columns:
            save_horizontal_bar(
                labels,
                lc_df["bias"].astype(float).tolist(),
                f"Bias by Land-Cover Class, Top {args.top_n} by Sample Count",
                "Bias, prediction minus truth",
                out_dir / "landcover_bias.png",
            )

        if "n" in lc_df.columns:
            save_horizontal_bar(
                labels,
                lc_df["n"].astype(float).tolist(),
                f"Sample Count by Land-Cover Class, Top {args.top_n}",
                "Number of samples",
                out_dir / "landcover_sample_count.png",
            )

        # Land-cover × channel-attention heatmap, only if attention columns exist
        chan_cols = [c for c in lc_df.columns if c.startswith("chan_attn_")]
        if chan_cols:
            heat = lc_df[chan_cols].astype(float).to_numpy()
            band_labels = [c.replace("chan_attn_", "") for c in chan_cols]

            save_heatmap(
                heat,
                "Channel Attention by Land-Cover Class",
                out_dir / "landcover_channel_attention_heatmap.png",
                xticklabels=band_labels,
                yticklabels=labels,
            )

    print("Saved plots to:", out_dir)


if __name__ == "__main__":
    main()