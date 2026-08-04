import os
import math
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

import rasterio
from rasterio.transform import xy as rio_xy
from rasterio.errors import RasterioIOError
from pyproj import Transformer


# ----------------------------
# Geohash utilities
# ----------------------------
BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
BASE32_MAP = {c: i for i, c in enumerate(BASE32)}


def geohash_encode(lat: float, lon: float, precision: int = 7) -> str:
    lat_interval = [-90.0, 90.0]
    lon_interval = [-180.0, 180.0]

    geohash = []
    is_even = True
    bit = 0
    ch = 0
    bits = [16, 8, 4, 2, 1]

    while len(geohash) < precision:
        if is_even:
            mid = (lon_interval[0] + lon_interval[1]) / 2
            if lon >= mid:
                ch |= bits[bit]
                lon_interval[0] = mid
            else:
                lon_interval[1] = mid
        else:
            mid = (lat_interval[0] + lat_interval[1]) / 2
            if lat >= mid:
                ch |= bits[bit]
                lat_interval[0] = mid
            else:
                lat_interval[1] = mid

        is_even = not is_even

        if bit < 4:
            bit += 1
        else:
            geohash.append(BASE32[ch])
            bit = 0
            ch = 0

    return "".join(geohash)


def geohash_to_indices(gh: str) -> np.ndarray:
    return np.array([BASE32_MAP.get(c, 0) for c in gh], dtype=np.int64)


def parse_hls_month_from_granule(granule_id: str) -> int:
    try:
        date_token = granule_id.split(".")[3]
        y = int(date_token[0:4])
        doy = int(date_token[4:7])
        dt = datetime(y, 1, 1) + timedelta(days=doy - 1)
        return dt.month
    except Exception:
        return 1


def month_sincos(month: int) -> np.ndarray:
    ang = 2.0 * math.pi * (float(month) / 12.0)
    return np.array([math.sin(ang), math.cos(ang)], dtype=np.float32)


# ----------------------------
# Köppen lookup
# ----------------------------
def load_koppen_legend(legend_path: str):
    code_to_label = {}

    print("\n[KOPPEN DEBUG] Loading legend from:", legend_path)
    print("[KOPPEN DEBUG] Legend exists:", Path(legend_path).exists())

    with open(legend_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            # Expected format:
            # 1:  Af   Tropical, rainforest   [0 0 255]
            if ":" not in line:
                continue

            code_part, rest = line.split(":", 1)
            code_part = code_part.strip()

            if not code_part.isdigit():
                continue

            code = int(code_part)

            rest_parts = rest.strip().split()
            if len(rest_parts) == 0:
                continue

            label = rest_parts[0]
            code_to_label[code] = label

    codes_sorted = sorted(code_to_label.keys())
    code_to_index = {c: i for i, c in enumerate(codes_sorted)}

    print("[KOPPEN DEBUG] Parsed legend class count:", len(codes_sorted))
    print("[KOPPEN DEBUG] First 20 parsed codes:", codes_sorted[:20])
    print("[KOPPEN DEBUG] First 20 code_to_label:", list(code_to_label.items())[:20])

    return code_to_label, codes_sorted, code_to_index


class KoppenGeigerLookup:
    def __init__(self, koppen_tif_path: str, legend_path: str):
        self.koppen_tif_path = koppen_tif_path
        self.legend_path = legend_path

        self.code_to_label, self.codes_sorted, self.code_to_index = load_koppen_legend(legend_path)

        self.unknown_index = len(self.codes_sorted)
        self.num_classes = len(self.codes_sorted) + 1

        self.src = None
        self.nodata = None
        self.to_raster = None
        self._pid = None

    def __getstate__(self):
        d = self.__dict__.copy()
        d["src"] = None
        d["to_raster"] = None
        d["_pid"] = None
        return d

    def _ensure_open(self):
        pid = os.getpid()

        if self.src is None or self._pid != pid:
            if self.src is not None:
                try:
                    self.src.close()
                except Exception:
                    pass

            self.src = rasterio.open(self.koppen_tif_path)
            self.nodata = self.src.nodata
            self.to_raster = Transformer.from_crs("EPSG:4326", self.src.crs, always_xy=True)
            self._pid = pid

    def close(self):
        try:
            if self.src is not None:
                self.src.close()
        except Exception:
            pass

        self.src = None
        self.to_raster = None
        self._pid = None

    def encode_onehot(self, lon: float, lat: float) -> np.ndarray:
        self._ensure_open()
        x, y = self.to_raster.transform(lon, lat)

        for attempt in (1, 2):
            try:
                val = next(self.src.sample([(x, y)]))[0]

                if self.nodata is not None and val == self.nodata:
                    code = -9999
                else:
                    code = int(val)

                onehot = np.zeros((self.num_classes,), dtype=np.float32)
                idx = self.code_to_index.get(code, self.unknown_index)
                onehot[idx] = 1.0

                return onehot

            except (RasterioIOError, StopIteration, ValueError):
                if attempt == 1:
                    self.close()
                    self._ensure_open()
                    continue

                onehot = np.zeros((self.num_classes,), dtype=np.float32)
                onehot[self.unknown_index] = 1.0
                return onehot


# ----------------------------
# Dataset
# ----------------------------
class GEDIHlsPatchDatasetFusion(Dataset):
    FIXED_BANDS = [
        "B02", "B03", "B04", "B05", "B06", "B07",
        "B8A", "B11", "B12", "EVI", "NDVI", "NLCD"
    ]

    def __init__(
        self,
        csv_path: str,
        geohash_precision: int = 7,
        target_col: str = "agbd_center",
        koppen_tif_path=None,
        koppen_legend_path=None,
    ):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.geohash_precision = geohash_precision
        self.target_col = target_col

        self.koppen_lookup = None

        if koppen_tif_path is not None and koppen_legend_path is not None:
            self.koppen_lookup = KoppenGeigerLookup(koppen_tif_path, koppen_legend_path)

    def __len__(self):
        return len(self.df)

    def _load_patch_fixed_channels(self, patch_path: Path, bands_str: str | None):
        with rasterio.open(patch_path) as src:
            raw = src.read().astype(np.float32)
            nodata = src.nodata
            crs = src.crs
            transform = src.transform

        if nodata is not None:
            raw[raw == nodata] = np.nan

        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

        out = np.zeros((len(self.FIXED_BANDS), 3, 3), dtype=np.float32)

        if isinstance(bands_str, str) and len(bands_str) > 0:
            used_bands = [b.strip() for b in bands_str.split(",")]

            for i, b in enumerate(used_bands):
                if i >= raw.shape[0]:
                    break

                if b in self.FIXED_BANDS:
                    out[self.FIXED_BANDS.index(b)] = raw[i]
        else:
            c = min(raw.shape[0], out.shape[0])
            out[:c] = raw[:c]

        for b in self.FIXED_BANDS:
            if b == "NLCD":
                continue

            j = self.FIXED_BANDS.index(b)
            mx = float(np.max(np.abs(out[j])))

            if mx > 2.0:
                out[j] = out[j] / 10000.0

        return out, crs, transform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        patch_path = Path(row["patch_tifs"])

        if not patch_path.is_absolute():
            patch_path = self.csv_path.parent / patch_path

        bands_str = row["bands"] if "bands" in self.df.columns else None
        x_img_np, crs, transform = self._load_patch_fixed_channels(patch_path, bands_str)

        x_img = torch.from_numpy(x_img_np)

        x, y = rio_xy(transform, 1, 1, offset="center")

        if crs is None:
            lat, lon = 0.0, 0.0
        else:
            tfm = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            lon, lat = tfm.transform(x, y)

        if "month" in self.df.columns:
            m = int(row["month"])
        elif "hls_granule" in self.df.columns:
            m = parse_hls_month_from_granule(str(row["hls_granule"]))
        else:
            m = 1

        m_sc = month_sincos(m)

        gh = geohash_encode(lat, lon, precision=self.geohash_precision)
        gh_idx = torch.from_numpy(geohash_to_indices(gh))

        if self.koppen_lookup is not None:
            koppen_onehot = self.koppen_lookup.encode_onehot(lon, lat)
        else:
            koppen_onehot = np.zeros((0,), dtype=np.float32)

        x_aux_np = np.concatenate([m_sc, koppen_onehot], axis=0).astype(np.float32)
        x_aux = torch.from_numpy(x_aux_np)

        y_val = float(row[self.target_col])

        if not np.isfinite(y_val):
            raise ValueError(f"Non-finite target at index {idx}: {y_val}")

        y = torch.tensor(y_val, dtype=torch.float32)

        return x_img, gh_idx.long(), x_aux, y


# ----------------------------
# Tiny Pixel ViT model
# Must match your training model exactly.
# ----------------------------
class TinyPixelViTFusionRegressor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        geohash_precision: int,
        aux_dim: int,
        geohash_emb_dim: int = 16,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        mlp_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_tokens = 9

        self.token_proj = nn.Linear(in_channels, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.gh_emb = nn.Embedding(32, geohash_emb_dim)

        fused_dim = embed_dim + geohash_emb_dim + aux_dim

        self.regressor = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x_img, gh_idx, x_aux):
        b, c, h, w = x_img.shape

        x = x_img.permute(0, 2, 3, 1).reshape(b, h * w, c)
        x = self.token_proj(x)

        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)

        x = x + self.pos_embed
        x = self.transformer(x)

        img_feat = self.norm(x[:, 0, :])

        gh_feat = self.gh_emb(gh_idx).mean(dim=1)

        fused = torch.cat([img_feat, gh_feat, x_aux], dim=1)

        out = self.regressor(fused)
        return out.squeeze(1)


# ----------------------------
# Metrics
# ----------------------------
def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    return float(1.0 - ss_res / (ss_tot + 1e-12))


def add_nlcd_names(df):
    nlcd_name_map = {
        11: "Open Water",
        12: "Perennial Ice/Snow",
        21: "Developed, Open Space",
        22: "Developed, Low Intensity",
        23: "Developed, Medium Intensity",
        24: "Developed, High Intensity",
        31: "Barren Land",
        41: "Deciduous Forest",
        42: "Evergreen Forest",
        43: "Mixed Forest",
        51: "Dwarf Scrub",
        52: "Shrub/Scrub",
        71: "Grassland/Herbaceous",
        72: "Sedge/Herbaceous",
        73: "Lichens",
        74: "Moss",
        81: "Pasture/Hay",
        82: "Cultivated Crops",
        90: "Woody Wetlands",
        95: "Emergent Herbaceous Wetlands",
    }

    df = df.copy()
    df["nlcd_name"] = df["nlcd_class"].map(nlcd_name_map).fillna("Unknown")

    return df


def eval_by_landcover(model, dataset, device, batch_size=256):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    base_ds = dataset.dataset if hasattr(dataset, "dataset") else dataset
    nlcd_idx = base_ds.FIXED_BANDS.index("NLCD")

    per_class_abs_err = defaultdict(list)
    per_class_sq_err = defaultdict(list)
    per_class_err = defaultdict(list)
    per_class_n = defaultdict(int)

    y_all = []
    p_all = []

    model.eval()

    with torch.no_grad():
        for x_img, gh_idx, x_aux, y in loader:
            x_img = x_img.to(device)
            gh_idx = gh_idx.to(device)
            x_aux = x_aux.to(device)
            y = y.to(device)

            pred = model(x_img, gh_idx, x_aux)

            nlcd_center = x_img[:, nlcd_idx, 1, 1].detach().cpu().numpy()

            y_np = y.detach().cpu().numpy()
            p_np = pred.detach().cpu().numpy()

            y_all.append(y_np)
            p_all.append(p_np)

            err = p_np - y_np
            abs_err = np.abs(err)
            sq_err = err ** 2

            for k in range(len(y_np)):
                cls = int(round(float(nlcd_center[k])))

                per_class_abs_err[cls].append(float(abs_err[k]))
                per_class_sq_err[cls].append(float(sq_err[k]))
                per_class_err[cls].append(float(err[k]))
                per_class_n[cls] += 1

    y_all = np.concatenate(y_all)
    p_all = np.concatenate(p_all)

    overall_mae = float(np.mean(np.abs(p_all - y_all)))
    overall_rmse = float(np.sqrt(np.mean((p_all - y_all) ** 2)))
    overall_r2 = r2_score(y_all, p_all)

    rows = []

    for cls, n in sorted(per_class_n.items(), key=lambda x: x[1], reverse=True):
        mae = float(np.mean(per_class_abs_err[cls]))
        rmse = float(np.sqrt(np.mean(per_class_sq_err[cls])))
        bias = float(np.mean(per_class_err[cls]))

        rows.append({
            "nlcd_class": cls,
            "n": n,
            "mae": mae,
            "rmse": rmse,
            "bias": bias,
        })

    return overall_mae, overall_rmse, overall_r2, pd.DataFrame(rows)


# ----------------------------
# Main evaluation
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", default="/s/chopin/e/proj/hyperspec/masfiq/csv_files/gedi_field_boundary_2021_AprilToAugust_version_3.csv")
    ap.add_argument("--ckpt", default="/s/chopin/e/proj/hyperspec/masfiq/models/tinyPixelViT_fusion_geohash_month_koppen_field_boundary_version_3.pth")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--geohash-precision", type=int, default=7)
    ap.add_argument("--out", default="tinyPixelViT_eval_field_boundary.json")

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    koppen_tif = os.environ.get(
        "KOPPEN_TIF",
        "/s/chopin/e/proj/hyperspec/masfiq/dataset/koppen_geiger_tif/1991_2020/koppen_geiger_0p00833333.tif",
    )

    legend = "/s/chopin/e/proj/hyperspec/masfiq/dataset/koppen_geiger_tif/legend.txt"

    ds = GEDIHlsPatchDatasetFusion(
        args.csv,
        geohash_precision=args.geohash_precision,
        target_col="agbd_center",
        koppen_tif_path=koppen_tif,
        koppen_legend_path=legend,
    )

    # Deterministic validation split.
    N = len(ds)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(N)

    val_size = int(N * args.val_frac)
    val_idx = idx[:val_size]
    val_ds = Subset(ds, val_idx)

    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    num_workers = min(8, max(2, num_workers))

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
        prefetch_factor=4,
    )

    x_img0, gh0, x_aux0, y0 = ds[0]
    in_channels = x_img0.shape[0]
    aux_dim = x_aux0.shape[0]

    print("in_channels =", in_channels, "aux_dim =", aux_dim)

    model = TinyPixelViTFusionRegressor(
        in_channels=in_channels,
        geohash_precision=args.geohash_precision,
        aux_dim=aux_dim,
        geohash_emb_dim=16,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        mlp_dim=128,
        hidden_dim=256,
        dropout=0.2,
    ).to(device)

    state = torch.load(args.ckpt, map_location=device)

    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state, strict=True)
    model.eval()

    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for x_img, gh_idx, x_aux, y in val_loader:
            x_img = x_img.to(device, non_blocking=True)
            gh_idx = gh_idx.to(device, non_blocking=True)
            x_aux = x_aux.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred = model(x_img, gh_idx, x_aux)

            y_true_all.append(y.detach().cpu().numpy())
            y_pred_all.append(pred.detach().cpu().numpy())

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    mae = float(np.mean(np.abs(y_pred_all - y_true_all)))
    rmse = float(np.sqrt(np.mean((y_pred_all - y_true_all) ** 2)))
    r2 = r2_score(y_true_all, y_pred_all)

    print("\n=== VAL METRICS ===")
    print(f"MAE  = {mae:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"R^2  = {r2:.4f}")

    out = {
        "ckpt": args.ckpt,
        "csv": args.csv,
        "seed": args.seed,
        "val_frac": args.val_frac,
        "metrics": {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        },
    }

    Path(args.out).write_text(pd.Series(out).to_json(), encoding="utf-8")
    print("\nSaved summary to:", args.out)

    print("\n=== LAND COVER PERFORMANCE ANALYSIS ===")

    overall_mae_lc, overall_rmse_lc, overall_r2_lc, lc_df = eval_by_landcover(
        model=model,
        dataset=val_ds,
        device=device,
        batch_size=args.batch_size,
    )

    lc_df = add_nlcd_names(lc_df)
    lc_df = lc_df.sort_values("n", ascending=False).reset_index(drop=True)

    print(f"Overall MAE  from land-cover eval = {overall_mae_lc:.4f}")
    print(f"Overall RMSE from land-cover eval = {overall_rmse_lc:.4f}")
    print(f"Overall R^2  from land-cover eval = {overall_r2_lc:.4f}")

    print("\nPer-land-cover results:")
    print(lc_df.to_string(index=False))

    lc_out_csv = Path(args.out).with_name(
        Path(args.out).stem + "_landcover_metrics.csv"
    )

    lc_df.to_csv(lc_out_csv, index=False)

    print("\nSaved land-cover metrics to:", lc_out_csv)


if __name__ == "__main__":
    main()