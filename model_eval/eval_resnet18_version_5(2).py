# for this file we edit path variable in the bash .sh file 

import os
import math
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models

import rasterio
from rasterio.transform import xy as rio_xy
from rasterio.errors import RasterioIOError
from pyproj import Transformer
from collections import defaultdict

# Geohash utils (same as yours)

BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
BASE32_MAP = {c: i for i, c in enumerate(BASE32)}


######################### edit here

fixed_bands = ["B02","B03","B04","B06","B07","B8A","B11","B12"]


#########################

def eval_by_landcover(model, dataset, device, batch_size=256):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    base_ds = dataset.dataset if hasattr(dataset, "dataset") else dataset
    

    # store per-class lists
    per_class_abs_err = defaultdict(list)
    per_class_sq_err  = defaultdict(list)
    per_class_n       = defaultdict(int)

    y_all, p_all = [], []

    model.eval()
    with torch.no_grad():
        for x_img, gh_idx, x_aux, y, nlcd_class in loader:
            x_img = x_img.to(device)
            gh_idx = gh_idx.to(device)
            x_aux = x_aux.to(device)
            y = y.to(device)

            #pred = model(x_img, gh_idx, x_aux)
            pred, _, _, _ = model.forward_with_attn(x_img, gh_idx, x_aux)

            # center NLCD (B,) -> to CPU ints
            nlcd_center = nlcd_class.detach().cpu().numpy()
            

            y_np = y.detach().cpu().numpy()
            p_np = pred.detach().cpu().numpy()

            y_all.append(y_np)
            p_all.append(p_np)

            abs_err = np.abs(p_np - y_np)
            sq_err  = (p_np - y_np) ** 2

            for k in range(len(y_np)):
                cls = int(round(float(nlcd_center[k])))
                per_class_abs_err[cls].append(float(abs_err[k]))
                per_class_sq_err[cls].append(float(sq_err[k]))
                per_class_n[cls] += 1

    y_all = np.concatenate(y_all)
    p_all = np.concatenate(p_all)

    # overall
    overall_mae = float(np.mean(np.abs(p_all - y_all)))
    overall_rmse = float(np.sqrt(np.mean((p_all - y_all) ** 2)))

    # per-class table
    rows = []
    for cls, n in sorted(per_class_n.items(), key=lambda x: x[1], reverse=True):
        mae = float(np.mean(per_class_abs_err[cls]))
        rmse = float(np.sqrt(np.mean(per_class_sq_err[cls])))
        rows.append({"nlcd_class": cls, "n": n, "mae": mae, "rmse": rmse})

    return overall_mae, overall_rmse, pd.DataFrame(rows)



def eval_attention_by_landcover(model, dataset, device, batch_size=256):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    base_ds = dataset.dataset if hasattr(dataset, "dataset") else dataset
    bands = base_ds.FIXED_BANDS
    

    per_class_n = defaultdict(int)
    per_class_abs_err = defaultdict(float)
    per_class_sq_err = defaultdict(float)
    per_class_bias = defaultdict(float)

    per_class_chan_sum = defaultdict(lambda: np.zeros((len(bands),), dtype=np.float64))
    per_class_spat_sum = defaultdict(lambda: np.zeros((3, 3), dtype=np.float64))
    per_class_fuse_sum = defaultdict(lambda: np.zeros((3,), dtype=np.float64))

    model.eval()
    with torch.no_grad():
        for x_img, gh_idx, x_aux, y, nlcd_class in loader:
            x_img = x_img.to(device)
            gh_idx = gh_idx.to(device)
            x_aux = x_aux.to(device)
            y = y.to(device)

            pred, w_chan, w_spat, w_fuse = model.forward_with_attn(x_img, gh_idx, x_aux)

            bsz = x_img.size(0)

            nlcd_center = nlcd_class.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            p_np = pred.detach().cpu().numpy()
            err_np = p_np - y_np

            wc = w_chan.detach().cpu().numpy().reshape(bsz, len(bands))
            ws = w_spat.detach().cpu().numpy().reshape(bsz, 3, 3)
            wf = w_fuse.detach().cpu().numpy().reshape(bsz, 3)

            for k in range(bsz):
                cls = int(round(float(nlcd_center[k])))

                per_class_n[cls] += 1
                per_class_abs_err[cls] += float(abs(err_np[k]))
                per_class_sq_err[cls] += float(err_np[k] ** 2)
                per_class_bias[cls] += float(err_np[k])

                per_class_chan_sum[cls] += wc[k]
                per_class_spat_sum[cls] += ws[k]
                per_class_fuse_sum[cls] += wf[k]

    rows = []
    for cls, n in sorted(per_class_n.items(), key=lambda x: x[1], reverse=True):
        row = {
            "nlcd_class": cls,
            "n": n,
            "mae": per_class_abs_err[cls] / n,
            "rmse": np.sqrt(per_class_sq_err[cls] / n),
            "bias": per_class_bias[cls] / n,
        }

        # mean channel attention by band
        chan_mean = per_class_chan_sum[cls] / n
        for band, value in zip(bands, chan_mean):
            row[f"chan_attn_{band}"] = float(value)

        # mean spatial attention by 3x3 position
        spat_mean = per_class_spat_sum[cls] / n
        for r in range(3):
            for c in range(3):
                row[f"spat_attn_r{r}_c{c}"] = float(spat_mean[r, c])

        # mean fusion attention
        fuse_mean = per_class_fuse_sum[cls] / n
        row["fusion_attn_img"] = float(fuse_mean[0])
        row["fusion_attn_geohash"] = float(fuse_mean[1])
        row["fusion_attn_aux"] = float(fuse_mean[2])

        rows.append(row)

    return pd.DataFrame(rows)


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
        date_token = granule_id.split(".")[3]  # "2021057T190939"
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
# Köppen legend + safe lookup
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
    """Worker-safe: opens raster per PID and retries once."""
    def __init__(self, koppen_tif_path: str, legend_path: str, unknown_label="UNK"):
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
# Dataset (same behavior)
# ----------------------------
class GEDIHlsPatchDatasetFusion(Dataset):
    FIXED_BANDS = fixed_bands

    def __init__(self, csv_path: str, geohash_precision: int = 7, target_col: str = "agbd_center",
                 koppen_tif_path=None, koppen_legend_path=None):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.geohash_precision = geohash_precision
        self.target_col = target_col

        self.koppen_lookup = None
        if koppen_tif_path and koppen_legend_path:
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
        nlcd_patch = np.zeros((3, 3), dtype=np.float32)

        if isinstance(bands_str, str) and len(bands_str) > 0:
            used_bands = [b.strip() for b in bands_str.split(",")]

            for i, b in enumerate(used_bands):
                if i >= raw.shape[0]:
                    break

                if b in self.FIXED_BANDS:
                    out[self.FIXED_BANDS.index(b)] = raw[i]

                if b == "NLCD":
                    nlcd_patch = raw[i]
        else:
            C = min(raw.shape[0], out.shape[0])
            out[:C] = raw[:C]

        for b in self.FIXED_BANDS:
            j = self.FIXED_BANDS.index(b)
            mx = float(np.max(np.abs(out[j])))

            if mx > 2.0:
                out[j] = out[j] / 10000.0

        return out, nlcd_patch, crs, transform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patch_path = Path(row["patch_tifs"])
        if not patch_path.is_absolute():
            patch_path = self.csv_path.parent / patch_path

        bands_str = row["bands"] if "bands" in self.df.columns else None
        x_img_np, nlcd_patch, crs, transform = self._load_patch_fixed_channels(patch_path, bands_str)
        x_img = torch.from_numpy(x_img_np)

        # center lat/lon
        x, y = rio_xy(transform, 1, 1, offset="center")
        if crs is None:
            lat, lon = 0.0, 0.0
        else:
            tfm = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            lon, lat = tfm.transform(x, y)

        # month
        if "month" in self.df.columns:
            m = int(row["month"])
        elif "hls_granule" in self.df.columns:
            m = parse_hls_month_from_granule(str(row["hls_granule"]))
        else:
            m = 1
        m_sc = month_sincos(m)

        # geohash
        gh = geohash_encode(lat, lon, precision=self.geohash_precision)
        gh_idx = torch.from_numpy(geohash_to_indices(gh))

        # koppen
        if self.koppen_lookup is not None:
            koppen_onehot = self.koppen_lookup.encode_onehot(lon, lat)
        else:
            koppen_onehot = np.zeros((0,), dtype=np.float32)

        x_aux_np = np.concatenate([m_sc, koppen_onehot], axis=0).astype(np.float32)
        x_aux = torch.from_numpy(x_aux_np)

        y = torch.tensor(float(row[self.target_col]), dtype=torch.float32)

        nlcd_class = torch.tensor(
            int(round(float(nlcd_patch[1, 1]))),
            dtype=torch.long
        )
        return x_img, gh_idx.long(), x_aux, y, nlcd_class

# ----------------------------
# Attention blocks (return weights)
# ----------------------------
class ChannelSE(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x, return_w=False):
        w = self.net(x)  # (B,C,1,1)
        y = x * w
        return (y, w) if return_w else y

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        p = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=p, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, return_w=False):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        w = self.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))  # (B,1,H,W)
        y = x * w
        return (y, w) if return_w else y

class FusionGate(nn.Module):
    def __init__(self, img_dim: int, gh_dim: int, aux_dim: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(img_dim + gh_dim + aux_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3)
        )

    def forward(self, img_feat, gh_feat, x_aux):
        z = torch.cat([img_feat, gh_feat, x_aux], dim=1)
        w = torch.softmax(self.mlp(z), dim=1)  # (B,3)
        return img_feat * w[:, 0:1], gh_feat * w[:, 1:2], x_aux * w[:, 2:3], w

# ----------------------------
# Model (same weights, plus a helper to get attention)
# ----------------------------
class ResNet18FusionRegressorAttn(nn.Module):
    def __init__(self, in_channels: int, aux_dim: int, geohash_emb_dim: int = 16,
                 hidden_dim: int = 256, dropout: float = 0.2,
                 use_channel_attn=True, use_spatial_attn=True, use_fusion_attn=True):
        super().__init__()
        self.use_channel_attn = use_channel_attn
        self.use_spatial_attn = use_spatial_attn
        self.use_fusion_attn = use_fusion_attn

        if use_channel_attn:
            self.channel_attn = ChannelSE(in_channels, reduction=4)
        if use_spatial_attn:
            self.spatial_attn = SpatialAttention(kernel_size=3)

        m = models.resnet18(weights=None)
        m.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        m.fc = nn.Identity()
        self.backbone = m

        self.gh_emb = nn.Embedding(32, geohash_emb_dim)

        if use_fusion_attn:
            self.fusion_gate = FusionGate(img_dim=512, gh_dim=geohash_emb_dim, aux_dim=aux_dim, hidden=128)

        fused_dim = 512 + geohash_emb_dim + aux_dim
        self.regressor = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x_img, gh_idx, x_aux):
        img_feat = self.backbone(x_img)
        gh_feat = self.gh_emb(gh_idx).mean(dim=1)
        fused = torch.cat([img_feat, gh_feat, x_aux], dim=1)
        return self.regressor(fused).squeeze(1)

    def forward_with_attn(self, x_img, gh_idx, x_aux):
        w_chan = None
        w_spat = None
        w_fuse = None

        if self.use_channel_attn:
            x_img, w_chan = self.channel_attn(x_img, return_w=True)
        if self.use_spatial_attn:
            x_img, w_spat = self.spatial_attn(x_img, return_w=True)

        img_feat = self.backbone(x_img)
        gh_feat = self.gh_emb(gh_idx).mean(dim=1)

        if self.use_fusion_attn:
            img_feat, gh_feat, x_aux, w_fuse = self.fusion_gate(img_feat, gh_feat, x_aux)

        fused = torch.cat([img_feat, gh_feat, x_aux], dim=1)
        pred = self.regressor(fused).squeeze(1)
        return pred, w_chan, w_spat, w_fuse

# ----------------------------
# Metrics
# ----------------------------
def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))

# ----------------------------
# Main eval
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--geohash-precision", type=int, default=7)
    ap.add_argument("--out", default="attn_eval_summary.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    koppen_tif = os.environ.get(
        "KOPPEN_TIF",
        "/s/chopin/e/proj/hyperspec/masfiq/dataset/koppen_geiger_tif/1991_2020/koppen_geiger_0p00833333.tif"
    )
    legend = "/s/chopin/e/proj/hyperspec/masfiq/dataset/koppen_geiger_tif/legend.txt"

    ds = GEDIHlsPatchDatasetFusion(
        args.csv,
        geohash_precision=args.geohash_precision,
        target_col="agbd_center",
        koppen_tif_path=koppen_tif,
        koppen_legend_path=legend
    )

    # deterministic split
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
        prefetch_factor=4
    )

    # infer aux_dim from one sample
    x_img0, gh0, x_aux0, y0, nlcd0 = ds[0]
    print("Sample NLCD class =", nlcd0.item())
    in_channels = x_img0.shape[0]
    aux_dim = x_aux0.shape[0]
    print("in_channels =", in_channels, "aux_dim =", aux_dim)

    model = ResNet18FusionRegressorAttn(
        in_channels=in_channels,
        aux_dim=aux_dim,
        geohash_emb_dim=16,
        hidden_dim=256,
        dropout=0.2,
        use_channel_attn=True,
        use_spatial_attn=True,
        use_fusion_attn=True
    ).to(device)

    state = torch.load(args.ckpt, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()

    bands = GEDIHlsPatchDatasetFusion.FIXED_BANDS
    chan_sum = np.zeros((len(bands),), dtype=np.float64)
    spat_sum = np.zeros((3, 3), dtype=np.float64)
    fuse_sum = np.zeros((3,), dtype=np.float64)
    n_seen = 0

    y_true_all = []
    y_pred_all = []

    use_amp = (device.type == "cuda")

    with torch.no_grad():
        for x_img, gh_idx, x_aux, y, nlcd_class in val_loader:
            x_img = x_img.to(device, non_blocking=True)
            gh_idx = gh_idx.to(device, non_blocking=True)
            x_aux  = x_aux.to(device, non_blocking=True)
            y      = y.to(device, non_blocking=True)

            if use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    pred, w_chan, w_spat, w_fuse = model.forward_with_attn(x_img, gh_idx, x_aux)
            else:
                pred, w_chan, w_spat, w_fuse = model.forward_with_attn(x_img, gh_idx, x_aux)

            y_true_all.append(y.detach().cpu().numpy())
            y_pred_all.append(pred.detach().cpu().numpy())

            bsz = x_img.size(0)
            n_seen += bsz

            if w_chan is not None:
                wc = w_chan.detach().cpu().numpy().reshape(bsz, len(bands))  # (B,C)
                chan_sum += wc.sum(axis=0)

            if w_spat is not None:
                ws = w_spat.detach().cpu().numpy().reshape(bsz, 3, 3)  # (B,3,3)
                spat_sum += ws.sum(axis=0)

            if w_fuse is not None:
                wf = w_fuse.detach().cpu().numpy()  # (B,3)
                fuse_sum += wf.sum(axis=0)

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    mae = float(np.mean(np.abs(y_pred_all - y_true_all)))
    rmse = float(np.sqrt(np.mean((y_pred_all - y_true_all) ** 2)))
    r2 = r2_score(y_true_all, y_pred_all)

    chan_mean = (chan_sum / max(1, n_seen)).tolist()
    spat_mean = (spat_sum / max(1, n_seen)).tolist()
    fuse_mean = (fuse_sum / max(1, n_seen)).tolist()

    # top-5 bands
    top5 = sorted(zip(bands, chan_mean), key=lambda x: x[1], reverse=True)[:5]

    print("\n=== VAL METRICS ===")
    print(f"MAE  = {mae:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"R^2  = {r2:.4f}")

    print("\n=== TOP-5 CHANNEL ATTENTION (mean weight) ===")
    for b, w in top5:
        print(f"{b:>4s} : {w:.4f}")

    print("\n=== MEAN SPATIAL ATTENTION MAP (3x3) ===")
    for r in range(3):
        print(" ".join([f"{spat_mean[r][c]:.3f}" for c in range(3)]))

    print("\n=== MEAN FUSION WEIGHTS (img, geo, aux) ===")
    print(f"img={fuse_mean[0]:.4f}, geo={fuse_mean[1]:.4f}, aux={fuse_mean[2]:.4f}")

    out = {
        "ckpt": args.ckpt,
        "csv": args.csv,
        "seed": args.seed,
        "val_frac": args.val_frac,
        "metrics": {"mae": mae, "rmse": rmse, "r2": r2},
        "top5_channel_attention": [{"band": b, "mean_weight": float(w)} for b, w in top5],
        "channel_attention_mean": {b: float(w) for b, w in zip(bands, chan_mean)},
        "spatial_attention_mean_3x3": spat_mean,
        "fusion_weights_mean": {"img": float(fuse_mean[0]), "geo": float(fuse_mean[1]), "aux": float(fuse_mean[2])},
    }
    Path(args.out).write_text(pd.Series(out).to_json(), encoding="utf-8")
    print("\nSaved summary to:", args.out)

        # ----------------------------
    # Land cover sensitivity analysis
    # ----------------------------
    print("\n=== LAND COVER SENSITIVITY ANALYSIS ===")

    overall_mae_lc, overall_rmse_lc, lc_df = eval_by_landcover(
        model=model,
        dataset=val_ds,
        device=device,
        batch_size=args.batch_size
    )

    lc_df = add_nlcd_names(lc_df)

    # optional: bias per class
    # recompute bias by class
    loader_lc = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    

    from collections import defaultdict
    per_class_err = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for x_img, gh_idx, x_aux, y, nlcd_class in loader_lc:
            x_img = x_img.to(device)
            gh_idx = gh_idx.to(device)
            x_aux = x_aux.to(device)
            y = y.to(device)

            #pred = model(x_img, gh_idx, x_aux)
            pred, _, _, _ = model.forward_with_attn(x_img, gh_idx, x_aux)


            nlcd_center = nlcd_class.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            p_np = pred.detach().cpu().numpy()
            err_np = p_np - y_np

            for k in range(len(y_np)):
                cls = int(round(float(nlcd_center[k])))
                per_class_err[cls].append(float(err_np[k]))

    bias_map = {}
    for cls, errs in per_class_err.items():
        bias_map[cls] = float(np.mean(errs))

    lc_df["bias"] = lc_df["nlcd_class"].map(bias_map)
    # sort by sample count
    lc_df = lc_df.sort_values("n", ascending=False).reset_index(drop=True)

    print(f"Overall MAE  from land-cover eval = {overall_mae_lc:.4f}")
    print(f"Overall RMSE from land-cover eval = {overall_rmse_lc:.4f}")
    print("\nPer-land-cover results:")
    print(lc_df.to_string(index=False))


    # ----------------------------
    # Land-cover attention interpretation
    # ----------------------------
    print("\n=== LAND COVER ATTENTION INTERPRETATION ===")

    lc_attn_df = eval_attention_by_landcover(
        model=model,
        dataset=val_ds,
        device=device,
        batch_size=args.batch_size
    )

    # This also adds nlcd_name, but we will avoid duplicate columns during merge
    lc_attn_df = add_nlcd_names(lc_attn_df)

    # Keep only attention columns from lc_attn_df.
    # Drop duplicate performance/name columns because lc_df already has them.
    drop_cols = ["n", "mae", "rmse", "bias", "nlcd_name"]
    drop_cols = [c for c in drop_cols if c in lc_attn_df.columns]

    lc_attn_only = lc_attn_df.drop(columns=drop_cols)

    # Merge normal land-cover metrics + attention interpretation into one table
    lc_combined_df = lc_df.merge(
        lc_attn_only,
        on="nlcd_class",
        how="left"
    )

    # Sort again
    lc_combined_df = lc_combined_df.sort_values("n", ascending=False).reset_index(drop=True)

    print("\nCombined land-cover metrics + attention interpretation:")
    print(lc_combined_df.to_string(index=False))

    # Save everything in the same CSV file
    # This line will overwrite the CSV file if the same lc_out_csv already exists:
    lc_out_csv = Path(args.out).with_name(Path(args.out).stem + "_landcover_metrics.csv")
    lc_combined_df.to_csv(lc_out_csv, index=False)

    print("\nSaved combined land-cover metrics and attention interpretation to:", lc_out_csv)






if __name__ == "__main__":
    main()