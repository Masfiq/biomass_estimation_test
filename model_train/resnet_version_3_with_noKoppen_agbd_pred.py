import math
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

import rasterio
from rasterio.transform import xy as rio_xy
from pyproj import Transformer
import os


# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ----------------------------
# Geohash utils (no extra lib)
# ----------------------------
BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
BASE32_MAP = {c: i for i, c in enumerate(BASE32)}

# Convert the patch center to lat/lon (WGS84)
# open the patch GeoTIFF and use its geotransform + CRS.
# For a 3×3 patch, the center pixel is (row=1, col=1).
#  convert that pixel's projected coordinates to EPSG:4326 (lat/lon).

# encode (lat, lon) into a geohash string, then turn it into integers
# geohash_encode(lat, lon, precision=7) builds a 7 character geohash.
# Each character is from base32:
# "0123456789bcdefghjkmnpqrstuvwxyz"
# Then geohash_to_indices() converts each character into an integer 0..31.

# What the model actually receives
# gh_idx is a vector of length G = precision (so 7 numbers),
# like [12, 3, 25, 8, 1, 30, 4].
# Inside the model 
# nn.Embedding(32, emb_dim) to map each of those integers to a small vector
# then take the mean across the 7 characters to get one fixed size geohash feature:

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
    # HLS.S30.T10TFL.2021087T185939.v2.0  -> YYYY+DOY -> month
    try:
        date_token = granule_id.split(".")[3]  # "2021087T185939"
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
# Köppen lookup from raster
# ----------------------------
def load_koppen_legend(legend_path: str):
    code_to_label = {}
    with open(legend_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            try:
                code = int(parts[0])
            except:
                continue
            label = parts[1] if len(parts) > 1 else "UNK"
            code_to_label[code] = label

    codes_sorted = sorted(code_to_label.keys())
    code_to_index = {c: i for i, c in enumerate(codes_sorted)}
    return code_to_label, codes_sorted, code_to_index

class KoppenGeigerLookup:
    def __init__(self, koppen_tif_path: str, legend_path: str):
        self.code_to_label, self.codes_sorted, self.code_to_index = load_koppen_legend(legend_path)
        self.unknown_index = len(self.codes_sorted)
        self.num_classes = len(self.codes_sorted) + 1  # +1 for UNK bucket

        self.src = rasterio.open(koppen_tif_path)
        self.nodata = self.src.nodata
        self.to_raster = Transformer.from_crs("EPSG:4326", self.src.crs, always_xy=True)

    def encode_onehot(self, lon: float, lat: float):
        x, y = self.to_raster.transform(lon, lat)
        val = next(self.src.sample([(x, y)]))[0]

        if self.nodata is not None and val == self.nodata:
            idx = self.unknown_index
        else:
            try:
                code = int(val)
                idx = self.code_to_index.get(code, self.unknown_index)
            except:
                idx = self.unknown_index

        onehot = np.zeros((self.num_classes,), dtype=np.float32)
        onehot[idx] = 1.0
        return onehot, idx


# ----------------------------
# Dataset (image + geohash + month + koppen)
# ----------------------------
class GEDIHlsPatchDatasetFusion(Dataset):
    # Keep a fixed band order so channels are consistent
    FIXED_BANDS = ["B02","B03","B04","B05","B06","B07","B8A","B11","B12","EVI","NDVI","NLCD"]

    def __init__(
        self,
        csv_path: str,
        geohash_precision: int = 7,
        target_col: str | None = None,
        koppen_tif_path="/s/chopin/e/proj/hyperspec/masfiq/dataset/koppen_geiger_tif/1991_2020/koppen_geiger_0p00833333.tif",
        koppen_legend_path= "/s/chopin/e/proj/hyperspec/masfiq/dataset/koppen_geiger_tif/legend.txt",
    ):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.geohash_precision = geohash_precision

        # choose target col automatically if not provided
        if target_col is None:
            if "agbd_mean" in self.df.columns:
                target_col = "agbd_mean"
            elif "agbd_center" in self.df.columns:
                target_col = "agbd_center"
            else:
                raise ValueError("CSV must contain 'agbd_mean' or 'agbd'.")
        self.target_col = target_col

        # init koppen lookup
        self.koppen_lookup = None
        self.koppen_dim = 0
        if koppen_tif_path and koppen_legend_path:
            self.koppen_lookup = KoppenGeigerLookup(koppen_tif_path, koppen_legend_path)
            self.koppen_dim = self.koppen_lookup.num_classes

    def __len__(self):
        return len(self.df)

    def _load_patch_fixed_channels(self, patch_path: Path, bands_str: str | None) -> np.ndarray:
        with rasterio.open(patch_path) as src:
            raw = src.read().astype(np.float32)  # (C,3,3)
            nodata = src.nodata

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
                    j = self.FIXED_BANDS.index(b)
                    out[j] = raw[i]
        else:
            C = min(raw.shape[0], out.shape[0])
            out[:C] = raw[:C]

        # scale reflectance-like bands if needed (don’t scale NLCD)
        for b in self.FIXED_BANDS:
            if b == "NLCD":
                continue
            j = self.FIXED_BANDS.index(b)
            if np.max(np.abs(out[j])) > 2.0:
                out[j] = out[j] / 10000.0

        return out

    def _center_latlon_from_patch(self, patch_path: Path) -> tuple[float, float]:
        with rasterio.open(patch_path) as src:
            crs = src.crs
            transform = src.transform
        x, y = rio_xy(transform, 1, 1, offset="center")  # middle pixel of 3x3
        if crs is None:
            return 0.0, 0.0
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(x, y)
        return float(lat), float(lon)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        patch_path = Path(row["patch_tifs"])
        if not patch_path.is_absolute():
            patch_path = self.csv_path.parent / patch_path
        if not patch_path.exists():
            raise FileNotFoundError(f"Patch file not found: {patch_path}")

        bands_str = row["bands"] if "bands" in self.df.columns else None
        x_img_np = self._load_patch_fixed_channels(patch_path, bands_str)
        x_img = torch.from_numpy(x_img_np)  # (C,3,3)

        # month -> sin/cos
        if "month" in self.df.columns:
            m = int(row["month"])
        elif "hls_granule" in self.df.columns:
            m = parse_hls_month_from_granule(str(row["hls_granule"]))
        else:
            m = 1
        m_sc = month_sincos(m)  # (2,)

        # lat/lon -> geohash indices
        lat, lon = self._center_latlon_from_patch(patch_path)
        gh = geohash_encode(lat, lon, precision=self.geohash_precision)
        gh_idx = torch.from_numpy(geohash_to_indices(gh)).long()  # (G,)

        # koppen one-hot from raster
        if self.koppen_lookup is not None:
            koppen_onehot, _ = self.koppen_lookup.encode_onehot(lon, lat)  # (K,)
        else:
            koppen_onehot = np.zeros((0,), dtype=np.float32)

        x_aux_np = np.concatenate([m_sc, koppen_onehot], axis=0).astype(np.float32)
        x_aux = torch.from_numpy(x_aux_np)  # (A,)

        y = torch.tensor(float(row[self.target_col]), dtype=torch.float32)

        return x_img, gh_idx, x_aux, y


# ----------------------------
# Model (ResNet18 backbone + geohash emb + aux + regressor)
# ----------------------------
class ResNet18FusionRegressor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        aux_dim: int,
        geohash_emb_dim: int = 16,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        m = models.resnet18(weights=None)
        m.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        m.fc = nn.Identity()  # output 512-d features
        self.backbone = m

        self.gh_emb = nn.Embedding(32, geohash_emb_dim)

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
        img_feat = self.backbone(x_img)           # (B,512)
        gh_feat = self.gh_emb(gh_idx).mean(1)     # (B,E)
        fused = torch.cat([img_feat, gh_feat, x_aux], dim=1)
        return self.regressor(fused).squeeze(1)   # (B,)


# ----------------------------
# Inference: save agbd & agbd_pred
# ----------------------------
if __name__ == "__main__":
    # ======= EDIT THESE =======
    csv_path = "/s/chopin/e/proj/hyperspec/masfiq/dataset/csv_files/gedi_field_boundary_metadata.csv"
    checkpoint_path = "/s/chopin/e/proj/hyperspec/masfiq/models/resnet18_fusion_geohash_month_NO_KOPPEN_field_boundary.pth"

    koppen_tif = os.environ.get(
        "KOPPEN_TIF",
        "/s/chopin/e/proj/hyperspec/masfiq/dataset/koppen_geiger_tif/1991_2020/koppen_geiger_0p00833333.tif"
    )
    legend_path = "/s/chopin/e/proj/hyperspec/masfiq/dataset/koppen_geiger_tif/legend.txt"

    out_csv = "/s/chopin/e/proj/hyperspec/masfiq/dataset/csv_files/gedi_with_predictions_fusion_NoKoppen_filed_boundary.csv"
    # ===========================

    dataset = GEDIHlsPatchDatasetFusion(
        csv_path=csv_path,
        geohash_precision=7,
        target_col=None,  # auto: agbd_mean then agbd
        koppen_tif_path=None,
        koppen_legend_path=None,
    )

    # infer dimensions from one sample
    x_img, gh_idx, x_aux, y = dataset[0]
    in_channels = x_img.shape[0]
    aux_dim = x_aux.shape[0]
    print("in_channels =", in_channels, "| aux_dim =", aux_dim, "| geohash_len =", gh_idx.numel())
    print("target_col  =", dataset.target_col)

    model = ResNet18FusionRegressor(
        in_channels=in_channels,
        aux_dim=aux_dim,
        geohash_emb_dim=16,
        hidden_dim=256,
        dropout=0.2,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    # supports either raw state_dict OR a dict with "model_state_dict"
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    print("Loaded checkpoint:", checkpoint_path)

    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    all_preds = []
    all_y = []

    with torch.no_grad():
        for x_img, gh_idx, x_aux, y in loader:
            x_img = x_img.to(device)
            gh_idx = gh_idx.to(device)
            x_aux = x_aux.to(device)

            preds = model(x_img, gh_idx, x_aux)

            all_preds.append(preds.cpu().numpy())
            all_y.append(y.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_y = np.concatenate(all_y, axis=0)

    df = dataset.df.copy()
    df["agbd_pred"] = all_preds

    # quick metrics
    mae = float(np.mean(np.abs(all_preds - all_y)))
    rmse = float(np.sqrt(np.mean((all_preds - all_y) ** 2)))
    print(f"MAE={mae:.4f}  RMSE={rmse:.4f}")

    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)

    # show actual vs pred
    cols = []
    for c in ["lat", "lon", "hls_granule", "row", "col"]:
        if c in df.columns:
            cols.append(c)
    cols += [dataset.target_col, "agbd_pred"]
    print(df[cols].head(10))
