# resnet18 model 

# change 
# the name of the csv file 
# model save location + name


#############changes made 
# updated koppen geiger class
# chnage koppen geiger call in getitem 
# create a temp folder for koppen tif and try to read it from there 

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import geopandas as gpd

import rasterio
from rasterio.windows import Window
from rasterio.warp import transform as rio_transform

import shutil

from pyproj import Transformer
from collections import Counter
import matplotlib.pyplot as plt
import torchvision.models as models

from rasterio.warp import reproject
from rasterio.enums import Resampling

#################################### edit these values
CSV_FILE_PATH = "/s/chopin/e/proj/hyperspec/masfiq/csv_files/gedi_California_top_half_2021_whole_year_metadata.csv"

NUM_EPOCHS=60
BATCH_SIZE=32
LEARNING_RATE=1e-3
# VAL_FRAC
# GEOHASH_PRECISION

MODEL_PATH = "/s/chopin/e/proj/hyperspec/masfiq/models/resnet18_fusion_geohash_month_koppen_California_top_half_2021.pth"


##################################### 
def load_koppen_legend(legend_path: str):
    code_to_label = {}

    print("\n[KOPPEN DEBUG] Loading legend from:", legend_path)
    print("[KOPPEN DEBUG] Legend exists:", Path(legend_path).exists())

    with open(legend_path, "r", encoding="utf-8") as f:
        for line in f:
            original_line = line.rstrip()
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

            label = rest_parts[0]   # Af, Am, Aw, BWh, etc.
            code_to_label[code] = label

    codes_sorted = sorted(code_to_label.keys())
    code_to_index = {c: i for i, c in enumerate(codes_sorted)}

    print("[KOPPEN DEBUG] Parsed legend class count:", len(codes_sorted))
    print("[KOPPEN DEBUG] First 20 parsed codes:", codes_sorted[:20])
    print("[KOPPEN DEBUG] First 20 code_to_label:", list(code_to_label.items())[:20])

    return code_to_label, codes_sorted, code_to_index


# changed koppen geiger class than version 2 

import os
import rasterio
from rasterio.errors import RasterioIOError
from pyproj import Transformer
import numpy as np

class KoppenGeigerLookup:
    def __init__(self, koppen_tif_path: str, legend_path: str, unknown_label="UNK"):
        self.koppen_tif_path = koppen_tif_path
        self.legend_path = legend_path

        self.code_to_label, self.codes_sorted, self.code_to_index = load_koppen_legend(legend_path)

        self.unknown_label = unknown_label
        self.unknown_index = len(self.codes_sorted)
        self.num_classes = len(self.codes_sorted) + 1

        # IMPORTANT: do NOT open raster here for multiprocessing
        self.src = None
        self.nodata = None
        self.to_raster = None
        self._pid = None

    def __getstate__(self):
        """Make this object picklable for DataLoader workers (drop open file handles)."""
        d = self.__dict__.copy()
        d["src"] = None
        d["to_raster"] = None
        d["_pid"] = None
        return d

    def _ensure_open(self):
        pid = os.getpid()
        if self.src is None or self._pid != pid:
            # reopen cleanly in this worker
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

    def sample_code(self, lon: float, lat: float) -> int:
        self._ensure_open()
        x, y = self.to_raster.transform(lon, lat)

        # Try once; if it fails, reopen + retry once
        for attempt in (1, 2):
            try:
                val = next(self.src.sample([(x, y)]))[0]
                if self.nodata is not None and val == self.nodata:
                    return -9999
                return int(val)
            except (RasterioIOError, StopIteration, ValueError):
                if attempt == 1:
                    # reopen and retry
                    self.close()
                    self._ensure_open()
                    continue
                return -9999

    def encode_onehot(self, lon: float, lat: float) -> tuple[np.ndarray, int]:
        code = self.sample_code(lon, lat)
        onehot = np.zeros((self.num_classes,), dtype=np.float32)

        idx = self.code_to_index.get(code, self.unknown_index)
        onehot[idx] = 1.0
        return onehot, idx


import math
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models

import rasterio
from rasterio.transform import xy as rio_xy

from pyproj import Transformer


# ----------------------------
# 0) Small utilities
# ----------------------------
BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
BASE32_MAP = {c: i for i, c in enumerate(BASE32)}

def geohash_encode(lat: float, lon: float, precision: int = 7) -> str:
    """
    Pure-python geohash encoder (no external dependency).
    Returns a base32 geohash string of length `precision`.
    """
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
    """
    Convert geohash string -> array of ints in [0,31], length = len(gh)
    """
    return np.array([BASE32_MAP.get(c, 0) for c in gh], dtype=np.int64)

def parse_hls_month_from_granule(granule_id: str) -> int:
    """
    Example granule_id:
      HLS.S30.T10TEK.2021057T190939.v2.0
    We parse YYYY + DOY (first 7 digits of the date token), then convert to month [1..12].
    """
    try:
        date_token = granule_id.split(".")[3]  # "2021057T190939"
        y = int(date_token[0:4])
        doy = int(date_token[4:7])
        dt = datetime(y, 1, 1) + timedelta(days=doy - 1)
        return dt.month
    except Exception:
        return 1  # safe fallback

def month_sincos(month: int) -> np.ndarray:
    """
    month: 1..12 -> [sin(2πm/12), cos(2πm/12)]
    """
    m = float(month)
    ang = 2.0 * math.pi * (m / 12.0)
    return np.array([math.sin(ang), math.cos(ang)], dtype=np.float32)


# ----------------------------
# 1) Dataset: image + (geohash, month, koppen)
# ----------------------------
class GEDIHlsPatchDatasetFusion(Dataset):
    """
    Reads your metadata CSV and patch GeoTIFFs.
    Returns:
      x_img   : (C,3,3) float32
      gh_idx  : (G,) int64   (geohash indices)
      x_aux   : (A,) float32 (month sin/cos + koppen one-hot)
      y       : ()   float32 target
    """
    # Fixed, global channel order (union of S30 + L30 + indices + NLCD)
    # Missing bands are filled with zeros.
    FIXED_BANDS = ["B02","B03","B04","B05","B06","B07","B8A","B11","B12","EVI","NDVI","NLCD"]

    def __init__(
        self,
        csv_path: str,
        geohash_precision: int = 7,
        target_col: str | None = None,
        koppen_tif_path=None, koppen_legend_path=None
    ):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)

        # target column: prefer agbd_mean (your current patch CSV), else agbd (older)
        if target_col is None:
            if "agbd_mean" in self.df.columns:
                target_col = "agbd_mean"
            elif "agbd_center" in self.df.columns:
                target_col = "agbd_center"
            else:
                raise ValueError("No target column found. Expected 'agbd_mean' or 'agbd' in CSV.")
        self.target_col = target_col

        self.geohash_precision = geohash_precision

        # Build a mapping for Köppen (categorical -> one-hot)
        self.koppen_lookup = None
        self.koppen_dim = 0

        if koppen_tif_path is not None and koppen_legend_path is not None:
            self.koppen_lookup = KoppenGeigerLookup(koppen_tif_path, koppen_legend_path)
            self.koppen_dim = self.koppen_lookup.num_classes

    def __len__(self):
        return len(self.df)

    def _load_patch_fixed_channels(self, patch_path: Path, bands_str: str | None) -> np.ndarray:
        """
        Load GeoTIFF (C,3,3) and map it into fixed channel order (len(FIXED_BANDS),3,3).
        """
        with rasterio.open(patch_path) as src:
            raw = src.read().astype(np.float32)  # (C,3,3)
            nodata = src.nodata

        if nodata is not None:
            raw[raw == nodata] = np.nan
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

        # Map by band names stored in CSV column "bands" (you write this already)
        out = np.zeros((len(self.FIXED_BANDS), 3, 3), dtype=np.float32)

        if isinstance(bands_str, str) and len(bands_str) > 0:
            used_bands = [b.strip() for b in bands_str.split(",")]
            # raw[i] corresponds to used_bands[i]
            for i, b in enumerate(used_bands):
                if i >= raw.shape[0]:
                    break
                if b in self.FIXED_BANDS:
                    j = self.FIXED_BANDS.index(b)
                    out[j] = raw[i]
        else:
            # If bands column missing, assume raw already matches FIXED_BANDS order as best effort
            C = min(raw.shape[0], out.shape[0])
            out[:C] = raw[:C]

        # Scaling: scale reflectance-like bands only (NOT NLCD)
        # This avoids NLCD categories (like 11, 21, 95) triggering divide-by-10000 on everything.
        for b in self.FIXED_BANDS:
            if b == "NLCD":
                continue
            j = self.FIXED_BANDS.index(b)
            mx = float(np.max(np.abs(out[j])))
            if mx > 2.0:  # looks like 0..10000 style
                out[j] = out[j] / 10000.0

        return out

    def _center_latlon_from_patch(self, patch_path: Path) -> tuple[float, float]:
        """
        Compute center lat/lon of the 3x3 patch using its GeoTransform + CRS.
        Center pixel is (row=1, col=1) in the 3x3.
        """
        with rasterio.open(patch_path) as src:
            crs = src.crs
            transform = src.transform

        # center of pixel (1,1)
        x, y = rio_xy(transform, 1, 1, offset="center")

        if crs is None:
            # fallback: return dummy, but model still runs
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
        x_img_np = self._load_patch_fixed_channels(patch_path, bands_str)  # (C,3,3)
        x_img = torch.from_numpy(x_img_np)  # float32

        # --- month encoding ---
        if "month" in self.df.columns:
            m = int(row["month"])
        elif "hls_granule" in self.df.columns:
            m = parse_hls_month_from_granule(str(row["hls_granule"]))
        else:
            m = 1
        m_sc = month_sincos(m)  # (2,)

        # --- geohash encoding (from patch center lat/lon) ---
        lat, lon = self._center_latlon_from_patch(patch_path)

        gh = geohash_encode(lat, lon, precision=self.geohash_precision)
        gh_idx = torch.from_numpy(geohash_to_indices(gh))  # (G,)

        # --- koppen encoding from raster (recommended) ---
        if self.koppen_lookup is not None:
            try:
                koppen_onehot, _ = self.koppen_lookup.encode_onehot(lon, lat)
            except Exception:
                # treat as unknown if anything goes wrong
                koppen_onehot = np.zeros((self.koppen_lookup.num_classes,), dtype=np.float32)
                koppen_onehot[self.koppen_lookup.unknown_index] = 1.0
        else:
            koppen_onehot = np.zeros((0,), dtype=np.float32)

        # --- build auxiliary vector: month sin/cos + koppen onehot ---
        x_aux_np = np.concatenate([m_sc, koppen_onehot], axis=0).astype(np.float32)
        x_aux = torch.from_numpy(x_aux_np)  # (A,)


        # --- target ---
        y_val = float(row[self.target_col])
        if not np.isfinite(y_val):
            raise ValueError(f"Non-finite target at idx={idx}: {y_val}")
        y = torch.tensor(y_val, dtype=torch.float32)

        return x_img, gh_idx.long(), x_aux, y


# ----------------------------
# 2) Model: ResNet18 + geohash embedding + concat + regressor
# ----------------------------
class ResNet18FusionRegressor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        geohash_precision: int,
        aux_dim: int,
        geohash_emb_dim: int = 16,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        # ResNet18 backbone (your small-input tweaks)
        m = models.resnet18(weights=None)

        m.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        # Make fc an Identity so forward() returns a 512-d feature vector
        m.fc = nn.Identity()

        self.backbone = m
        img_feat_dim = 512  # resnet18 output after avgpool/flatten

        # Geohash embedding (each character is a token in [0..31])
        self.gh_emb = nn.Embedding(32, geohash_emb_dim)

        fused_dim = img_feat_dim + geohash_emb_dim + aux_dim

        # Regressor (MLP head)
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
        """
        x_img: (B,C,3,3)
        gh_idx: (B,G)
        x_aux: (B,A)
        """
        img_feat = self.backbone(x_img)  # (B,512)

        # geohash embedding: (B,G,emb) -> mean over G -> (B,emb)
        gh_feat = self.gh_emb(gh_idx)           # (B,G,E)
        gh_feat = gh_feat.mean(dim=1)           # (B,E)

        fused = torch.cat([img_feat, gh_feat, x_aux], dim=1)  # (B, fused_dim)
        out = self.regressor(fused)  # (B,1)
        return out.squeeze(1)        # (B,)


# ----------------------------
# 3) Dataloaders + training
# ----------------------------
def create_dataloaders(csv_path, batch_size=32, val_frac=0.2, geohash_precision=7):
    # koppen_tif = "/s/chopin/e/proj/hyperspec/masfiq/dataset/koppen_geiger_tif/1991_2020/koppen_geiger_0p00833333.tif"
    # legend     = "/s/chopin/e/proj/hyperspec/masfiq/dataset/koppen_geiger_tif/legend.txt"
    koppen_tif = os.environ.get(
        "KOPPEN_TIF",
        "/s/chopin/e/proj/hyperspec/masfiq/dataset/koppen_geiger_tif/1991_2020/koppen_geiger_0p00833333.tif"
    )
    legend = "/s/chopin/e/proj/hyperspec/masfiq/dataset/koppen_geiger_tif/legend.txt"

    ds = GEDIHlsPatchDatasetFusion(
        csv_path,
        target_col="agbd_center",
        geohash_precision=7,
        koppen_tif_path=koppen_tif,
        koppen_legend_path=legend
    )

    N = len(ds)
    idx = np.arange(N)
    np.random.shuffle(idx)

    val_size = int(N * val_frac)
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2,pin_memory=True,
    persistent_workers=False,
    prefetch_factor=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2,pin_memory=True,persistent_workers=False, prefetch_factor=4)

    # Peek
    x_img, gh_idx, x_aux, y = ds[0]
    in_channels = x_img.shape[0]
    aux_dim = x_aux.shape[0]
    print("Sample x_img:", x_img.shape, "in_channels=", in_channels)
    print("Sample gh_idx:", gh_idx.shape, "geohash_precision=", gh_idx.numel())
    print("Sample x_aux:", x_aux.shape, "aux_dim=", aux_dim)
    print("Sample y:", y.item())

    return train_loader, val_loader, in_channels, aux_dim


def train_model(
    csv_path,
    num_epochs=60,
    batch_size=32,
    lr=1e-3,
    val_frac=0.2,
    geohash_precision=7,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, val_loader, in_channels, aux_dim = create_dataloaders(
        csv_path,
        batch_size=batch_size,
        val_frac=val_frac,
        geohash_precision=geohash_precision,
    )

    model = ResNet18FusionRegressor(
        in_channels=in_channels,
        geohash_precision=geohash_precision,
        aux_dim=aux_dim,
        geohash_emb_dim=16,
        hidden_dim=256,
        dropout=0.2,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def mae(pred, target):
        return torch.mean(torch.abs(pred - target))

    for epoch in range(1, num_epochs + 1):
        # ---- train ----
        model.train()
        tr_mse, tr_mae, n_tr = 0.0, 0.0, 0

        for x_img, gh_idx, x_aux, y in train_loader:
            x_img = x_img.to(device, non_blocking=True)
            gh_idx = gh_idx.to(device, non_blocking=True)
            x_aux = x_aux.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred = model(x_img, gh_idx, x_aux)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            bs = x_img.size(0)
            tr_mse += loss.item() * bs
            tr_mae += mae(pred, y).item() * bs
            n_tr += bs

        tr_mse /= max(n_tr, 1)
        tr_mae /= max(n_tr, 1)

        # ---- val ----
        model.eval()
        va_mse, va_mae, n_va = 0.0, 0.0, 0

        with torch.no_grad():
            for x_img, gh_idx, x_aux, y in val_loader:
                x_img = x_img.to(device)
                gh_idx = gh_idx.to(device)
                x_aux = x_aux.to(device)
                y = y.to(device)

                pred = model(x_img, gh_idx, x_aux)
                loss = criterion(pred, y)

                bs = x_img.size(0)
                va_mse += loss.item() * bs
                va_mae += mae(pred, y).item() * bs
                n_va += bs

        va_mse /= max(n_va, 1)
        va_mae /= max(n_va, 1)

        print(
            f"Epoch {epoch:03d} | "
            f"Train MSE {tr_mse:.4f} MAE {tr_mae:.4f} | "
            f"Val MSE {va_mse:.4f} MAE {va_mae:.4f}"
        )

    return model


if __name__ == "__main__":
    csv_path = CSV_FILE_PATH
    model = train_model(
        csv_path,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        val_frac=0.2,
        geohash_precision=7,
    )
    out_path = Path(MODEL_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved: {out_path}")
