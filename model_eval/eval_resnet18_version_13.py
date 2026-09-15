# eval for resnet18_version_13 — 5x5 patches, version_6 CSV, Huber-trained
# path variables are in the bash .sh file
# version_6 CSV/patches bake SAR directly into the patch tif (GCP georeferencing,
# <=3 day match), so the separate Sentinel1Lookup/SENTINEL1_DIR mechanism is removed:
# S1_VV/S1_VH now arrive through the same "bands" CSV column mapping as every other band.

import re
from rasterio.windows import Window
from rasterio.transform import xy as rio_xy, rowcol, from_gcps
from rasterio.crs import CRS

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
from rasterio.errors import RasterioIOError
from pyproj import Transformer
from collections import defaultdict

import xarray as xr

BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
BASE32_MAP = {c: i for i, c in enumerate(BASE32)}

######################### edit here

fixed_bands = [
    "B02","B03","B04",
    "B05","B06","B07",       # L30 (Landsat) only — zero for S30 patches
    "B8A","B11","B12",       # S30 (Sentinel-2) only — zero for L30 patches
    "EVI","NDVI","NLCD",
    "S1_VV","S1_VH",
]

GRIDMET_DIR = "/s/chopin/e/proj/hyperspec/masfiq/dataset/gridMET_weather_data"

# SAR is baked into the patch tif by build_patches_version_6, so there is no separate
# live Sentinel-1 lookup at eval time anymore.

#########################

def eval_by_landcover(model, dataset, device, batch_size=256):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

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

            pred, _, _, _ = model.forward_with_attn(x_img, gh_idx, x_aux)

            nlcd_center = nlcd_class.detach().cpu().numpy()

            y_np = np.expm1(y.detach().cpu().numpy())
            p_np = np.expm1(pred.detach().cpu().numpy())

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

    overall_mae = float(np.mean(np.abs(p_all - y_all)))
    overall_rmse = float(np.sqrt(np.mean((p_all - y_all) ** 2)))

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
    per_class_spat_sum = defaultdict(lambda: np.zeros((5, 5), dtype=np.float64))
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
            y_np = np.expm1(y.detach().cpu().numpy())
            p_np = np.expm1(pred.detach().cpu().numpy())
            err_np = p_np - y_np

            wc = w_chan.detach().cpu().numpy().reshape(bsz, len(bands))
            ws = w_spat.detach().cpu().numpy().reshape(bsz, 5, 5)
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

        chan_mean = per_class_chan_sum[cls] / n
        for band, value in zip(bands, chan_mean):
            row[f"chan_attn_{band}"] = float(value)

        spat_mean = per_class_spat_sum[cls] / n
        for r in range(5):
            for c in range(5):
                row[f"spat_attn_r{r}_c{c}"] = float(spat_mean[r, c])

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
        date_token = granule_id.split(".")[3]
        y = int(date_token[0:4])
        doy = int(date_token[4:7])
        dt = datetime(y, 1, 1) + timedelta(days=doy - 1)
        return dt.month
    except Exception:
        return 1


def parse_hls_date_from_granule(granule_id: str) -> np.datetime64:
    try:
        date_token = granule_id.split(".")[3]
        y = int(date_token[0:4])
        doy = int(date_token[4:7])
        dt = datetime(y, 1, 1) + timedelta(days=doy - 1)
        return np.datetime64(dt.date())
    except Exception:
        return np.datetime64("2021-01-01")

def month_sincos(month: int) -> np.ndarray:
    ang = 2.0 * math.pi * (float(month) / 12.0)
    return np.array([math.sin(ang), math.cos(ang)], dtype=np.float32)

def load_koppen_legend(legend_path: str):
    code_to_label = {}

    print("\n[KOPPEN DEBUG] Loading legend from:", legend_path)
    print("[KOPPEN DEBUG] Legend exists:", Path(legend_path).exists())

    with open(legend_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

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


class GridMETLookup:
    """
    Loads a small spatial subset of the gridMET rasters ONCE into memory.

    Why a subset: the raw gridMET files are CONUS-wide (365 x 585 x 1386) and are
    HDF5-chunked at (61, 98, 231) ~= 5.5 MB per chunk. A single-point .sel() pulls
    a whole chunk into the netCDF chunk cache, so random access across the training
    set slowly caches the entire grid (~7 GB) in EVERY DataLoader worker. With
    persistent train + val workers that grows past the cgroup limit and gets
    OOM-killed at the epoch boundary.

    The study area spans only a fraction of CONUS, so we slice to its bounding box
    (plus padding), pull it eagerly into a numpy cube, and close the files. Lookups
    are then pure numpy: no file handles, no chunk cache, no per-sample disk I/O.
    """

    VAR_ORDER = ["pr", "rmax", "rmin", "sph", "tmmn", "tmmx"]

    def __init__(self, gridmet_dir: str,
                 lat_min=None, lat_max=None,
                 lon_min=None, lon_max=None,
                 pad: float = 0.5):
        self.gridmet_dir = Path(gridmet_dir)

        self.var_files = {
            "pr": self.gridmet_dir / "pr_2021.nc",
            "rmax": self.gridmet_dir / "rmax_2021.nc",
            "rmin": self.gridmet_dir / "rmin_2021.nc",
            "sph": self.gridmet_dir / "sph_2021.nc",
            "tmmn": self.gridmet_dir / "tmmn_2021.nc",
            "tmmx": self.gridmet_dir / "tmmx_2021.nc",
        }

        self.lats = None
        self.lons = None
        self.days = None
        self.cube = None  # (6, nday, nlat, nlon) float32

        self._load_subset(lat_min, lat_max, lon_min, lon_max, pad)

    def _coord_name(self, ds, candidates):
        for c in candidates:
            if c in ds.coords:
                return c
            if c in ds.dims:
                return c
        raise ValueError(f"Could not find coordinate among {candidates}")

    def _data_var_name(self, ds, preferred_name):
        if preferred_name in ds.data_vars:
            return preferred_name
        return list(ds.data_vars.keys())[0]

    def _load_subset(self, lat_min, lat_max, lon_min, lon_max, pad):
        planes = []

        for vi, name in enumerate(self.VAR_ORDER):
            path = self.var_files[name]
            if not path.exists():
                raise FileNotFoundError(f"Missing gridMET file: {path}")

            with xr.open_dataset(path) as ds:
                var_name = self._data_var_name(ds, name)
                lat_name = self._coord_name(ds, ["lat", "latitude", "y"])
                lon_name = self._coord_name(ds, ["lon", "longitude", "x"])
                time_name = self._coord_name(ds, ["day", "time", "date"])

                lats_full = ds[lat_name].values.astype(np.float64)
                lons_full = ds[lon_name].values.astype(np.float64)

                # Resolve the bounding box to index slices. argmin/argmax style
                # masking handles either ascending or descending coordinates.
                if lat_min is None or lat_max is None:
                    lat_sl = slice(None)
                else:
                    m = np.where((lats_full >= lat_min - pad) &
                                 (lats_full <= lat_max + pad))[0]
                    if m.size == 0:
                        raise ValueError(
                            f"gridMET lat range {lats_full.min()}..{lats_full.max()} "
                            f"does not cover requested {lat_min}..{lat_max}"
                        )
                    lat_sl = slice(int(m.min()), int(m.max()) + 1)

                if lon_min is None or lon_max is None:
                    lon_sl = slice(None)
                else:
                    m = np.where((lons_full >= lon_min - pad) &
                                 (lons_full <= lon_max + pad))[0]
                    if m.size == 0:
                        raise ValueError(
                            f"gridMET lon range {lons_full.min()}..{lons_full.max()} "
                            f"does not cover requested {lon_min}..{lon_max}"
                        )
                    lon_sl = slice(int(m.min()), int(m.max()) + 1)

                sub = ds[var_name].isel({lat_name: lat_sl, lon_name: lon_sl})
                arr = np.asarray(sub.values, dtype=np.float32)

                if vi == 0:
                    self.lats = lats_full[lat_sl]
                    self.lons = lons_full[lon_sl]
                    self.days = ds[time_name].values.astype("datetime64[D]")

                planes.append(arr)

        self.cube = np.stack(planes, axis=0).astype(np.float32)

        print(
            f"[GRIDMET] loaded subset cube {self.cube.shape} "
            f"({self.cube.nbytes / 1e6:.1f} MB) "
            f"lat {self.lats.min():.3f}..{self.lats.max():.3f} "
            f"lon {self.lons.min():.3f}..{self.lons.max():.3f}",
            flush=True,
        )

    def sample_raw(self, lon: float, lat: float, date_value: np.datetime64) -> np.ndarray:
        i_lat = int(np.abs(self.lats - lat).argmin())
        i_lon = int(np.abs(self.lons - lon).argmin())

        d = np.datetime64(date_value, "D")
        i_day = int(
            np.abs((self.days - d).astype("timedelta64[D]").astype(np.int64)).argmin()
        )

        return self.cube[:, i_day, i_lat, i_lon].astype(np.float32)

    def normalize(self, raw: np.ndarray) -> np.ndarray:
        """
        Order: [pr, rmax, rmin, sph, tmmn, tmmx]
        tmmn/tmmx are usually Kelvin in gridMET, so convert to Celsius.
        """
        pr, rmax, rmin, sph, tmmn, tmmx = raw

        pr_norm = np.log1p(max(pr, 0.0)) / 5.0
        rmax_norm = rmax / 100.0
        rmin_norm = rmin / 100.0
        sph_norm = sph * 1000.0
        tmmn_norm = (tmmn - 273.15) / 40.0
        tmmx_norm = (tmmx - 273.15) / 40.0

        out = np.array(
            [pr_norm, rmax_norm, rmin_norm, sph_norm, tmmn_norm, tmmx_norm],
            dtype=np.float32,
        )

        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    def encode(self, lon: float, lat: float, date_value: np.datetime64) -> np.ndarray:
        raw = self.sample_raw(lon, lat, date_value)
        return self.normalize(raw)



# 0) Small utilities

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


def parse_hls_date_from_granule(granule_id: str) -> np.datetime64:
    """
    Example:
    HLS.S30.T10TEK.2021057T190939.v2.0

    This extracts:
    year = 2021
    day of year = 057
    """
    try:
        date_token = granule_id.split(".")[3]
        y = int(date_token[0:4])
        doy = int(date_token[4:7])
        dt = datetime(y, 1, 1) + timedelta(days=doy - 1)
        return np.datetime64(dt.date())
    except Exception:
        return np.datetime64("2021-01-01")

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



# 1) Dataset: image + (geohash, month, koppen)


class GEDIHlsPatchDatasetFusion(Dataset):
    FIXED_BANDS = fixed_bands

    def __init__(self, csv_path: str, geohash_precision: int = 7, target_col: str = "agbd_center",
             koppen_tif_path=None, koppen_legend_path=None, gridmet_dir=None):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.geohash_precision = geohash_precision
        self.target_col = target_col

        self.koppen_lookup = None
        if koppen_tif_path and koppen_legend_path:
            self.koppen_lookup = KoppenGeigerLookup(koppen_tif_path, koppen_legend_path)

        self.gridmet_lookup = None
        self.gridmet_dim = 0

        if gridmet_dir is not None:
            if "lat" in self.df.columns and "lon" in self.df.columns:
                gm_bounds = dict(
                    lat_min=float(self.df["lat"].min()),
                    lat_max=float(self.df["lat"].max()),
                    lon_min=float(self.df["lon"].min()),
                    lon_max=float(self.df["lon"].max()),
                )
            else:
                gm_bounds = {}
            self.gridmet_lookup = GridMETLookup(gridmet_dir, **gm_bounds)
            self.gridmet_dim = 6

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

        out = np.zeros((len(self.FIXED_BANDS), raw.shape[1], raw.shape[2]), dtype=np.float32)
        nlcd_patch = np.zeros((raw.shape[1], raw.shape[2]), dtype=np.float32)

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
            if b in ("NLCD", "S1_VV", "S1_VH"):
                continue
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

        if not patch_path.exists():
            raise FileNotFoundError(f"Patch file not found: {patch_path}")

        bands_str = row["bands"] if "bands" in self.df.columns else None
        x_img_np, nlcd_patch, crs, transform = self._load_patch_fixed_channels(
            patch_path,
            bands_str
        )

        # dynamic center pixel for any patch size
        ph, pw = x_img_np.shape[1], x_img_np.shape[2]
        x, y = rio_xy(transform, ph // 2, pw // 2, offset="center")

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

        if "hls_granule" in self.df.columns:
            date_value = parse_hls_date_from_granule(str(row["hls_granule"]))
        elif "date" in self.df.columns:
            date_value = np.datetime64(pd.to_datetime(row["date"]).date())
        else:
            date_value = np.datetime64("2021-01-01")

        # Sentinel-1 VV/VH are no longer fetched live here — build_patches_version_6
        # bakes them into the patch tif directly, so _load_patch_fixed_channels already
        # filled x_img_np[vv_idx]/[vh_idx] from the "bands" CSV column above.

        x_img = torch.from_numpy(x_img_np)

        gh = geohash_encode(lat, lon, precision=self.geohash_precision)
        gh_idx = torch.from_numpy(geohash_to_indices(gh))

        if self.koppen_lookup is not None:
            try:
                koppen_onehot = self.koppen_lookup.encode_onehot(lon, lat)
            except Exception:
                koppen_onehot = np.zeros((self.koppen_lookup.num_classes,), dtype=np.float32)
                koppen_onehot[self.koppen_lookup.unknown_index] = 1.0
        else:
            koppen_onehot = np.zeros((0,), dtype=np.float32)

        if self.gridmet_lookup is not None:
            try:
                gridmet_features = self.gridmet_lookup.encode(lon, lat, date_value)
            except Exception:
                gridmet_features = np.zeros((6,), dtype=np.float32)
        else:
            gridmet_features = np.zeros((0,), dtype=np.float32)

        x_aux_np = np.concatenate(
            [m_sc, koppen_onehot, gridmet_features],
            axis=0
        ).astype(np.float32)

        x_aux = torch.from_numpy(x_aux_np)

        y_val = float(row[self.target_col])
        if not np.isfinite(y_val):
            raise ValueError(f"Non-finite target at idx={idx}: {y_val}")

        y = torch.tensor(y_val, dtype=torch.float32)

        nlcd_class = torch.tensor(
            int(round(float(nlcd_patch[ph // 2, pw // 2]))),
            dtype=torch.long
        )

        return x_img, gh_idx.long(), x_aux, y, nlcd_class


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
        w = self.net(x)
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
        w = self.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
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
        w = torch.softmax(self.mlp(z), dim=1)
        return img_feat * w[:, 0:1], gh_feat * w[:, 1:2], x_aux * w[:, 2:3], w


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
        if self.use_channel_attn:
            x_img = self.channel_attn(x_img)
        if self.use_spatial_attn:
            x_img = self.spatial_attn(x_img)

        img_feat = self.backbone(x_img)
        gh_feat = self.gh_emb(gh_idx).mean(dim=1)

        if self.use_fusion_attn:
            img_feat, gh_feat, x_aux, _ = self.fusion_gate(img_feat, gh_feat, x_aux)

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


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/s/chopin/e/proj/hyperspec/masfiq/csv_files/gedi_california_north_10_2021_AprilToAugust_version_6.csv")
    ap.add_argument("--ckpt", default="/s/chopin/e/proj/hyperspec/masfiq/models/resnet18_version_13_fusion_geohash_month_koppen_withAttentionLayer_SAR_version_6_California_North_10_2021.pth")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--geohash-precision", type=int, default=7)
    ap.add_argument("--out", default="resnet18_version_13_eval_version_6.json")
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
        target_col="agbd_log",
        koppen_tif_path=koppen_tif,
        koppen_legend_path=legend,
        gridmet_dir=GRIDMET_DIR,
    )

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

    x_img0, gh0, x_aux0, y0, nlcd0 = ds[0]
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
    spat_sum = np.zeros((5, 5), dtype=np.float64)
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
                wc = w_chan.detach().cpu().numpy().reshape(bsz, len(bands))
                chan_sum += wc.sum(axis=0)

            if w_spat is not None:
                ws = w_spat.detach().cpu().numpy().reshape(bsz, 5, 5)
                spat_sum += ws.sum(axis=0)

            if w_fuse is not None:
                wf = w_fuse.detach().cpu().numpy()
                fuse_sum += wf.sum(axis=0)

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    # convert log space → Mg/ha
    y_true_all = np.expm1(y_true_all)
    y_pred_all = np.expm1(y_pred_all)

    mae = float(np.mean(np.abs(y_pred_all - y_true_all)))
    rmse = float(np.sqrt(np.mean((y_pred_all - y_true_all) ** 2)))
    r2 = r2_score(y_true_all, y_pred_all)

    chan_mean = (chan_sum / max(1, n_seen)).tolist()
    spat_mean = (spat_sum / max(1, n_seen)).tolist()
    fuse_mean = (fuse_sum / max(1, n_seen)).tolist()

    top5 = sorted(zip(bands, chan_mean), key=lambda x: x[1], reverse=True)[:5]

    print("\n=== VAL METRICS (Mg/ha) ===")
    print(f"MAE  = {mae:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"R^2  = {r2:.4f}")

    print("\n=== TOP-5 CHANNEL ATTENTION (mean weight) ===")
    for b, w in top5:
        print(f"{b:>4s} : {w:.4f}")

    print("\n=== MEAN SPATIAL ATTENTION MAP (5x5) ===")
    for r in range(5):
        print(" ".join([f"{spat_mean[r][c]:.3f}" for c in range(5)]))

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
        "spatial_attention_mean_5x5": spat_mean,
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

    loader_lc = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    per_class_err = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for x_img, gh_idx, x_aux, y, nlcd_class in loader_lc:
            x_img = x_img.to(device)
            gh_idx = gh_idx.to(device)
            x_aux = x_aux.to(device)
            y = y.to(device)

            pred, _, _, _ = model.forward_with_attn(x_img, gh_idx, x_aux)

            nlcd_center = nlcd_class.detach().cpu().numpy()
            y_np = np.expm1(y.detach().cpu().numpy())
            p_np = np.expm1(pred.detach().cpu().numpy())
            err_np = p_np - y_np

            for k in range(len(y_np)):
                cls = int(round(float(nlcd_center[k])))
                per_class_err[cls].append(float(err_np[k]))

    bias_map = {}
    for cls, errs in per_class_err.items():
        bias_map[cls] = float(np.mean(errs))

    lc_df["bias"] = lc_df["nlcd_class"].map(bias_map)
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

    lc_attn_df = add_nlcd_names(lc_attn_df)

    drop_cols = ["n", "mae", "rmse", "bias", "nlcd_name"]
    drop_cols = [c for c in drop_cols if c in lc_attn_df.columns]

    lc_attn_only = lc_attn_df.drop(columns=drop_cols)

    lc_combined_df = lc_df.merge(
        lc_attn_only,
        on="nlcd_class",
        how="left"
    )

    lc_combined_df = lc_combined_df.sort_values("n", ascending=False).reset_index(drop=True)

    print("\nCombined land-cover metrics + attention interpretation:")
    print(lc_combined_df.to_string(index=False))

    lc_out_csv = Path(args.out).with_name(Path(args.out).stem + "_landcover_metrics.csv")
    lc_combined_df.to_csv(lc_out_csv, index=False)

    print("\nSaved combined land-cover metrics and attention interpretation to:", lc_out_csv)


if __name__ == "__main__":
    main()
