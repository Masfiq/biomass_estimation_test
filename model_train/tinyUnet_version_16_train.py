# tinyUnet version 16

#------------------Bands 
#added c bands(vv and vh) again
# 11 spectral   B02 B03 B04 B05 B06 B07 B8A B11 B12 EVI NDVI vv vh 
# c band is going in as log(10)/3 instead of raw value 


#-------------------optimizer (was Adam on previous version)
# AdamW with weight decay 
# tinyUnet has used plain Adam with NO weight decay since version_1, while the resnet18

#------------------------aux feature 
# Month as sin/cos
# Köppen-Geiger climate class, one-hot
# Grid met feature - precip, max/min humidity, specific humidity, max/min temp
# Copernicus DEM: elevation, slope, aspect sin, aspect cos
# 17 nlcd (added as fraction of pixel, was previously one hot encode of the center pixel)

#--------------------- loss function (delta was 5 in huber loss in previous version)
# huber loss
# delta 2.5
# |e| ≤ 2.5   →  0.5 × e²             quadratic
# |e| > 2.5   →  2.5 × (|e| − 1.25)   linear
# Errors are in sqrt space, since the target is agbd_sqrt

#--------------------------extra
# added windsorization for the biomass value cap

#------------------------ROI
# california_north_10

#explanation
#in the note.md

#workflow 
#in the draw.io 




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
from rasterio.transform import xy as rio_xy, rowcol
from rasterio.transform import from_gcps
from rasterio.crs import CRS

from pyproj import Transformer

import re
import xarray as xr

################################## CHANGE VALUE HERE 
#CSV_PATH = "/s/chopin/e/proj/hyperspec/masfiq/csv_files/gedi_California_california_north_10_2021_whole_year_metadata.csv"
# CHANGE C: version_8 patches/CSV (NLCD one-hot, degrade_flag filter, agbd_se recorded)
CSV_PATH = "/s/chopin/e/proj/hyperspec/masfiq/csv_files/gedi_california_north_10_2021_AprilToAugust_version_8.csv"


NUM_EPOCHS=100
BATCH_SIZE=32
LEARNING_RATE=1e-4
GEOHASH_PRESITION=7

OUTPUT_PATH = Path("/s/chopin/e/proj/hyperspec/masfiq/models/tinyUnet_version_16_fusion_geohash_month_koppen_withAttentionLayer_SARlog_DEM_sqrt_NLCDfrac_delta2p5_winsor995_EOFk4_version_8_California_North_10_2021.pth")

# SAR is now baked into the patch tif by build_patches_version_6 (correct GCP
# georeferencing, <=3 day match to the coincident HLS/GEDI acquisition), so there is
# no separate live Sentinel-1 lookup at train time anymore.

GRIDMET_DIR = "/s/chopin/e/proj/hyperspec/masfiq/dataset/gridMET_weather_data"

# CHANGE A: Copernicus GLO-30 DEM. 12 one-degree tiles covering N41-N42 / W125-W120
# mosaicked into a single .vrt (no pixel duplication, GDAL reads the tiles through it).
DEM_PATH = "/s/chopin/e/proj/hyperspec/masfiq/dataset/copernicus_dem_30m/copernicus_dem_30m.vrt"

SEED = 42

# CHANGE L: winsorization cutoff, as a percentile of the TRAINING labels.
# Set to None to disable and reproduce version_14 exactly.
WINSOR_PCT = 99.5

# CHANGE M: number of EOF/PCA components kept per sensor. Set to None to disable and
# reproduce version_15 exactly. 4 holds 99.5% of the spectral variation; 3 holds 95%.
EOF_K = 4
# Fitted basis cache. Written by the first training run, READ by the eval script.
# Delete this file to force a refit (e.g. after changing EOF_K or the CSV).
EOF_BASIS_PATH = Path("/s/chopin/e/proj/hyperspec/masfiq/models/eof_basis_version_8_k4.npz")

#Set WINSOR_PCT = None to turn it off 




##################################

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

def parse_hls_date_from_granule(granule_id: str) -> np.datetime64:
    """
    Example: HLS.S30.T10TEK.2021057T190939.v2.0 -> year 2021, day-of-year 057
    """
    try:
        date_token = granule_id.split(".")[3]
        y = int(date_token[0:4])
        doy = int(date_token[4:7])
        dt = datetime(y, 1, 1) + timedelta(days=doy - 1)
        return np.datetime64(dt.date())
    except Exception:
        return np.datetime64("2021-01-01")


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


# CHANGE A -- new in version_8.
class DEMLookup:
    """
    Copernicus GLO-30 elevation, plus slope and aspect derived from a 3x3 window.

    Returns 4 auxiliary features per shot:
        [elev_km, slope_norm, aspect_sin, aspect_cos]

    Aspect is split into sin/cos because it is circular -- 0 deg and 360 deg are the
    same direction -- the same reason month is encoded as sin/cos in this pipeline.
    Feeding raw degrees would tell the network that north-facing (0) and north-facing
    (360) are maximally different.

    The DEM is EPSG:4326, so a pixel is ~0.0002778 deg square in ANGLE but not in
    METRES: ~30.9 m north-south everywhere, but only ~23 m east-west at this latitude
    (cos(41.75) ~= 0.746). Slope uses the true per-axis metre spacing; assuming square
    pixels would make east-west gradients come out ~34% too shallow.
    """

    M_PER_DEG_LAT = 111320.0

    def __init__(self, dem_path: str):
        self.dem_path = Path(dem_path)
        if not self.dem_path.exists():
            raise FileNotFoundError(f"DEM not found: {self.dem_path}")
        self._src = None
        self._pid = None

    def __getstate__(self):
        """Never pickle an open rasterio handle across a DataLoader worker fork."""
        d = self.__dict__.copy()
        d["_src"] = None
        d["_pid"] = None
        return d

    def _ensure_open(self):
        # Reopen per worker process; a handle inherited across fork is not safe.
        pid = os.getpid()
        if self._src is None or self._pid != pid:
            self._src = rasterio.open(self.dem_path)
            self._pid = pid

    def encode(self, lon: float, lat: float) -> np.ndarray:
        self._ensure_open()
        src = self._src

        row, col = src.index(lon, lat)
        row, col = int(row), int(col)

        # Need one pixel of margin on every side for the central difference.
        if not (1 <= row < src.height - 1 and 1 <= col < src.width - 1):
            return np.zeros(4, dtype=np.float32)

        z = src.read(1, window=((row - 1, row + 2), (col - 1, col + 2))).astype(np.float64)
        if z.shape != (3, 3) or not np.all(np.isfinite(z)):
            return np.zeros(4, dtype=np.float32)

        elev = float(z[1, 1])

        yres_m = abs(src.res[1]) * self.M_PER_DEG_LAT
        xres_m = abs(src.res[0]) * self.M_PER_DEG_LAT * math.cos(math.radians(lat))

        # Central differences. Row index grows southward, so the north-south
        # gradient is (north - south) = z[0,1] - z[2,1].
        dzdx = (z[1, 2] - z[1, 0]) / (2.0 * xres_m)
        dzdy = (z[0, 1] - z[2, 1]) / (2.0 * yres_m)

        slope_rad = math.atan(math.hypot(dzdx, dzdy))
        # Aspect = downhill direction, measured clockwise from north.
        aspect_rad = math.atan2(-dzdx, -dzdy)

        out = np.array(
            [
                elev / 1000.0,                   # km, ~0..2.5 over these shots
                math.degrees(slope_rad) / 90.0,  # 0..1
                math.sin(aspect_rad),
                math.cos(aspect_rad),            # +1 = north-facing
            ],
            dtype=np.float32,
        )
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


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


# CHANGE M ---------------------------------------------------------------------------
# Which 8 of the 11 spectral slots each sensor actually fills. Indices are into the
# post-NLCD-delete channel order, i.e. FIXED_BANDS[:11].
EOF_SENSOR_BANDS = {
    "L30": ["B02", "B03", "B04", "B05", "B06", "B07", "EVI", "NDVI"],
    "S30": ["B02", "B03", "B04", "B8A", "B11", "B12", "EVI", "NDVI"],
}


def eof_sensor_of(granule_id) -> str:
    """L30 vs S30 from the granule id, e.g. HLS.S30.T10TDL.2021213T190305.v2.0."""
    return "S30" if ".S30." in str(granule_id) else "L30"


def fit_eof_basis(df, spectral_names, train_idx, k, out_path,
                  max_patches_per_sensor=3000, seed=SEED):
    """Fit one PCA basis per sensor on TRAINING patches only, and cache it.

    Returns {sensor: {"idx", "mean", "std", "comp"}}.

    If out_path already exists it is loaded instead of refitted. That is what keeps the
    eval script consistent with training: a basis fitted on a different set of rows would
    load without error and silently change the meaning of every input channel.
    """
    if out_path.exists():
        z = np.load(out_path, allow_pickle=False)
        basis = {}
        for sensor in EOF_SENSOR_BANDS:
            basis[sensor] = {
                "idx":  z[f"{sensor}_idx"],
                "mean": z[f"{sensor}_mean"],
                "std":  z[f"{sensor}_std"],
                "comp": z[f"{sensor}_comp"],
            }
        print(f"[EOF] loaded cached basis k={z['k'][0]} from {out_path}", flush=True)
        return basis

    rng = np.random.default_rng(seed)
    sub = df.iloc[train_idx]
    basis = {}

    for sensor, names in EOF_SENSOR_BANDS.items():
        idx = np.array([spectral_names.index(b) for b in names], dtype=np.int64)
        rows = sub[sub["hls_granule"].map(eof_sensor_of) == sensor]
        if len(rows) == 0:
            raise RuntimeError(f"[EOF] no training rows for sensor {sensor}")
        take = rows.iloc[rng.permutation(len(rows))[:max_patches_per_sensor]]

        pix = []
        for _, r in take.iterrows():
            try:
                bl = [b.strip() for b in str(r["bands"]).split(",")]
                with rasterio.open(r["patch_tifs"]) as src:
                    a = src.read().astype(np.float32)
                    nd = src.nodata
                if nd is not None:
                    a[a == nd] = np.nan
                a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
                if not all(b in bl for b in names):
                    continue
                v = np.stack([a[bl.index(b)] for b in names], axis=0)   # (8,5,5)
                pix.append(v.reshape(len(names), -1).T)                 # (25,8)
            except Exception:
                continue
        if not pix:
            raise RuntimeError(f"[EOF] could not read any {sensor} training patches")
        X = np.concatenate(pix, axis=0).astype(np.float64)              # (n_px, 8)

        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std = np.where(std > 1e-12, std, 1.0)      # a constant band must not divide by 0
        Z = (X - mean) / std
        ev, evec = np.linalg.eigh(np.cov(Z.T))
        order = np.argsort(ev)[::-1]
        ev, evec = ev[order], evec[:, order]
        comp = evec[:, :k]                          # (8, k)
        var = float(ev[:k].sum() / ev.sum())

        basis[sensor] = {"idx": idx, "mean": mean.astype(np.float32),
                         "std": std.astype(np.float32), "comp": comp.astype(np.float32)}
        print(f"[EOF] {sensor}: fitted on {len(X):,} pixels from {len(pix):,} patches, "
              f"k={k} holds {var:.1%} of the variation", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, k=np.array([k]),
             **{f"{s}_{key}": val for s, d in basis.items() for key, val in d.items()})
    print(f"[EOF] wrote basis to {out_path}", flush=True)
    return basis


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
    # CHANGE C: the single "NLCD" entry is replaced by 17 one-hot planes. The raw
    # "NLCD" code band still exists in the patch tif and is intentionally omitted
    # here, so _load_patch_fixed_channels reads past it. 11 + 17 + 2 = 30 channels.
    # CHANGE G: the 17 one-hot PLANES are gone. The raw "NLCD" code band is loaded
    # LAST purely so __getitem__ can read its centre pixel, and is then deleted from
    # the image tensor -- the network sees 11 spectral channels.
    # Must stay last: NLCD_RAW_IDX below indexes into this list.
    # CHANGE I: S1_VV / S1_VH restored, placed BEFORE "NLCD" so the raw code band stays
    # last. The network sees 11 spectral + 2 SAR = 13 channels.
    FIXED_BANDS = [
        "B02","B03","B04","B05","B06","B07","B8A","B11","B12","EVI","NDVI",
        "S1_VV","S1_VH",
        "NLCD",
    ]
    NLCD_RAW_IDX = 13

    # CHANGE G: fixed CONUS legend, identical to build_patches_version_8. Hardcoded,
    # never inferred, so the southern test ROI cannot change aux_dim.
    NLCD_CLASSES = (11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95)
    NLCD_AUX_DIM = len(NLCD_CLASSES) + 1      # + NLCD_OTHER catch-all
    # CHANGE K: dict lookup instead of tuple.index(), because version_13 resolves a
    # class per PIXEL (25 per sample) rather than once per sample.
    NLCD_CLASS_TO_IDX = {c: i for i, c in enumerate(NLCD_CLASSES)}

    def __init__(
        self,
        csv_path: str,
        geohash_precision: int = 7,
        target_col: str | None = None,
        koppen_tif_path=None, koppen_legend_path=None,
        gridmet_dir=None,
        dem_path=None,          # CHANGE A -- new in version_8
        eof_basis=None,         # CHANGE M -- {sensor: {idx, mean, std, comp}} or None
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

        self.gridmet_lookup = None
        self.gridmet_dim = 0
        if gridmet_dir is not None:
            print("Gridmet dir added \n")
            # Restrict gridMET to this CSV's footprint so each worker holds a few MB
            # instead of progressively caching the whole CONUS grid (see GridMETLookup).
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

        # CHANGE A -- new in version_8. Terrain aux features from the Copernicus DEM.
        # CHANGE M: set by create_dataloaders after fitting/loading. None reproduces v15.
        self.eof_basis = eof_basis
        # channel order AFTER the NLCD delete is FIXED_BANDS minus the NLCD entry, so the
        # 11 spectral slots are FIXED_BANDS[:11] and SAR follows at 11, 12.
        self.spectral_names = [b for b in self.FIXED_BANDS
                               if not b.startswith("NLCD") and not b.startswith("S1_")]
        self.sar_names = [b for b in self.FIXED_BANDS if b.startswith("S1_")]

        self.dem_lookup = None
        self.dem_dim = 0
        if dem_path is not None:
            print("DEM path added\n", flush=True)
            self.dem_lookup = DEMLookup(dem_path)
            self.dem_dim = 4        # elev, slope, aspect_sin, aspect_cos

    def __len__(self):
        return len(self.df)

    def _load_patch_fixed_channels(self, patch_path: Path, bands_str: str | None) -> np.ndarray:
        """
        Load GeoTIFF (C,3,3) and map it into fixed channel order (len(FIXED_BANDS),3,3).
        """
        with rasterio.open(patch_path) as src:
            raw = src.read().astype(np.float32)  # (C,3,3)
            nodata = src.nodata
            crs = src.crs
            transform = src.transform

        if nodata is not None:
            raw[raw == nodata] = np.nan
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

        # Map by band names stored in CSV column "bands" (you write this already)
        out = np.zeros((len(self.FIXED_BANDS), raw.shape[1], raw.shape[2]), dtype=np.float32)

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

        # CHANGE J: explicit per-source normalisation, replacing the old
        # "/10000 if max > 2.0" heuristic.
        # Optical (B02..B12, EVI, NDVI): nothing to do. hls_download_version_2.py applies
        #   scale_factor=0.0001 at download time, so these are already 0-1 on disk. The
        #   old rule therefore never fired once -- checked across 4,800 band-patches.
        # NLCD: not a channel at all in version_11+. The raw code band is read for the aux
        #   fractions and then deleted, so there is nothing here to scale.
        # SAR: raw GRD digital numbers (~30..900), which is why every version up to 13 fed
        #   radar at ~400x the magnitude of the optical bands.
        S1_SCALE = 3.0
        for b in ("S1_VV", "S1_VH"):
            if b in self.FIXED_BANDS:
                j = self.FIXED_BANDS.index(b)
                v = out[j]
                # np.where evaluates BOTH branches before choosing, so log10 runs on the
                # zeros too; the 1e-6 floor keeps that discarded result finite instead of
                # -inf plus a runtime warning. Zeros mark swath-edge gaps and must stay 0,
                # or "no radar here" becomes a large negative the network reads as a real
                # measurement.
                out[j] = np.where(v > 0, np.log10(np.maximum(v, 1e-6)) / S1_SCALE, 0.0)

        return out, crs, transform

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
        x_img_np, crs, transform = self._load_patch_fixed_channels(patch_path, bands_str)

        # --- CHANGE K: NLCD class FRACTIONS over the whole 5x5, then drop the band ---
        # Count how many of the 25 pixels are each class and divide by 25. Nearest-
        # neighbour resampling in build_patches means the stored values are already exact
        # class codes; rint() is defensive only.
        nlcd_codes = np.rint(x_img_np[self.NLCD_RAW_IDX]).astype(np.int32).ravel()   # 25
        nlcd_frac = np.zeros((self.NLCD_AUX_DIM,), dtype=np.float32)
        w = np.float32(1.0 / nlcd_codes.size)
        for code in nlcd_codes:
            # NLCD_CLASS_TO_IDX is a dict, so this is a hash lookup per pixel, not a
            # linear .index() scan -- 25 lookups per sample, 153k samples per epoch.
            nlcd_frac[self.NLCD_CLASS_TO_IDX.get(int(code), self.NLCD_AUX_DIM - 1)] += w
        # delete the raw code channel so the network never sees a bare 11..95 magnitude
        x_img_np = np.delete(x_img_np, self.NLCD_RAW_IDX, axis=0)

        # --- CHANGE M: project the 11 spectral channels onto this sensor's EOF basis ---
        # Channels are now [11 spectral, S1_VV, S1_VH]. Only the spectral block is
        # replaced; SAR passes through untouched, already log10/3 scaled above.
        if self.eof_basis is not None:
            sensor = eof_sensor_of(row["hls_granule"]) if "hls_granule" in self.df.columns else "L30"
            b = self.eof_basis[sensor]
            n_spec = len(self.spectral_names)
            spec = x_img_np[b["idx"], :, :]                       # (8, 5, 5)
            flat = spec.reshape(spec.shape[0], -1).T              # (25, 8)
            z = (flat - b["mean"]) / b["std"]
            proj = (z @ b["comp"]).T                              # (k, 25)
            eof_ch = proj.reshape(-1, spec.shape[1], spec.shape[2]).astype(np.float32)
            sar_ch = x_img_np[n_spec:, :, :]                      # (2, 5, 5)
            x_img_np = np.concatenate([eof_ch, sar_ch], axis=0)   # (k+2, 5, 5)

        x_img = torch.from_numpy(np.ascontiguousarray(x_img_np))

        # center pixel — dynamic for any patch size (3x3 → index 1,1 / 5x5 → index 2,2)
        ph, pw = x_img_np.shape[1], x_img_np.shape[2]
        x, y = rio_xy(transform, ph // 2, pw // 2, offset="center")
        if crs is None:
            lat, lon = 0.0, 0.0
        else:
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(x, y)

        # --- month encoding ---
        if "month" in self.df.columns:
            m = int(row["month"])
        elif "hls_granule" in self.df.columns:
            m = parse_hls_month_from_granule(str(row["hls_granule"]))
        else:
            m = 1
        m_sc = month_sincos(m)  # (2,)

      

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

        # --- date for gridMET and Sentinel-1 ---
        if "hls_granule" in self.df.columns:
            date_value = parse_hls_date_from_granule(str(row["hls_granule"]))
        elif "date" in self.df.columns:
            date_value = np.datetime64(pd.to_datetime(row["date"]).date())
        else:
            date_value = np.datetime64("2021-01-01")

        # --- gridMET weather encoding ---
        if self.gridmet_lookup is not None:
            try:
                gridmet_features = self.gridmet_lookup.encode(lon, lat, date_value)
            except Exception:
                gridmet_features = np.zeros((6,), dtype=np.float32)
        else:
            gridmet_features = np.zeros((0,), dtype=np.float32)

        # Sentinel-1 VV/VH are no longer fetched live here — build_patches_version_6
        # bakes them into the patch tif directly, so _load_patch_fixed_channels already
        # filled x_img_np[vv_idx]/[vh_idx] from the "bands" CSV column above.

        # --- CHANGE A: DEM terrain encoding (new in version_8) ---
        if self.dem_lookup is not None:
            try:
                dem_features = self.dem_lookup.encode(lon, lat)
            except Exception:
                dem_features = np.zeros((4,), dtype=np.float32)
        else:
            dem_features = np.zeros((0,), dtype=np.float32)

        # --- build auxiliary vector: month sin/cos + koppen onehot + gridMET + DEM ---
        # CHANGE K: nlcd_frac (was nlcd_onehot) appended LAST. Same 17 slots, same
        # position, so aux_dim is unchanged at 60. eval must use the same order.
        x_aux_np = np.concatenate(
            [m_sc, koppen_onehot, gridmet_features, dem_features, nlcd_frac],
            axis=0
        ).astype(np.float32)
        x_aux = torch.from_numpy(x_aux_np)  # (A,)


        # --- target ---
        y_val = float(row[self.target_col])
        if not np.isfinite(y_val):
            raise ValueError(f"Non-finite target at idx={idx}: {y_val}")
        y = torch.tensor(y_val, dtype=torch.float32)

        return x_img, gh_idx.long(), x_aux, y
    


class ChannelSE(nn.Module):
    """Channel/band attention (Squeeze-and-Excitation)."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),      # (B,C,1,1)
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.net(x)                  # (B,C,1,1)
        return x * w


class SpatialAttention(nn.Module):
    """CBAM-style spatial attention. For 3×3, kernel_size=3 is enough."""
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        p = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=p, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)          # (B,1,H,W)
        max_map, _ = torch.max(x, dim=1, keepdim=True)        # (B,1,H,W)
        attn = self.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * attn


class FusionGate(nn.Module):
    """Per-sample modality gating over (img_feat, gh_feat, x_aux)."""
    def __init__(self, img_dim: int, gh_dim: int, aux_dim: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(img_dim + gh_dim + aux_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3)  # weights for (img, geohash, aux)
        )

    def forward(self, img_feat, gh_feat, x_aux):
        z = torch.cat([img_feat, gh_feat, x_aux], dim=1)
        w = torch.softmax(self.mlp(z), dim=1)                 # (B,3)
        img_feat = img_feat * w[:, 0:1]
        gh_feat  = gh_feat  * w[:, 1:2]
        x_aux    = x_aux    * w[:, 2:3]
        return img_feat, gh_feat, x_aux, w


# ----------------------------
# 2) Model: ResNet18 + geohash embedding + concat + regressor
# ----------------------------
class ResNet18FusionRegressorAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        geohash_precision: int,
        aux_dim: int,
        geohash_emb_dim: int = 16,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        use_channel_attn: bool = True,
        use_spatial_attn: bool = True,
        use_fusion_attn: bool = True,
    ):
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

        img_feat = self.backbone(x_img)                 # (B,512)
        gh_feat = self.gh_emb(gh_idx).mean(dim=1)       # (B,E)

        if self.use_fusion_attn:
            img_feat, gh_feat, x_aux, _ = self.fusion_gate(img_feat, gh_feat, x_aux)

        fused = torch.cat([img_feat, gh_feat, x_aux], dim=1)
        return self.regressor(fused).squeeze(1)
    



class TinyUNetFusionRegressor(nn.Module):
    """
    Tiny U-Net-style regression model for 3x3 patches.
    Input:
      x_img:  (B, C, 3, 3)
      gh_idx: (B, G)
      x_aux:  (B, A)
    Output:
      biomass prediction: (B,)
    """

    def __init__(
        self,
        in_channels: int,
        geohash_precision: int,
        aux_dim: int,
        geohash_emb_dim: int = 16,
        base_channels: int = 32,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Geohash branch
        self.gh_emb = nn.Embedding(32, geohash_emb_dim)

        # Encoder block 1
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, base_channels),
            nn.ReLU(inplace=True),
        )

        # Encoder block 2
        # No maxpool because the patch is only 3x3.
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.GroupNorm(4, base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.GroupNorm(4, base_channels * 2),
            nn.ReLU(inplace=True),
        )

        # Decoder-like block with skip connection
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 2 + base_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, base_channels),
            nn.ReLU(inplace=True),
        )

        # Convert final feature map to vector
        self.pool = nn.AdaptiveAvgPool2d(1)

        img_feat_dim = base_channels
        fused_dim = img_feat_dim + geohash_emb_dim + aux_dim

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
        e1 = self.enc1(x_img)              # (B, base, 3, 3)
        e2 = self.enc2(e1)                 # (B, base*2, 3, 3)

        # U-Net-style skip connection
        d1 = torch.cat([e2, e1], dim=1)    # (B, base*3, 3, 3)
        d1 = self.dec1(d1)                 # (B, base, 3, 3)

        img_feat = self.pool(d1).flatten(1)  # (B, base)

        gh_feat = self.gh_emb(gh_idx).mean(dim=1)  # (B, geohash_emb_dim)

        fused = torch.cat([img_feat, gh_feat, x_aux], dim=1)
        out = self.regressor(fused)

        return out.squeeze(1)
    

class TinyUNetFusionRegressorAttn(nn.Module):
    """
    Tiny U-Net-style regression model with:
      1. Channel attention on input bands
      2. Spatial attention on 3x3 patch
      3. Fusion attention over image/geohash/aux features

    Input:
      x_img:  (B, C, 3, 3)
      gh_idx: (B, G)
      x_aux:  (B, A)

    Output:
      biomass prediction: (B,)
    """

    def __init__(
        self,
        in_channels: int,
        geohash_precision: int,
        aux_dim: int,
        geohash_emb_dim: int = 16,
        base_channels: int = 32,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        use_channel_attn: bool = True,
        use_spatial_attn: bool = True,
        use_fusion_attn: bool = True,
    ):
        super().__init__()

        self.use_channel_attn = use_channel_attn
        self.use_spatial_attn = use_spatial_attn
        self.use_fusion_attn = use_fusion_attn

        # Attention before Tiny U-Net
        if use_channel_attn:
            self.channel_attn = ChannelSE(in_channels, reduction=4)

        if use_spatial_attn:
            self.spatial_attn = SpatialAttention(kernel_size=3)

        # Geohash branch
        self.gh_emb = nn.Embedding(32, geohash_emb_dim)

        # Encoder block 1
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, base_channels),
            nn.ReLU(inplace=True),
        )

        # Encoder block 2
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.GroupNorm(4, base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.GroupNorm(4, base_channels * 2),
            nn.ReLU(inplace=True),
        )

        # Decoder-like block with skip connection
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 2 + base_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, base_channels),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        img_feat_dim = base_channels

        # Fusion gate over image, geohash, and auxiliary features
        if use_fusion_attn:
            self.fusion_gate = FusionGate(
                img_dim=img_feat_dim,
                gh_dim=geohash_emb_dim,
                aux_dim=aux_dim,
                hidden=128
            )

        fused_dim = img_feat_dim + geohash_emb_dim + aux_dim

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
        pred, _, _, _ = self.forward_with_attn(x_img, gh_idx, x_aux)
        return pred

    def forward_with_attn(self, x_img, gh_idx, x_aux):
        w_chan = None
        w_spat = None
        w_fuse = None

        # Channel attention
        if self.use_channel_attn:
            w_chan = self.channel_attn.net(x_img)   # (B, C, 1, 1)
            x_img = x_img * w_chan

        # Spatial attention
        if self.use_spatial_attn:
            avg_map = torch.mean(x_img, dim=1, keepdim=True)
            max_map, _ = torch.max(x_img, dim=1, keepdim=True)
            w_spat = self.spatial_attn.sigmoid(
                self.spatial_attn.conv(torch.cat([avg_map, max_map], dim=1))
            )                                      # (B, 1, 3, 3)
            x_img = x_img * w_spat

        # Tiny U-Net feature extraction
        e1 = self.enc1(x_img)                       # (B, base, 3, 3)
        e2 = self.enc2(e1)                          # (B, base*2, 3, 3)

        d1 = torch.cat([e2, e1], dim=1)             # (B, base*3, 3, 3)
        d1 = self.dec1(d1)                          # (B, base, 3, 3)

        img_feat = self.pool(d1).flatten(1)         # (B, base)

        # Geohash feature
        gh_feat = self.gh_emb(gh_idx).mean(dim=1)   # (B, geohash_emb_dim)

        # Fusion attention
        if self.use_fusion_attn:
            img_feat, gh_feat, x_aux, w_fuse = self.fusion_gate(
                img_feat, gh_feat, x_aux
            )

        fused = torch.cat([img_feat, gh_feat, x_aux], dim=1)
        pred = self.regressor(fused).squeeze(1)

        return pred, w_chan, w_spat, w_fuse


# ----------------------------
# 3) Dataloaders + training
# ----------------------------
def _cgroup_paths():
    """Locate this process's memory cgroup (v2 first, then v1)."""
    import os
    try:
        with open("/proc/self/cgroup") as fh:
            lines = fh.read().strip().splitlines()
    except Exception:
        return None
    # cgroup v2: single line "0::/path"
    for ln in lines:
        parts = ln.split(":")
        if len(parts) == 3 and parts[1] == "":
            p = "/sys/fs/cgroup" + parts[2]
            if os.path.exists(os.path.join(p, "memory.max")):
                return ("v2", p)
    # cgroup v1
    for ln in lines:
        parts = ln.split(":")
        if len(parts) == 3 and "memory" in parts[1]:
            p = "/sys/fs/cgroup/memory" + parts[2]
            if os.path.exists(os.path.join(p, "memory.limit_in_bytes")):
                return ("v1", p)
    return None


def _cg_read(path, name):
    try:
        with open(f"{path}/{name}") as fh:
            v = fh.read().strip()
        return None if v == "max" else int(v)
    except Exception:
        return None


_CG = _cgroup_paths()


def _mem_report(tag):
    """Report cgroup usage -- the exact accounting the OOM killer uses."""
    import os
    ver, path = (_CG if _CG else (None, None))
    if path is None:
        print(f"[MEM] {tag}: (no cgroup visible)", flush=True)
        return
    if ver == "v2":
        cur = _cg_read(path, "memory.current")
        lim = _cg_read(path, "memory.max")
        peak = _cg_read(path, "memory.peak")
        extra = ""
        try:
            with open(f"{path}/memory.stat") as fh:
                st = dict(l.split() for l in fh.read().split("\n") if " " in l)
            extra = (f" anon={int(st.get('anon',0))/1e9:.1f}G"
                     f" file={int(st.get('file',0))/1e9:.1f}G"
                     f" slab={int(st.get('slab',0))/1e9:.1f}G")
        except Exception:
            pass
    else:
        cur = _cg_read(path, "memory.usage_in_bytes")
        lim = _cg_read(path, "memory.limit_in_bytes")
        peak = _cg_read(path, "memory.max_usage_in_bytes")
        extra = ""
        try:
            with open(f"{path}/memory.stat") as fh:
                st = dict(l.split() for l in fh.read().split("\n") if " " in l)
            extra = (f" rss={int(st.get('total_rss',0))/1e9:.1f}G"
                     f" cache={int(st.get('total_cache',0))/1e9:.1f}G")
        except Exception:
            pass
    f = lambda v: "n/a" if v is None else f"{v/1e9:.1f}G"
    print(f"[MEM] {tag}: use={f(cur)} peak={f(peak)} limit={f(lim)}{extra}", flush=True)


def create_dataloaders(csv_path, batch_size=32, val_frac=0.2, geohash_precision=7, seed=SEED):
    # koppen_tif = "/s/chopin/e/proj/hyperspec/masfiq/dataset/koppen_geiger_tif/1991_2020/koppen_geiger_0p00833333.tif"
    # legend     = "/s/chopin/e/proj/hyperspec/masfiq/dataset/koppen_geiger_tif/legend.txt"
    koppen_tif = os.environ.get(
        "KOPPEN_TIF",
        "/s/chopin/e/proj/hyperspec/masfiq/dataset/koppen_geiger_tif/1991_2020/koppen_geiger_0p00833333.tif"
    )
    legend = "/s/chopin/e/proj/hyperspec/masfiq/dataset/koppen_geiger_tif/legend.txt"

    # CHANGE M: build the dataset once WITHOUT the basis so its band bookkeeping is
    # available, fit/load on training rows, then attach. Fitting needs train_idx, which
    # needs len(ds), so the order is: construct -> split -> fit -> attach.
    ds = GEDIHlsPatchDatasetFusion(
        csv_path,
        target_col="agbd_sqrt",     # CHANGE B -- was "agbd_log" in version_7
        geohash_precision=7,
        koppen_tif_path=koppen_tif,
        koppen_legend_path=legend,
        gridmet_dir=GRIDMET_DIR,
        dem_path=DEM_PATH,          # CHANGE A -- new in version_8
    )

    N = len(ds)
    # Seeded split so the train/val partition is reproducible AND identical to
    # resnet18_version_12, which lets the three architectures be compared on the
    # same validation data instead of each drawing its own random split.
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)

    val_size = int(N * val_frac)
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]

    # CHANGE M: fit (or load) the EOF basis on TRAINING rows only, then attach it to the
    # dataset so both Subsets see the same projection.
    if EOF_K is not None:
        ds.eof_basis = fit_eof_basis(ds.df, ds.spectral_names, train_idx,
                                     EOF_K, EOF_BASIS_PATH)
        print(f"[EOF] k={EOF_K}: {len(ds.spectral_names)} spectral channels -> {EOF_K}, "
              f"plus {len(ds.sar_names)} SAR = {EOF_K + len(ds.sar_names)} in_channels",
              flush=True)

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    # CHANGE L: cutoff from the TRAINING rows only. Computing it over the full CSV would
    # let validation shots influence a number the training loop uses -- a small leak, but
    # free to avoid. target_col is agbd_sqrt, so the cap is already in sqrt space and is
    # applied directly to y with no back-transform.
    winsor_cap = None
    if WINSOR_PCT is not None:
        tgt = ds.df[ds.target_col].to_numpy()[train_idx]
        tgt = tgt[np.isfinite(tgt)]
        winsor_cap = float(np.percentile(tgt, WINSOR_PCT))
        n_clipped = int((tgt > winsor_cap).sum())
        print(f"[WINSOR] p{WINSOR_PCT} of {ds.target_col} over {len(tgt):,} training rows "
              f"= {winsor_cap:.4f} (= {winsor_cap**2:.1f} Mg/ha); "
              f"{n_clipped:,} training shots clipped ({n_clipped/len(tgt):.2%})", flush=True)
        print("[WINSOR] validation labels are NOT clipped", flush=True)

    # Worker count is the multiplier on every per-process cost: shared-memory
    # segments for tensor handoff, GDAL block caches, and copy-on-write pages
    # faulted out of the parent. Overridable via DL_WORKERS without editing code.
    num_workers = int(os.environ.get("DL_WORKERS", "4"))
    num_workers = max(1, num_workers)

    # persistent_workers=False is deliberate. With it True, train_loader's workers
    # AND val_loader's workers are all held alive at once (2 x num_workers procs),
    # and nothing they accumulate -- GDAL block caches, COW-copied pages from the
    # parent, allocator fragmentation -- is ever returned to the OS. Peak arrives at
    # the epoch-1 -> epoch-2 boundary, which is exactly where this job was OOM-killed.
    # Tearing workers down after each pass bounds the footprint; the cost is a few
    # seconds of respawn per phase.
    val_workers = max(2, num_workers // 2)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True,
                          persistent_workers=False, prefetch_factor=2)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=val_workers, pin_memory=True,
                          persistent_workers=False, prefetch_factor=2)
    print(f"[LOADER] train workers={num_workers} val workers={val_workers} "
          f"persistent=False prefetch=2", flush=True)


    # Peek
    x_img, gh_idx, x_aux, y = ds[0]
    in_channels = x_img.shape[0]
    aux_dim = x_aux.shape[0]
    print("Sample x_img:", x_img.shape, "in_channels=", in_channels)
    print("Sample gh_idx:", gh_idx.shape, "geohash_precision=", gh_idx.numel())
    print("Sample x_aux:", x_aux.shape, "aux_dim=", aux_dim)
    print("Sample y:", y.item())

    return train_loader, val_loader, in_channels, aux_dim, winsor_cap


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
    _mem_report("startup")

    train_loader, val_loader, in_channels, aux_dim, winsor_cap = create_dataloaders(
        csv_path,
        batch_size=batch_size,
        val_frac=val_frac,
        geohash_precision=geohash_precision,
    )

    model = TinyUNetFusionRegressorAttn(
    in_channels=in_channels,
    geohash_precision=geohash_precision,
    aux_dim=aux_dim,
    geohash_emb_dim=16,
    base_channels=32,
    hidden_dim=256,
    dropout=0.2,
    use_channel_attn=True,
    use_spatial_attn=True,
    use_fusion_attn=True,
    ).to(device)    

    # >>> ADD THIS HERE (fine-tune init) <<<
    # init_ckpt = "/s/chopin/e/proj/hyperspec/masfiq/models/field_boundary_biomass_resnet18withNLCD_features_cnn_8band_3x3.pth"
    # ckpt = torch.load(init_ckpt, map_location=device)
    # state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    # model.load_state_dict(state, strict=False)  # strict=False because attention params are new

    # CHANGE B knock-on: delta scaled 1.0 -> 5.0 with the target range (log1p spans
    # 0.27..7.0, sqrt spans 0.56..33.2, ~4.75x). Leaving it at 1.0 would push nearly
    # every error past the quadratic/linear knee and silently turn Huber into MAE.
    # CHANGE H: 5.0 -> 2.5. At 5.0 ~94% of errors sat in the quadratic half, so the
    # robust branch never engaged. Typical error is ~2.8 sqrt-units.
    criterion = nn.HuberLoss(delta=2.5)
    # CHANGE E: was plain Adam with no weight decay. Matches resnet18_version_15.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    def mae(pred, target):
        return torch.mean(torch.abs(pred - target))
    
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Best-validation checkpointing: the final epoch is often not the best one once
    # the LR scheduler has plateaued, so save a separate snapshot every time
    # validation Huber improves, instead of only saving whatever epoch is last.
    best_val_mse = float("inf")
    best_epoch = 0
    best_out_path = OUTPUT_PATH.with_name(OUTPUT_PATH.stem + "_best" + OUTPUT_PATH.suffix)

    for epoch in range(1, num_epochs + 1):
        # ---- train ----
        model.train()
        tr_mse, tr_mae, n_tr = 0.0, 0.0, 0

        # CHANGE F: per-batch [MEM] probes removed. Startup and end-of-epoch remain.
        for x_img, gh_idx, x_aux, y in train_loader:
            x_img = x_img.to(device, non_blocking=True)
            gh_idx = gh_idx.to(device, non_blocking=True)
            x_aux  = x_aux.to(device, non_blocking=True)
            y      = y.to(device, non_blocking=True)

            # CHANGE L: clip the TRAINING targets only. Deliberately absent from the
            # validation pass below, so val metrics stay measured against real labels.
            if winsor_cap is not None:
                y = torch.clamp(y, max=winsor_cap)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    pred = model(x_img, gh_idx, x_aux)
                    loss = criterion(pred, y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(x_img, gh_idx, x_aux)
                loss = criterion(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                x_img = x_img.to(device, non_blocking=True)
                gh_idx = gh_idx.to(device, non_blocking=True)
                x_aux = x_aux.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                pred = model(x_img, gh_idx, x_aux)
                loss = criterion(pred, y)

                bs = x_img.size(0)
                va_mse += loss.item() * bs
                va_mae += mae(pred, y).item() * bs
                n_va += bs

        va_mse /= max(n_va, 1)
        va_mae /= max(n_va, 1)

        _mem_report(f"end epoch {epoch:03d}")

        scheduler.step(va_mse)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d} | "
            f"Train Huber {tr_mse:.4f} MAE {tr_mae:.4f} | "
            f"Val Huber {va_mse:.4f} MAE {va_mae:.4f} | "
            f"LR {current_lr:.2e}"
        )

        if va_mse < best_val_mse:
            best_val_mse = va_mse
            best_epoch = epoch
            best_out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_out_path)
            print(f"  New best model (epoch {epoch:03d}, Val Huber {va_mse:.4f}), saved to {best_out_path}")

    print(f"Best epoch: {best_epoch:03d} (Val Huber {best_val_mse:.4f}) -> {best_out_path}")

    return model


if __name__ == "__main__":
    csv_path = CSV_PATH
    model = train_model(
        csv_path,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        val_frac=0.2,
        geohash_precision=GEOHASH_PRESITION,
    )
    out_path = OUTPUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved: {out_path}")

