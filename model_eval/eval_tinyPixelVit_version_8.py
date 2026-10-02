# eval for tinyPixelViT_version_8 — 5x5 patches, version_10 CSV, Huber-trained
# Built from eval_tinyUnet_version_17.py (already verified to reproduce the v17 training
# data and validation split exactly), with only the model swapped. tinyPixelVit_version_8
# was itself built from tinyUnet_version_17_train.py with only the model swapped, so the
# data pipeline here matches its training run:
#   * 13 image channels: 11 spectral + S1_VV + S1_VH (raw NLCD loaded last, then deleted)
#   * SAR log10(x)/3, NLCD aux = patch class FRACTIONS
#   * clean-patch filter hls_finite_px >= 25 applied BEFORE the split (same as training)
#   * winsorisation and EOF were OFF in training; neither appears here
#
# ---- what differs from the tinyUnet evals ---------------------------------------------
# TinyPixelViT has NO channel / spatial / fusion attention, so the attention reports
# (top-5 channel attention, 5x5 spatial map, fusion weights, per-class attention table)
# are removed rather than printed as zeros. Metrics and land-cover tables are unchanged.
#
# ---- professor's land-cover sensitivity breakdown -------------------------------------
#   (1) Planted pasture/hay 81  (2) Cropland 82  (3) Deciduous forest 41
#   (4) Evergreen forest 42     (5) Mixed forest 43
#   (6) Forest = 41+42+43       (7) Agriculture = 81+82
# Grouped by the CENTRE-pixel NLCD class. Saved as <out>_professor_landcover.csv.
# -------------------------------------------------------------------------------------

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

import re
import rasterio
from rasterio.windows import Window
from rasterio.transform import xy as rio_xy, rowcol, from_gcps
from rasterio.crs import CRS
import xarray as xr
from rasterio.errors import RasterioIOError
from pyproj import Transformer


# Must equal HLS_MIN_FINITE_PX_TRAIN in tinyUnet_version_17_train.py.
HLS_MIN_FINITE_PX_TRAIN = 25

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


# SAR is baked into the patch tif by build_patches_version_6, so there is no separate
# live Sentinel-1 lookup at eval time anymore.
GRIDMET_DIR = "/s/chopin/e/proj/hyperspec/masfiq/dataset/gridMET_weather_data"

# CHANGE A: Copernicus GLO-30 DEM mosaic (18 one-degree tiles, N40-N42 / W125-W120).
# Must be the same file the version_8 training run read.
DEM_PATH = "/s/chopin/e/proj/hyperspec/masfiq/dataset/copernicus_dem_30m/copernicus_dem_30m.vrt"


# CHANGE B: back-transform helper. Replaces every inverse-log call in version_7.
def sqrt_to_linear(a: np.ndarray) -> np.ndarray:
    """sqrt space -> Mg/ha. Clamp at 0 first so a negative prediction does not
    square into a spuriously large positive biomass."""
    return np.clip(a, 0.0, None) ** 2


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


# CHANGE A -- new in version_8. Copied verbatim from tinyUnet_version_8_train.py.
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


# ----------------------------
# 1) Dataset: image + (geohash, month, koppen)
# ----------------------------


class GEDIHlsPatchDatasetFusion(Dataset):
    # CHANGE C: must match tinyUnet_version_9_train.py exactly. 11 + 17 + 2 = 30.
    # CHANGE G: 17 one-hot planes gone; raw "NLCD" loaded last and then deleted.
    # Must stay last -- NLCD_RAW_IDX indexes into this list.
    # CHANGE I: S1_VV / S1_VH restored, BEFORE "NLCD" so the raw code band stays last.
    # 11 spectral + 2 SAR = 13 channels after the NLCD delete.
    FIXED_BANDS = [
        "B02","B03","B04","B05","B06","B07","B8A","B11","B12","EVI","NDVI",
        "S1_VV","S1_VH",
        "NLCD",
    ]
    NLCD_RAW_IDX = 13

    # CHANGE G: identical to tinyUnet_version_11_train.py and build_patches_version_8.
    NLCD_CLASSES = (11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95)
    NLCD_AUX_DIM = len(NLCD_CLASSES) + 1      # + NLCD_OTHER catch-all
    # CHANGE K: dict lookup, because version_13 resolves a class per PIXEL.
    NLCD_CLASS_TO_IDX = {c: i for i, c in enumerate(NLCD_CLASSES)}

    def __init__(
        self,
        csv_path: str,
        geohash_precision: int = 7,
        target_col: str = "agbd_center",
        koppen_tif_path=None,
        koppen_legend_path=None,
        gridmet_dir=None,
        dem_path=None,          # CHANGE A -- new in version_8
    ):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        # Same filter, same place as tinyUnet_version_17_train.py, so the seeded split
        # reproduces the training run's validation set exactly.
        if HLS_MIN_FINITE_PX_TRAIN is not None and "hls_finite_px" in self.df.columns:
            n0 = len(self.df)
            self.df = self.df[self.df["hls_finite_px"] >= HLS_MIN_FINITE_PX_TRAIN].reset_index(drop=True)
            print(f"[HLS FILTER] kept {len(self.df):,} of {n0:,} rows with "
                  f"hls_finite_px >= {HLS_MIN_FINITE_PX_TRAIN} "
                  f"({n0 - len(self.df):,} dropped)", flush=True)
        self.geohash_precision = geohash_precision
        self.target_col = target_col

        self.koppen_lookup = None

        if koppen_tif_path is not None and koppen_legend_path is not None:
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

        # CHANGE A -- new in version_8. Terrain aux features from the Copernicus DEM.
        self.dem_lookup = None
        self.dem_dim = 0
        if dem_path is not None:
            # Constructor raises if the VRT is missing rather than silently zero-filling:
            # a checkpoint trained with 42 aux inputs cannot be scored with 38.
            self.dem_lookup = DEMLookup(dem_path)
            self.dem_dim = 4        # elev, slope, aspect_sin, aspect_cos

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

        # CHANGE J: per-source normalisation, identical to tinyUnet_version_14/15_train.
        # Optical is already 0-1 from download time. SAR is raw GRD digital numbers
        # (~30..900) -> log10(x)/3. Feeding RAW SAR to this log-trained model would load
        # without error and score nonsense (the reverse mismatch measured R2 0.41 -> 0.18).
        S1_SCALE = 3.0
        for b in ("S1_VV", "S1_VH"):
            if b in self.FIXED_BANDS:
                j = self.FIXED_BANDS.index(b)
                v = out[j]
                out[j] = np.where(v > 0, np.log10(np.maximum(v, 1e-6)) / S1_SCALE, 0.0)

        return out, crs, transform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        patch_path = Path(row["patch_tifs"])

        if not patch_path.is_absolute():
            patch_path = self.csv_path.parent / patch_path

        bands_str = row["bands"] if "bands" in self.df.columns else None
        x_img_np, crs, transform = self._load_patch_fixed_channels(patch_path, bands_str)

        # --- CHANGE K: NLCD class FRACTIONS over the whole 5x5, then drop the band ---
        # Count how many of the 25 pixels are each class, divide by 25. MUST match
        # tinyUnet_version_13_train.py exactly.
        nlcd_codes = np.rint(x_img_np[self.NLCD_RAW_IDX]).astype(np.int32).ravel()
        nlcd_frac = np.zeros((self.NLCD_AUX_DIM,), dtype=np.float32)
        w = np.float32(1.0 / nlcd_codes.size)
        for code in nlcd_codes:
            nlcd_frac[self.NLCD_CLASS_TO_IDX.get(int(code), self.NLCD_AUX_DIM - 1)] += w
        x_img_np = np.delete(x_img_np, self.NLCD_RAW_IDX, axis=0)

        x_img = torch.from_numpy(np.ascontiguousarray(x_img_np))

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

        gh = geohash_encode(lat, lon, precision=self.geohash_precision)
        gh_idx = torch.from_numpy(geohash_to_indices(gh))

        if self.koppen_lookup is not None:
            koppen_onehot = self.koppen_lookup.encode_onehot(lon, lat)
        else:
            koppen_onehot = np.zeros((0,), dtype=np.float32)

        if "hls_granule" in self.df.columns:
            date_value = parse_hls_date_from_granule(str(row["hls_granule"]))
        elif "date" in self.df.columns:
            date_value = np.datetime64(pd.to_datetime(row["date"]).date())
        else:
            date_value = np.datetime64("2021-01-01")

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

        # CHANGE A: dem_features appended last, matching the training-time order exactly.
        # CHANGE G: nlcd_onehot appended LAST -- same order as the training script.
        x_aux_np = np.concatenate(
            [m_sc, koppen_onehot, gridmet_features, dem_features, nlcd_frac], axis=0
        ).astype(np.float32)
        x_aux = torch.from_numpy(x_aux_np)

        y_val = float(row[self.target_col])

        if not np.isfinite(y_val):
            raise ValueError(f"Non-finite target at index {idx}: {y_val}")

        y = torch.tensor(y_val, dtype=torch.float32)

        return x_img, gh_idx.long(), x_aux, y


class ChannelSE(nn.Module):
    # Channel/band attention (Squeeze-and-Excitation)
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



#  Model:

class TinyPixelViTFusionRegressor(nn.Module):
    """
    Tiny ViT-style regression model for 3x3 multispectral patches.

    Idea:
      - Treat each pixel as a token.
      - 3x3 patch gives 9 tokens.
      - Each token contains all spectral/index bands.
      - Fuse image-token feature with geohash and auxiliary features.
    """

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
        self.num_tokens = 25  # 5x5 pixels

        # Each pixel token has C band values.
        self.token_proj = nn.Linear(in_channels, embed_dim)

        # CLS token for final image representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embedding for CLS + 9 pixel tokens
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
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(embed_dim)

        # Geohash branch
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
        """
        x_img:  (B, C, 3, 3)
        gh_idx: (B, G)
        x_aux:  (B, A)
        """

        B, C, H, W = x_img.shape

        # Convert from (B, C, 3, 3) to (B, 9, C)
        x = x_img.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Project each pixel token to embedding space
        x = self.token_proj(x)  # (B, 9, embed_dim)

        # Add CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls, x], dim=1)          # (B, 10, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer encoder
        x = self.transformer(x)

        # Use CLS token as image feature
        img_feat = self.norm(x[:, 0, :])        # (B, embed_dim)

        # Geohash feature
        gh_feat = self.gh_emb(gh_idx).mean(dim=1)

        # Fusion
        fused = torch.cat([img_feat, gh_feat, x_aux], dim=1)

        out = self.regressor(fused)
        return out.squeeze(1)


    




# Dataloaders + training

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


def nlcd_code_from_aux(x_aux, base_ds):
    """Recover each sample's NLCD class code from the AUX vector.

    version_11 deletes the raw NLCD channel from the image tensor (its centre pixel is
    copied into aux first), so the class can no longer be read from x_img -- doing so
    raises IndexError, or silently reads a spectral band if the index happens to fall in
    range. The trailing NLCD_AUX_DIM entries of x_aux hold the encoding, and argmax
    recovers the class: exact for version_11/12's one-hot, and the MAJORITY class for
    version_13's fractions, which is the right grouping either way.

    The final slot is the NLCD_OTHER catch-all (nodata, or a class outside the CONUS
    legend); it maps to -1 so it is reported as its own group rather than merged.
    """
    blk = x_aux[:, -base_ds.NLCD_AUX_DIM:].detach().cpu().numpy()
    codes = np.array(list(base_ds.NLCD_CLASSES) + [-1], dtype=np.int64)
    return codes[blk.argmax(axis=1)]


PROFESSOR_CATEGORIES = [
    ("(1) Planted pasture/hay [81]", {81}),
    ("(2) Cropland [82]",            {82}),
    ("(3) Deciduous forest [41]",    {41}),
    ("(4) Evergreen forest [42]",    {42}),
    ("(5) Mixed forest [43]",        {43}),
    ("(6) Forest [41+42+43]",        {41, 42, 43}),
    ("(7) Agriculture [81+82]",      {81, 82}),
]


def center_nlcd_codes(df, rows):
    """NLCD class of the CENTRE pixel [2,2] for each row, read from the patch tif.

    The centre class is not in the aux vector (version_13+ stores patch FRACTIONS), so it
    is read directly: one 1x1 window read per validation patch.
    """
    out = np.full(len(rows), -1, dtype=np.int64)
    for k, ridx in enumerate(rows):
        r = df.iloc[int(ridx)]
        try:
            bl = [b.strip() for b in str(r["bands"]).split(",")]
            with rasterio.open(r["patch_tifs"]) as src:
                v = src.read(bl.index("NLCD") + 1, window=((2, 3), (2, 3)))[0, 0]
            if np.isfinite(v):
                out[k] = int(np.rint(v))
        except Exception:
            pass
    return out


def professor_landcover_table(y_true, y_pred, center_codes, nlcd_frac, class_to_idx):
    """Metrics per requested category. y_* in Mg/ha, row-aligned with center_codes."""
    rows = []
    for name, codes in PROFESSOR_CATEGORIES:
        m = np.isin(center_codes, list(codes))
        n = int(m.sum())
        row = {"category": name,
               "nlcd_codes": "+".join(str(c) for c in sorted(codes)),
               "n": n}
        if n == 0:
            rows.append(row)
            continue
        yt, yp = y_true[m], y_pred[m]
        err = yp - yt
        mean_t = float(yt.mean())
        rmse = float(np.sqrt(np.mean(err ** 2)))
        ss_tot = float(((yt - mean_t) ** 2).sum())
        # patch purity: share of the 5x5 window made up of this category's classes
        cols = [class_to_idx[c] for c in codes if c in class_to_idx]
        purity = float(nlcd_frac[m][:, cols].sum(axis=1).mean()) if cols else float("nan")
        row.update({
            "mean_true_agbd": mean_t,
            "mean_pred_agbd": float(yp.mean()),
            "mae": float(np.mean(np.abs(err))),
            "rmse": rmse,
            "nrmse": rmse / mean_t if mean_t > 0 else float("nan"),
            "bias": float(err.mean()),
            "r2": (1.0 - float((err ** 2).sum()) / ss_tot) if (n >= 2 and ss_tot > 0) else float("nan"),
            "mean_patch_purity": purity,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def eval_by_landcover(model, dataset, device, batch_size=256):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    base_ds = dataset.dataset if hasattr(dataset, "dataset") else dataset

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

            # BUGFIX: was x_img[:, nlcd_idx, 2, 2], which raises IndexError in
            # version_11+ because NLCD is no longer an image channel.
            nlcd_center = nlcd_code_from_aux(x_aux, base_ds)

            # CHANGE B: sqrt space -> Mg/ha (was the inverse-log call in version_7)
            y_np = sqrt_to_linear(y.detach().cpu().numpy())
            p_np = sqrt_to_linear(pred.detach().cpu().numpy())

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


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", default="/s/chopin/e/proj/hyperspec/masfiq/csv_files/gedi_california_north_10_2021_AprilToAugust_version_6.csv")
    ap.add_argument("--ckpt", default="/s/chopin/e/proj/hyperspec/masfiq/models/tinyUnet_version_7_fusion_geohash_month_koppen_withAttentionLayer_SAR_version_6_California_North_10_2021_best.pth")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--geohash-precision", type=int, default=7)
    ap.add_argument("--out", default="tinyUnet_version_7_eval_version_6.json")

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
        target_col="agbd_sqrt",     # CHANGE B -- was "agbd_log" in version_7
        koppen_tif_path=koppen_tif,
        koppen_legend_path=legend,
        gridmet_dir=GRIDMET_DIR,
        dem_path=DEM_PATH,          # CHANGE A -- new in version_8
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

    # BUGFIX: drop the raw "NLCD" entry -- see the note above. len(bands) must
    # equal the model's in_channels, not len(FIXED_BANDS).
    bands = [b for i, b in enumerate(GEDIHlsPatchDatasetFusion.FIXED_BANDS)
             if i != GEDIHlsPatchDatasetFusion.NLCD_RAW_IDX]

    y_true_all = []
    y_pred_all = []
    nlcd_frac_all = []      # NEW: aux NLCD fractions, for patch purity per category

    n_seen = 0

    with torch.no_grad():
        for x_img, gh_idx, x_aux, y in val_loader:
            x_img = x_img.to(device, non_blocking=True)
            gh_idx = gh_idx.to(device, non_blocking=True)
            x_aux = x_aux.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred = model(x_img, gh_idx, x_aux)    # ViT: no attention outputs

            y_true_all.append(y.detach().cpu().numpy())
            y_pred_all.append(pred.detach().cpu().numpy())
            nlcd_frac_all.append(x_aux[:, -ds.NLCD_AUX_DIM:].detach().cpu().numpy())

            bsz = x_img.size(0)
            n_seen += bsz

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    # CHANGE B: convert sqrt space → Mg/ha (version_7 inverted log1p here instead)
    y_true_all = sqrt_to_linear(y_true_all)
    y_pred_all = sqrt_to_linear(y_pred_all)

    mae = float(np.mean(np.abs(y_pred_all - y_true_all)))
    rmse = float(np.sqrt(np.mean((y_pred_all - y_true_all) ** 2)))
    r2 = r2_score(y_true_all, y_pred_all)

    print("\n=== VAL METRICS (Mg/ha) ===")
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

    # ---- professor's land-cover sensitivity breakdown ----
    # val_loader has shuffle=False over Subset(val_idx), so predictions come out in
    # val_idx order and the centre codes below line up with them row for row.
    print("\n=== LAND-COVER SENSITIVITY (professor's categories, centre-pixel NLCD) ===")
    center_codes = center_nlcd_codes(ds.df, val_idx)
    nlcd_frac_all = np.concatenate(nlcd_frac_all)
    assert len(center_codes) == len(y_true_all) == len(nlcd_frac_all), "row alignment broken"
    prof_df = professor_landcover_table(y_true_all, y_pred_all, center_codes,
                                        nlcd_frac_all, ds.NLCD_CLASS_TO_IDX)
    with pd.option_context("display.width", 220, "display.max_columns", 20,
                           "display.float_format", "{:.4f}".format):
        print(prof_df.to_string(index=False))
    prof_csv = Path(args.out).with_name(Path(args.out).stem + "_professor_landcover.csv")
    prof_df.to_csv(prof_csv, index=False)
    print("Saved:", prof_csv)
    out["professor_landcover"] = prof_df.to_dict(orient="records")

    Path(args.out).write_text(pd.Series(out).to_json(), encoding="utf-8")
    print("\nSaved summary to:", args.out)

    print("\n=== LAND COVER PERFORMANCE (majority NLCD class of each patch) ===")

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
    print(lc_df.to_string(index=False))

    lc_out_csv = Path(args.out).with_name(Path(args.out).stem + "_landcover_metrics.csv")
    lc_df.to_csv(lc_out_csv, index=False)
    print("\nSaved per-land-cover metrics to:", lc_out_csv)


if __name__ == "__main__":
    main()
