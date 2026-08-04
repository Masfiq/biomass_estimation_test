# Similar to resnet18 version 6 
# we will add gradient clipping here 

# resnet18 model 
# all 12 bands "B02","B03","B04","B05","B06","B07","B8A","B11","B12","EVI","NDVI","NLCD"
# aux - koppen + month encoding + geohash + gridmet  

# these are the gridMet variables that we are using 
# pr_2021.nc     = precipitation
# rmax_2021.nc   = maximum relative humidity
# rmin_2021.nc   = minimum relative humidity
# sph_2021.nc    = specific humidity
# tmmn_2021.nc   = minimum temperature
# tmmx_2021.nc   = maximum temperature

#############changes made 
# this one is using seed as we are using a fixed seed for validation 

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
from rasterio.errors import RasterioIOError

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
from rasterio.transform import xy as rio_xy, rowcol, from_gcps
from rasterio.crs import CRS

from pyproj import Transformer

import os
import xarray as xr 

import re 


################################## CHANGE VALUE HERE 
#CSV_PATH = "/s/chopin/e/proj/hyperspec/masfiq/csv_files/gedi_California_california_north_10_2021_whole_year_metadata.csv"
CSV_PATH = "/s/chopin/e/proj/hyperspec/masfiq/csv_files/gedi_california_north_10_2021_AprilToAugust_version_5.csv"


NUM_EPOCHS=100
BATCH_SIZE=32
LEARNING_RATE=1e-5
GEOHASH_PRESITION=7

#OUTPUT_PATH = Path("/s/chopin/e/proj/hyperspec/masfiq/models/resnet18_version_9_fusion_geohash_month_koppen_withAttentionLayer_SAR1_California_North_10_2021.pth")
OUTPUT_PATH = Path("/s/chopin/e/proj/hyperspec/masfiq/models/resnet18_version_10_fusion_geohash_month_koppen_withAttentionLayer_version_5_California_North_10_2021.pth")


SEED = 42


GRIDMET_DIR = "/s/chopin/e/proj/hyperspec/masfiq/dataset/gridMET_weather_data"


#SENTINEL1_DIR = "/s/chopin/e/proj/hyperspec/masfiq/dataset/sentinel_1_data/sentinel_1_data_california_north_10_2021_April_toAugust"
SENTINEL1_DIR = None


fixed_bands = [
    "B02","B03","B04",
    "B05","B06","B07",       # L30 (Landsat) only — zero for S30 patches
    "B8A","B11","B12",       # S30 (Sentinel-2) only — zero for L30 patches
    "EVI","NDVI","NLCD",
]


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



class GridMETLookup:
    def __init__(self, gridmet_dir: str):
        self.gridmet_dir = Path(gridmet_dir)

        self.var_files = {
            "pr": self.gridmet_dir / "pr_2021.nc",
            "rmax": self.gridmet_dir / "rmax_2021.nc",
            "rmin": self.gridmet_dir / "rmin_2021.nc",
            "sph": self.gridmet_dir / "sph_2021.nc",
            "tmmn": self.gridmet_dir / "tmmn_2021.nc",
            "tmmx": self.gridmet_dir / "tmmx_2021.nc",
        }

        self.datasets = None
        self._pid = None

    def __getstate__(self):
        d = self.__dict__.copy()
        d["datasets"] = None
        d["_pid"] = None
        return d

    def _ensure_open(self):
        pid = os.getpid()

        if self.datasets is None or self._pid != pid:
            self.close()
            self.datasets = {}

            for name, path in self.var_files.items():
                if not path.exists():
                    raise FileNotFoundError(f"Missing gridMET file: {path}")

                self.datasets[name] = xr.open_dataset(path)

            self._pid = pid

    def close(self):
        if self.datasets is not None:
            for ds in self.datasets.values():
                try:
                    ds.close()
                except Exception:
                    pass

        self.datasets = None
        self._pid = None

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

        # fallback: use first data variable
        return list(ds.data_vars.keys())[0]

    def sample_raw(self, lon: float, lat: float, date_value: np.datetime64) -> np.ndarray:
        self._ensure_open()

        values = []

        for name, ds in self.datasets.items():
            var_name = self._data_var_name(ds, name)

            lat_name = self._coord_name(ds, ["lat", "latitude", "y"])
            lon_name = self._coord_name(ds, ["lon", "longitude", "x"])
            time_name = self._coord_name(ds, ["day", "time", "date"])

            da = ds[var_name]

            val = da.sel(
                {
                    lat_name: lat,
                    lon_name: lon,
                    time_name: date_value,
                },
                method="nearest",
            ).values

            values.append(float(np.asarray(val)))

        return np.array(values, dtype=np.float32)

    def normalize(self, raw: np.ndarray) -> np.ndarray:
        """
        Order:
        [pr, rmax, rmin, sph, tmmn, tmmx]

        Simple physical scaling.
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
    


class Sentinel1Lookup:
    def __init__(self, sentinel1_dir: str, debug=True, max_debug_prints=20):
        self.sentinel1_dir = Path(sentinel1_dir)
        self.debug = debug
        self.max_debug_prints = max_debug_prints
        self.debug_count = 0

        print("\n[S1 DEBUG] Sentinel1Lookup initialized", flush=True)
        print("[S1 DEBUG] sentinel1_dir:", self.sentinel1_dir, flush=True)
        print("[S1 DEBUG] sentinel1_dir exists:", self.sentinel1_dir.exists(), flush=True)

        self.scenes = self._scan_scenes()

        print("[S1 DEBUG] Number of Sentinel-1 scenes with VV/VH:", len(self.scenes), flush=True)

        # for s in self.scenes[:5]:
        #     print(
        #         f"[S1 DEBUG] scene date={s['date']} vv={s['vv']} vh={s['vh']}",
        #         flush=True
        #     )




    def _get_spatial_ref(self, src):
        """
        Get CRS and transform from normal CRS first.
        If CRS is missing, try GCPs.
        If the raster already looks like lon/lat, assume EPSG:4326.
        """

        # Case 1: normal CRS exists
        if src.crs is not None and str(src.crs).strip() != "":
            return src.crs, src.transform

        # Case 2: try GCPs
        try:
            gcps, gcp_crs = src.gcps
            if gcp_crs is not None and len(gcps) > 0:
                return gcp_crs, from_gcps(gcps)
        except Exception:
            pass

        # Case 3: if bounds look like longitude/latitude, assume EPSG:4326
        b = src.bounds
        if (
            -180 <= b.left <= 180 and
            -180 <= b.right <= 180 and
            -90 <= b.bottom <= 90 and
            -90 <= b.top <= 90
        ):
            return CRS.from_epsg(4326), src.transform

        raise ValueError(
            f"Sentinel-1 raster has no usable CRS or GCP CRS. "
            f"crs={src.crs}, bounds={src.bounds}"
        )

    def _should_print(self):
        return self.debug and self.debug_count < self.max_debug_prints

    def _parse_date(self, path: Path):
        """
        Example filename contains:
        20210818T142315
        """
        m = re.search(r"(\d{8})T\d{6}", str(path))
        if m is None:
            return None

        date_str = m.group(1)
        dt = datetime.strptime(date_str, "%Y%m%d").date()
        return np.datetime64(dt)

    def _scan_scenes(self):
        grouped = {}

        tif_files = list(self.sentinel1_dir.glob("**/measurement/*.tif")) + \
                    list(self.sentinel1_dir.glob("**/measurement/*.tiff"))

        print("[S1 DEBUG] Total Sentinel-1 measurement tif files found:", len(tif_files), flush=True)

        for path in tif_files:
            name = path.name.lower()
            date_value = self._parse_date(path)

            if date_value is None:
                continue

            key = (str(path.parent), str(date_value))

            if key not in grouped:
                grouped[key] = {"date": date_value, "vv": None, "vh": None}

            if "-vv-" in name:
                grouped[key]["vv"] = path
            elif "-vh-" in name:
                grouped[key]["vh"] = path

        scenes = []
        for _, item in grouped.items():
            if item["vv"] is not None and item["vh"] is not None:
                scenes.append(item)

        scenes = sorted(scenes, key=lambda x: x["date"])
        return scenes

    def _date_distance_days(self, d1, d2):
        return abs((pd.to_datetime(str(d1)) - pd.to_datetime(str(d2))).days)

    def _normalize_s1_patch(self, arr: np.ndarray) -> np.ndarray:
        """
        Normalize Sentinel-1 COG pixel values without saturating everything to 1.

        This is a practical fixed scaling for the current raw COG values.
        It keeps relative differences between pixels.
        """
        arr = arr.astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        # Remove negative/invalid values
        arr = np.maximum(arr, 0.0)

        # Log compression for positive SAR-like values
        arr_norm = np.log1p(arr) / np.log1p(1000.0)

        # Keep inside 0 to 1
        arr_norm = np.clip(arr_norm, 0.0, 1.0)

        return arr_norm.astype(np.float32)

    def _read_patch_from_tif(self, tif_path: Path, lon: float, lat: float):
        with rasterio.open(tif_path) as src:
            raster_crs, raster_transform = self._get_spatial_ref(src)

            # If Sentinel-1 raster is already lon/lat
            if CRS.from_user_input(raster_crs).to_epsg() == 4326:
                x, y = lon, lat
            else:
                transformer = Transformer.from_crs(
                    "EPSG:4326",
                    raster_crs,
                    always_xy=True
                )
                x, y = transformer.transform(lon, lat)

            row, col = rowcol(raster_transform, x, y)
            row = int(row)
            col = int(col)

            if row < 0 or row >= src.height or col < 0 or col >= src.width:
                raise ValueError(
                    f"Point outside Sentinel-1 raster bounds. "
                    f"row={row}, col={col}, height={src.height}, width={src.width}"
                )

            window = Window(col - 1, row - 1, 3, 3)

            patch = src.read(
                1,
                window=window,
                boundless=True,
                fill_value=0
            ).astype(np.float32)

        return patch

    def sample_patch(self, lon: float, lat: float, date_value: np.datetime64) -> np.ndarray:
        """
        Returns:
            s1_patch: shape (2, 3, 3)
                      channel 0 = VV
                      channel 1 = VH
        """
        if len(self.scenes) == 0:
            print("[S1 ERROR] No Sentinel-1 VV/VH scenes found.", flush=True)
            return np.zeros((2, 3, 3), dtype=np.float32)

        scenes_sorted = sorted(
            self.scenes,
            key=lambda s: self._date_distance_days(s["date"], date_value)
        )

        if self._should_print():
            print("\n[S1 DEBUG] sample_patch() called", flush=True)
            print(f"[S1 DEBUG] requested lon={lon}, lat={lat}, date={date_value}", flush=True)

        for scene in scenes_sorted[:10]:
            try:
                vv_raw = self._read_patch_from_tif(scene["vv"], lon, lat)
                vh_raw = self._read_patch_from_tif(scene["vh"], lon, lat)

                vv_norm = self._normalize_s1_patch(vv_raw)
                vh_norm = self._normalize_s1_patch(vh_raw)

                out = np.stack([vv_norm, vh_norm], axis=0).astype(np.float32)

                if self._should_print():
                    delta_days = self._date_distance_days(scene["date"], date_value)

                    print(f"[S1 DEBUG] selected scene date={scene['date']}", flush=True)
                    print(f"[S1 DEBUG] day difference={delta_days}", flush=True)
                    print(f"[S1 DEBUG] VV path={scene['vv']}", flush=True)
                    print(f"[S1 DEBUG] VH path={scene['vh']}", flush=True)
                    print(f"[S1 DEBUG] VV raw min/max={vv_raw.min():.6f}/{vv_raw.max():.6f}", flush=True)
                    print(f"[S1 DEBUG] VH raw min/max={vh_raw.min():.6f}/{vh_raw.max():.6f}", flush=True)
                    print(f"[S1 DEBUG] VV norm min/max={vv_norm.min():.6f}/{vv_norm.max():.6f}", flush=True)
                    print(f"[S1 DEBUG] VH norm min/max={vh_norm.min():.6f}/{vh_norm.max():.6f}", flush=True)
                    print(f"[S1 DEBUG] output shape={out.shape}", flush=True)

                self.debug_count += 1
                return out

            except Exception as e:
                if self._should_print():
                    print(
                        f"[S1 WARNING] Could not use scene date={scene['date']}. Error: {e}",
                        flush=True
                    )
                continue

        print(
            f"[S1 ERROR] Could not sample any Sentinel-1 scene for lon={lon}, lat={lat}, date={date_value}. Using zeros.",
            flush=True
        )

        return np.zeros((2, 3, 3), dtype=np.float32)


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
    FIXED_BANDS = fixed_bands

    def __init__(
        self,
        csv_path: str,
        geohash_precision: int = 7,
        target_col: str | None = None,
        koppen_tif_path=None,
        koppen_legend_path=None,
        gridmet_dir=None,
        sentinel1_dir=None
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
            self.gridmet_lookup = GridMETLookup(gridmet_dir)
            self.gridmet_dim = 6

        self.sentinel1_lookup = None
        self.sentinel1_dim = 0

        if sentinel1_dir is not None:
            print("Sentinel-1 dir added\n", flush=True)
            self.sentinel1_lookup = Sentinel1Lookup(sentinel1_dir, debug=False)
            self.sentinel1_dim = 2

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

        # center pixel (1,1)
        x, y = rio_xy(transform, 1, 1, offset="center")
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

        # --- date for GridMET and Sentinel-1 ---
        if "hls_granule" in self.df.columns:
            date_value = parse_hls_date_from_granule(str(row["hls_granule"]))
        elif "date" in self.df.columns:
            date_value = np.datetime64(pd.to_datetime(row["date"]).date())
        else:
            date_value = np.datetime64("2021-01-01")

      

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

        # --- gridMET weather encoding ---
        # --- gridMET weather encoding ---
        if self.gridmet_lookup is not None:
            try:
                gridmet_features = self.gridmet_lookup.encode(lon, lat, date_value)
            except Exception:
                gridmet_features = np.zeros((6,), dtype=np.float32)
        else:
            gridmet_features = np.zeros((0,), dtype=np.float32)


        # --- Sentinel-1 VV/VH image channels ---
        if self.sentinel1_lookup is not None:
            try:
                # print(
                #     f"\n[S1 DEBUG __getitem__] idx={idx}, lat={lat}, lon={lon}, date={date_value}",
                #     flush=True
                # )

                s1_patch = self.sentinel1_lookup.sample_patch(lon, lat, date_value)

                vv_idx = self.FIXED_BANDS.index("S1_VV")
                vh_idx = self.FIXED_BANDS.index("S1_VH")

                x_img_np[vv_idx] = s1_patch[0]
                x_img_np[vh_idx] = s1_patch[1]

                # print("[S1 DEBUG __getitem__] S1 patch added successfully", flush=True)

                # if np.all(s1_patch == 0):
                #     # print("[S1 WARNING __getitem__] S1 patch is all zeros", flush=True)
                # else:
                #     # print("[S1 DEBUG __getitem__] S1 patch added successfully", flush=True)

                    
                # print("[S1 DEBUG __getitem__] S1_VV center:", x_img_np[vv_idx, 1, 1], flush=True)
                # print("[S1 DEBUG __getitem__] S1_VH center:", x_img_np[vh_idx, 1, 1], flush=True)

            except Exception as e:
                print(
                    f"[S1 ERROR __getitem__] idx={idx}, failed to add Sentinel-1. Error: {e}",
                    flush=True
                )
        else:
            pass

        # --- build auxiliary vector: month sin/cos + koppen onehot + gridMET ---
        x_aux_np = np.concatenate(
            [m_sc, koppen_onehot, gridmet_features],
            axis=0
        ).astype(np.float32)

        x_aux = torch.from_numpy(x_aux_np)


        x_img = torch.from_numpy(x_img_np)

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


# ----------------------------
# 3) Dataloaders + training
# ----------------------------
def create_dataloaders(csv_path, batch_size=32, val_frac=0.2, geohash_precision=GEOHASH_PRESITION, seed = 42):
    # koppen_tif = "/s/chopin/e/proj/hyperspec/masfiq/dataset/koppen_geiger_tif/1991_2020/koppen_geiger_0p00833333.tif"
    # legend     = "/s/chopin/e/proj/hyperspec/masfiq/dataset/koppen_geiger_tif/legend.txt"
    koppen_tif = os.environ.get(
        "KOPPEN_TIF",
        "/s/chopin/e/proj/hyperspec/masfiq/dataset/koppen_geiger_tif/1991_2020/koppen_geiger_0p00833333.tif"
    )
    legend = "/s/chopin/e/proj/hyperspec/masfiq/dataset/koppen_geiger_tif/legend.txt"

    ds = GEDIHlsPatchDatasetFusion(
        csv_path,
        target_col="agbd_log",
        geohash_precision=GEOHASH_PRESITION,
        koppen_tif_path=koppen_tif,
        koppen_legend_path=legend,
        gridmet_dir=GRIDMET_DIR,
        sentinel1_dir=SENTINEL1_DIR
    )

    N = len(ds)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)

    val_size = int(N * val_frac)
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "13"))
    num_workers = min(15, max(2, num_workers))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True,
                          persistent_workers=True, prefetch_factor=4)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True,
                          persistent_workers=True, prefetch_factor=4)

    # Peek
    x_img, gh_idx, x_aux, y = ds[0]
    in_channels = x_img.shape[0]
    aux_dim = x_aux.shape[0]
    print("Sample x_img:", x_img.shape, "in_channels=", in_channels)

    if "S1_VV" in ds.FIXED_BANDS and "S1_VH" in ds.FIXED_BANDS:
        vv_idx = ds.FIXED_BANDS.index("S1_VV")
        vh_idx = ds.FIXED_BANDS.index("S1_VH")

        print("Sample S1_VV patch:", x_img[vv_idx], flush=True)
        print("Sample S1_VH patch:", x_img[vh_idx], flush=True)
        print("Sample S1_VV center:", x_img[vv_idx, 1, 1].item(), flush=True)
        print("Sample S1_VH center:", x_img[vh_idx, 1, 1].item(), flush=True)



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
    geohash_precision=GEOHASH_PRESITION,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, val_loader, in_channels, aux_dim = create_dataloaders(
        csv_path,
        batch_size=batch_size,
        val_frac=val_frac,
        geohash_precision=geohash_precision,
        seed=SEED
    )

    model = ResNet18FusionRegressorAttn(
        in_channels=in_channels,
        geohash_precision=geohash_precision,
        aux_dim=aux_dim,
        geohash_emb_dim=16,
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

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    def mae(pred, target):
        return torch.mean(torch.abs(pred - target))
    
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # use_amp = False 
    # scaler = None 

    for epoch in range(1, num_epochs + 1):
        # ---- train ----
        model.train()
        tr_mse, tr_mae, n_tr = 0.0, 0.0, 0

        for x_img, gh_idx, x_aux, y in train_loader:
            x_img = x_img.to(device, non_blocking=True)
            gh_idx = gh_idx.to(device, non_blocking=True)
            x_aux  = x_aux.to(device, non_blocking=True)
            y      = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    pred = model(x_img, gh_idx, x_aux)
                    loss = criterion(pred, y)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(x_img, gh_idx, x_aux)
                loss = criterion(pred, y)

                if not torch.isfinite(loss):
                    print("Non-finite loss detected")
                    print("pred min/max:", pred.min().item(), pred.max().item())
                    print("y min/max:", y.min().item(), y.max().item())
                    raise RuntimeError("Loss became NaN or Inf")



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

        scheduler.step(va_mse)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d} | "
            f"Train Huber {tr_mse:.4f} MAE {tr_mae:.4f} | "
            f"Val Huber {va_mse:.4f} MAE {va_mae:.4f} | "
            f"LR {current_lr:.2e}"
        )

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

