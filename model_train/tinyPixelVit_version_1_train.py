# we are using tiny pixel Vit model 




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

import os
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
from rasterio.transform import xy as rio_xy

from pyproj import Transformer

from rasterio.errors import RasterioIOError



################################## CHANGE VALUE HERE 
#CSV_PATH = "/s/chopin/e/proj/hyperspec/masfiq/csv_files/gedi_California_california_north_10_2021_whole_year_metadata.csv"
CSV_PATH = "/s/chopin/e/proj/hyperspec/masfiq/csv_files/gedi_california_north_10_2021_AprilToAugust_version_3.csv"


NUM_EPOCHS=100
BATCH_SIZE=32
LEARNING_RATE=1e-4
GEOHASH_PRESITION=7

#OUTPUT_PATH = Path("/s/chopin/e/proj/hyperspec/masfiq/models/tinyPixelViT_fusion_geohash_month_koppen_California_North_10_2021.pth")
OUTPUT_PATH = Path("/s/chopin/e/proj/hyperspec/masfiq/models/tinyPixelViT_fusion_geohash_month_koppen_version_3_California_North_10_2021.pth")







##################################

###################koppen debug 
DEBUG_KOPPEN = True
DEBUG_KOPPEN_SAMPLE_IDXS = {0, 100, 1000, 10000}

##################end

###################koppen debug 
# def load_koppen_legend(legend_path: str):
#     code_to_label = {}

#     if DEBUG_KOPPEN:
#         print("\n[KOPPEN DEBUG] Loading legend from:", legend_path)
#         print("[KOPPEN DEBUG] Legend exists:", Path(legend_path).exists())

#         if Path(legend_path).exists():
#             print("[KOPPEN DEBUG] First 10 raw legend lines:")
#             with open(legend_path, "r", encoding="utf-8") as f_preview:
#                 for i, line in enumerate(f_preview):
#                     if i >= 10:
#                         break
#                     print(f"  raw line {i}: {line.rstrip()}")

#     skipped_lines = 0

#     with open(legend_path, "r", encoding="utf-8") as f:
#         for line in f:
#             original_line = line.rstrip()
#             line = line.strip()

#             if not line or line.startswith("#"):
#                 continue

#             parts = line.split()

#             try:
#                 code = int(parts[0])
#             except Exception:
#                 skipped_lines += 1
#                 if DEBUG_KOPPEN and skipped_lines <= 10:
#                     print("[KOPPEN DEBUG] Skipped legend line because first item is not int:", original_line)
#                 continue

#             label = parts[1] if len(parts) > 1 else "UNK"
#             code_to_label[code] = label

#     codes_sorted = sorted(code_to_label.keys())
#     code_to_index = {c: i for i, c in enumerate(codes_sorted)}

#     if DEBUG_KOPPEN:
#         print("[KOPPEN DEBUG] Parsed legend class count:", len(codes_sorted))
#         print("[KOPPEN DEBUG] First 20 parsed codes:", codes_sorted[:20])
#         print("[KOPPEN DEBUG] First 20 code_to_label:", list(code_to_label.items())[:20])
#         print("[KOPPEN DEBUG] Skipped non-integer legend lines:", skipped_lines)

#     return code_to_label, codes_sorted, code_to_index
# ##################end


# original koppen method 
# def load_koppen_legend(legend_path: str):
#     code_to_label = {}
#     with open(legend_path, "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line or line.startswith("#"):
#                 continue
#             parts = line.split()
#             try:
#                 code = int(parts[0])
#             except:
#                 continue
#             label = parts[1] if len(parts) > 1 else "UNK"
#             code_to_label[code] = label

#     codes_sorted = sorted(code_to_label.keys())
#     code_to_index = {c: i for i, c in enumerate(codes_sorted)}
#     return code_to_label, codes_sorted, code_to_index


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



class KoppenGeigerLookup:
    def __init__(self, koppen_tif_path: str, legend_path: str, unknown_label="UNK"):
        self.koppen_tif_path = koppen_tif_path
        self.legend_path = legend_path

        self.code_to_label, self.codes_sorted, self.code_to_index = load_koppen_legend(legend_path)

        self.unknown_label = unknown_label
        self.unknown_index = len(self.codes_sorted)
        self.num_classes = len(self.codes_sorted) + 1

        ####################### koppen debug 
        if DEBUG_KOPPEN:
            print("\n[KOPPEN DEBUG] KoppenGeigerLookup initialized")
            print("[KOPPEN DEBUG] koppen_tif_path:", self.koppen_tif_path)
            print("[KOPPEN DEBUG] tif exists:", Path(self.koppen_tif_path).exists())
            print("[KOPPEN DEBUG] legend_path:", self.legend_path)
            print("[KOPPEN DEBUG] legend exists:", Path(self.legend_path).exists())
            print("[KOPPEN DEBUG] real class count:", len(self.codes_sorted))
            print("[KOPPEN DEBUG] unknown_index:", self.unknown_index)
            print("[KOPPEN DEBUG] num_classes:", self.num_classes)

        ######################## end koppen debug 



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






# 0) Small utilities

BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
BASE32_MAP = {c: i for i, c in enumerate(BASE32)}

def geohash_encode(lat: float, lon: float, precision: int = 7) -> str:
   
    # Pure-python geohash encoder (no external dependency).
    # Returns a base32 geohash string of length `precision`.
    # lat = 40.5853
    # lon = -105.0844
    # geohash = "9xjq..." The more characters I use, the more precise the location becomes.
    
    lat_interval = [-90.0, 90.0]
    lon_interval = [-180.0, 180.0]
    geohash = []
    is_even = True
    bit = 0
    ch = 0
    bits = [16, 8, 4, 2, 1]

    # When is_even=True, it checks longitude.
    # When is_even=False, it checks latitude.

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
    
    # Example granule_id:
    #   HLS.S30.T10TEK.2021057T190939.v2.0
    # We parse YYYY + DOY (first 7 digits of the date token), then convert to month [1..12].
   
    try:
        date_token = granule_id.split(".")[3]  # "2021057T190939" so 57th day of 2021 year  
        y = int(date_token[0:4])
        doy = int(date_token[4:7])
        dt = datetime(y, 1, 1) + timedelta(days=doy - 1) # This becomes around: February 26, 2021 We use doy - 1 because January 1 is already day 1
        return dt.month
    except Exception:
        return 1  # safe fallback

def month_sincos(month: int) -> np.ndarray:
    
    # sin-cos encoding month: 1..12 -> [sin(2πm/12), cos(2πm/12)] 
    
    m = float(month)
    ang = 2.0 * math.pi * (m / 12.0)
    return np.array([math.sin(ang), math.cos(ang)], dtype=np.float32)



#  Dataset: image + (geohash, month, koppen)

class GEDIHlsPatchDatasetFusion(Dataset):

    # Reads metadata CSV of patch GeoTIFFs.
    # Returns:
    #   x_img   : (C,3,3) float32
    #   gh_idx  : (G,) int64   (geohash indices)
    #   x_aux   : (A,) float32 (month sin/cos + koppen one-hot)
    #   y       : ()   float32 target
    
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
        self.df = pd.read_csv(self.csv_path) # take the csv to dataframe

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
    
    # this method returns 
    #     out       = final image patch array, shape (12, 3, 3)
    # crs       = coordinate reference system of the GeoTIFF
    # transform = pixel-to-map coordinate transform
    def _load_patch_fixed_channels(self, patch_path: Path, bands_str: str | None) -> np.ndarray:
      
        # Load GeoTIFF (C,3,3) and map it into fixed channel order (len(FIXED_BANDS),3,3).
      
        with rasterio.open(patch_path) as src:
            raw = src.read().astype(np.float32)  # (C,3,3).  C = number of bands stored in that patch file
            nodata = src.nodata
            crs = src.crs
            transform = src.transform

            #  nodata    = value used for missing pixels
            # crs       = map projection information
            # transform = how pixel locations map to real-world coordinates

        if nodata is not None:
            raw[raw == nodata] = np.nan # make no data Nan
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)# then convert it to 0.0 

        # Map by band names stored in CSV column "bands" (you write this already)
        out = np.zeros((len(self.FIXED_BANDS), 3, 3), dtype=np.float32) # if 12 bands it is (12,3,3) 12 matrices each matrix has 3 rows and 3 columns

        if isinstance(bands_str, str) and len(bands_str) > 0:
            used_bands = [b.strip() for b in bands_str.split(",")] # If the CSV has a valid bands column, this line splits the band list.
            # raw[i] corresponds to used_bands[i]

            # below loop
            # enumerate(used_bands) gives two things at the same time
            # i = index of the band in the used_bands
            # b = band name 
            # raw[0] is B02 → goes to out[0]
            # raw[1] is B03 → goes to out[1]
            # raw[3] is NDVI → goes to out[10]
            # raw[4] is NLCD → goes to out[11]
            for i, b in enumerate(used_bands): 
                if i >= raw.shape[0]: # raw.shape = (12, 3, 3)
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
        #         This checks the maximum value of each band.
        # If the max value is greater than 2.0, the code assumes the band is stored in scaled reflectance format, like:
        # 0 to 10000
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
        # reads one row of the CSV
        row = self.df.iloc[idx]


        # Get the patch image path from the CSV. If the path is relative, convert it into a full path. Then check whether the file exists. If it does not exist, stop and show an error.
        patch_path = Path(row["patch_tifs"])
        if not patch_path.is_absolute():
            patch_path = self.csv_path.parent / patch_path
        if not patch_path.exists():
            raise FileNotFoundError(f"Patch file not found: {patch_path}")

        bands_str = row["bands"] if "bands" in self.df.columns else None

        # print(crs) - EPSG:32610    / EPSG:4326
        # EPSG:4326  = latitude/longitude
        # EPSG:32610 = UTM Zone 10N
        # EPSG:32611 = UTM Zone 11N
        # print(transform) Affine(30.0, 0.0, 500000.0,
        #    0.0, -30.0, 4200000.0)
        # x_img_np = the image/patch stack with all the bands
        x_img_np, crs, transform = self._load_patch_fixed_channels(patch_path, bands_str) 
        
        # model is written in PyTorch. PyTorch models cannot directly train on NumPy arrays. They need PyTorch tensors.So this line changes:
        # NumPy array → PyTorch tensor
        x_img = torch.from_numpy(x_img_np)

        # center pixel (1,1)
        # converts the center pixel location (row=1, col=1) into map coordinates using the raster’s transform.
        x, y = rio_xy(transform, 1, 1, offset="center") 
        if crs is None:
            lat, lon = 0.0, 0.0
        else:
            #converts the map coordinates from the raster’s CRS into standard latitude/longitude
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(x, y)

        # month encoding
        #  
        if "month" in self.df.columns:
            m = int(row["month"])
        elif "hls_granule" in self.df.columns:
            # getting teh month as we know hls file name from csv 
            m = parse_hls_month_from_granule(str(row["hls_granule"]))
        else:
            m = 1
        m_sc = month_sincos(m)  # (2,) # m_sc is month encoding 

      

        gh = geohash_encode(lat, lon, precision=self.geohash_precision)
        # making the numpy array of the geohash index 
        gh_idx = torch.from_numpy(geohash_to_indices(gh))  # (G,)

        # --- koppen encoding from raster (recommended) ---
        # Suppose the Köppen classes are mapped like this:

                # 0 = Af
                # 1 = Am
                # 2 = Aw
                # 3 = BWh
                # 4 = BSk
                # ...

                # If a location belongs to class BSk, and BSk has index 4, then the one-hot vector becomes:

                # [0, 0, 0, 0, 1, 0, 0, ...]

                # Only the matching class position is 1. All others are 0.
        if self.koppen_lookup is not None:
            # it tries to get the Köppen class for the patch location:
            # lon and lat are the center location of the 3×3 patch
            try:
                koppen_onehot, _ = self.koppen_lookup.encode_onehot(lon, lat)
            except Exception:
                # treat as unknown if anything goes wrong
                
                # If the raster lookup fails:

                # except Exception:

                # then the code creates a vector of zeros:

                # koppen_onehot = np.zeros((self.koppen_lookup.num_classes,), dtype=np.float32)

                # Then it marks the unknown class as 1:

                # koppen_onehot[self.koppen_lookup.unknown_index] = 1.0
                koppen_onehot = np.zeros((self.koppen_lookup.num_classes,), dtype=np.float32)
                koppen_onehot[self.koppen_lookup.unknown_index] = 1.0
        else:
            # If self.koppen_lookup is None, then:

            # koppen_onehot = np.zeros((0,), dtype=np.float32)

            # This means:

            # Do not use Köppen information. Create an empty feature vector.
            koppen_onehot = np.zeros((0,), dtype=np.float32)

        # --- build auxiliary vector: month sin/cos + koppen onehot ---
        # The vector looks like this 
        # x_aux_np = [
        # 1.0, 0.0,        
        # 0, 0, 0, 0, 0, 1, 0, 0, 0, ..., 0   # Köppen one-hot
        # ]
        x_aux_np = np.concatenate([m_sc, koppen_onehot], axis=0).astype(np.float32)
        x_aux = torch.from_numpy(x_aux_np)  # (A,)


        # --- target ---
        # row is one row from your CSV file. self.target_col is the column name you selected as the target, for example:

        # target_col="agbd_center"
        y_val = float(row[self.target_col])
        #This checks whether the target value is valid.
        if not np.isfinite(y_val):
            raise ValueError(f"Non-finite target at idx={idx}: {y_val}")
        
        # This converts the target value into a PyTorch tensor.
        y = torch.tensor(y_val, dtype=torch.float32)
        # x_img        = image patch, shape usually (12, 3, 3)
        # gh_idx.long() = geohash index vector, integer type
        # x_aux        = auxiliary vector, month + Köppen
        # y            = target biomass value
        # when the DataLoader combines many samples into a batch, for batch size 32, the y values become:

        # tensor([85.42, 67.10, 102.33, ..., 45.80])

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
        self.num_tokens = 9  # 3x3 pixels

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

def create_dataloaders(csv_path, batch_size=32, val_frac=0.2, geohash_precision=7):
    # koppen_tif = "/s/chopin/e/proj/hyperspec/masfiq/dataset/koppen_geiger_tif/1991_2020/koppen_geiger_0p00833333.tif"
    # legend     = "/s/chopin/e/proj/hyperspec/masfiq/dataset/koppen_geiger_tif/legend.txt"
    koppen_tif = os.environ.get(
        "KOPPEN_TIF",
        "/s/chopin/e/proj/hyperspec/masfiq/dataset/koppen_geiger_tif/1991_2020/koppen_geiger_0p00833333.tif"
    )
    legend = "/s/chopin/e/proj/hyperspec/masfiq/dataset/koppen_geiger_tif/legend.txt"


    # creating an object of the GEDIHlsPatchDatasetFusion class
    ds = GEDIHlsPatchDatasetFusion(
        csv_path,
        target_col="agbd_log",
        geohash_precision=7,
        koppen_tif_path=koppen_tif,
        koppen_legend_path=legend
    )


    # N is basically the number of rows in the CSV file, how many patches 
    N = len(ds)
    #creates index numbers: [0, 1, 2, 3, ..., 99999]
    idx = np.arange(N)

    #     randomly shuffles those indices.
    # This is done so the code can randomly split the dataset into training and validation sets.
    np.random.shuffle(idx)

    val_size = int(N * val_frac)
    #     If shuffled idx is:[3, 8, 1, 6, 0, 9, 2, 5, 7, 4] Then: val_idx = idx[:2] becomes: [3, 8] and: train_idx = idx[2:]becomes: [1, 6, 0, 9, 2, 5, 7, 4]
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]


    # creating two subsets of the original dataset ds based on the index lists
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    #This checks how many CPUs I requested in my SLURM bash script in the variable SLURM_CPUS_PER_TASK
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "13"))
    num_workers = min(15, max(2, num_workers))


    #  dataset object train_ds knows how to return one sample:
    # x_img, gh_idx, x_aux, y
    # But the model should not train one sample at a time. It is more efficient to train using a batch, like 32 samples at once. The DataLoader does that batching 

    # dataloader object breakdown train_loader
    #     Take training samples from train_ds
    # Group them into batches of size 32
    # Shuffle them every epoch
    # Use multiple CPU workers to load data faster
    #     pin_memory=True This helps transfer data from CPU memory to GPU memory faster. persistent_workers=True This keeps the DataLoader worker processes alive between epochs.
    #     prefetch_factor=4 This means each worker prepares some future batches in advance. So while the GPU is training on the current batch, the CPU workers are already preparing the next batches.

 


    #     So when  training loop says:
    # for x_img, gh_idx, x_aux, y in train_loader
    # it gives one batch at a time.
    # If batch_size = 32, then each batch contains
    # 32 image patches
    # 32 geohash vectors
    # 32 auxiliary feature vectors
    # 32 target biomass values

    

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True,
                          persistent_workers=True, prefetch_factor=4)
    
    # For validation, we usually do not shuffle, because the model is not learning from validation data. We only want to evaluate performance consistently.
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True,
                          persistent_workers=True, prefetch_factor=4)

    # Peek one sample from ds object 
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
    

    # we are iusing pytorch for DL , 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, val_loader, in_channels, aux_dim = create_dataloaders(
        csv_path,
        batch_size=batch_size,
        val_frac=val_frac,
        geohash_precision=geohash_precision,
    )

    model = TinyPixelViTFusionRegressor(
    in_channels=in_channels,
    geohash_precision=geohash_precision,
    aux_dim=aux_dim,
    geohash_emb_dim=16,
    embed_dim=64,
    num_heads=4,
    num_layers=2,
    mlp_dim=128,
    hidden_dim=256,
    dropout=0.2,
    ).to(device)  

    # >>> ADD THIS HERE (fine-tune init) <<<
    # init_ckpt = "/s/chopin/e/proj/hyperspec/masfiq/models/field_boundary_biomass_resnet18withNLCD_features_cnn_8band_3x3.pth"
    # ckpt = torch.load(init_ckpt, map_location=device)
    # state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    # model.load_state_dict(state, strict=False)  # strict=False because attention params are new

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def mae(pred, target):
        return torch.mean(torch.abs(pred - target))
    
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

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

        print(
            f"Epoch {epoch:03d} | "
            f"Train MSE {tr_mse:.4f} MAE {tr_mae:.4f} | "
            f"Val MSE {va_mse:.4f} MAE {va_mae:.4f}"
        )

    return model


if __name__ == "__main__":
    csv_path = CSV_PATH

    # number of epochs - the model will see the full training dataset 60 times.
    # batch size -  how many samples the model processes at one time before updating the weights. 
    # batch size = 32 means 32 patches per training step
    # learning rate - how big each weight update is.
    # High learning rate = model learns faster, but may become unstable
    # Low learning rate = model learns slower, but usually more stable
    # # #val_frac = 0.2
    # This means 20% of the dataset is used for validation, and the remaining 80% is used for training.
    #One epoch means the model goes through the entire training dataset once.
    # So if your training set has, for example:
    # 80,000 training patches
    # and batch size is:
    # 32
    # then in one epoch, the model still uses all 80,000 patches. It just processes them in small groups:
    # Batch 1: 32 patches
    # Batch 2: 32 patches
    # Batch 3: 32 patches .....
    # Number of batches per epoch would be:
    # 80,000 / 32 = 2,500 batches per epoch

    # updating the weight means 
    #     1. Model makes predictions for 32 patches
    # 2. Loss is calculated
    # 3. Backpropagation calculates gradients
    # 4. Optimizer updates trainable weights across the whole model

    #     So it updates weights in many parts of the model, such as:

    # Convolution layers
    # Transformer layers, if using ViT
    # Attention layers, if present
    # Geohash embedding layer
    # Final regression MLP

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

