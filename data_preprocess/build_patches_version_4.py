# Building 3x3 patches for HLS (8 bands) + NLCD (1 band) = 9 bands total. No SAR bands in this version.
# v3 changes vs v2:
# (1) DOY filter keeps only April–August GEDI shots (DOY 91–240) to match HLS seasonal window
# (2) agbd_log column added to CSV (log1p of agbd_center) alongside agbd_raw and agbd_center
# same as version 3 just added the SAR 1 link. 

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import geopandas as gpd

import rasterio
from rasterio.windows import Window
from rasterio.warp import transform as rio_transform, reproject
from rasterio.enums import Resampling

import shutil
import os
import atexit
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import re
from datetime import datetime

# -------------------------
# Globals set in each worker
# -------------------------
_GRANULE_DICT = None
_NLCD_SRC = None
_S1_INDEX = None
_MAX_S1_DAYS_DIFF = None
_S1_TO_DB = False

############################ all the paths here # edit 
GEDI_FOLDER = r"/s/chopin/e/proj/hyperspec/masfiq/dataset/gedi_l4a_california_north_10_2021_whole_year"

#GEDI_FOLDER = r"/s/chopin/e/proj/hyperspec/masfiq/dataset/gedi_l4a_field_boundary_2021_whole_year"



HLS_FOLDER = r"/s/chopin/e/proj/hyperspec/masfiq/dataset/hls_data_California_north_10_2021_AprilToAugust"

#HLS_FOLDER = r"/s/chopin/e/proj/hyperspec/masfiq/dataset/hls_data_field_boundary_2021_AprilToAugust"

#S1_ROOT = None 

S1_ROOT = None

# Match Sentinel-1 scene to HLS date within this many days
MAX_S1_DAYS_DIFF = 12

# Keep False for now. Use True only if your Sentinel-1 values are valid positive power values.
S1_TO_DB = False



NLCD_PATH = r"/s/chopin/e/proj/hyperspec/masfiq/dataset/Annual_NLCD_LndCov_2021_CU_C1V1/Annual_NLCD_LndCov_2021_CU_C1V1.tif"

#PATCHES_ROOT = "/s/chopin/e/proj/hyperspec/masfiq/dataset/patch_tifs_california_north_10_2021_AprilToAugust"

PATCHES_ROOT = "/s/chopin/e/proj/hyperspec/masfiq/dataset/patch_tifs_california_north_10_2021_AprilToAugust_version_3"

#PATCHES_ROOT = "/s/chopin/e/proj/hyperspec/masfiq/dataset/patch_tifs_field_boundary_2021_AprilToAugust"

#OUT_CSV_PATH="/s/chopin/e/proj/hyperspec/masfiq/csv_files/gedi_California_california_north_10_2021_whole_year_metadata.csv"

OUT_CSV_PATH = "/s/chopin/e/proj/hyperspec/masfiq/csv_files/gedi_california_north_10_2021_AprilToAugust_version_3.csv"


#OUT_CSV_PATH="/s/chopin/e/proj/hyperspec/masfiq/csv_files/gedi_field_boundary_2021_whole_year_metadata.csv"

# diff between path= r"path" and path = "path"
# nlcd_path = "C:\new\test.tif"   # \n becomes newline, \t becomes tab (bad)
# nlcd_path = r"C:\new\test.tif"  # safe

#max_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "24"))
MAX_WORKERS = 16

CHUNK_SIZE = 20

GEOJSON_FILE =  gpd.read_file("/s/chopin/e/proj/hyperspec/masfiq/biomass_estimation_test/geojson_files/california_north_10.geojson")

#GEOJSON_FILE =  gpd.read_file("/s/chopin/e/proj/hyperspec/masfiq/biomass_estimation_test/geojson_files/Field_Boundary.geojson")

##################################################
def parse_hls_datetime(granule_id):
    """
    Example HLS granule ID:
    HLS.S30.T10TEK.2021108T190939.v2.0

    HLS date format is year + day-of-year:
    2021108T190939 = 2021, day 108, time 19:09:39
    """
    for part in granule_id.split("."):
        if re.fullmatch(r"\d{7}T\d{6}", part):
            return datetime.strptime(part, "%Y%jT%H%M%S")
    return None


def parse_s1_datetime_from_safe_name(safe_name):
    """
    Example Sentinel-1 SAFE name:
    S1A_IW_GRDH_1SDV_20210818T142315_20210818T142340_...
    """
    match = re.search(r"_(\d{8}T\d{6})_", safe_name)
    if match is None:
        return None
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")


def find_vv_vh_files(measurement_folder):
    """
    Finds VV and VH TIFF files inside a Sentinel-1 SAFE measurement folder.
    """
    measurement_folder = Path(measurement_folder)

    tif_files = list(measurement_folder.glob("*.tif")) + list(measurement_folder.glob("*.tiff"))

    vv_file = None
    vh_file = None

    for tif in tif_files:
        name = tif.name.lower()
        if "-vv-" in name:
            vv_file = tif
        elif "-vh-" in name:
            vh_file = tif

    return vv_file, vh_file


def index_sentinel1_scenes(s1_root):
    """
    Index all Sentinel-1 SAFE folders.

    Returns a list of scenes. Each scene contains:
    - SAFE name
    - acquisition time
    - VV TIFF path
    - VH TIFF path
    - CRS and bounds for spatial matching
    """
    s1_root = Path(s1_root)
    scenes = []

    for safe_dir in sorted(s1_root.rglob("*.SAFE")):
        measurement_folder = safe_dir / "measurement"
        if not measurement_folder.exists():
            continue

        vv_file, vh_file = find_vv_vh_files(measurement_folder)

        if vv_file is None or vh_file is None:
            print(f"Skipping {safe_dir.name}: VV or VH missing")
            continue

        acq_time = parse_s1_datetime_from_safe_name(safe_dir.name)

        try:
            with rasterio.open(vv_file) as src:
                scene = {
                    "safe_name": safe_dir.name,
                    "safe_path": str(safe_dir),
                    "vv_path": str(vv_file),
                    "vh_path": str(vh_file),
                    "acq_time": acq_time,
                    "crs_wkt": src.crs.to_wkt() if src.crs is not None else None,
                    "bounds": tuple(src.bounds),
                }
                scenes.append(scene)

        except Exception as e:
            print(f"Could not read Sentinel-1 scene {safe_dir.name}: {e}")

    print(f"Total Sentinel-1 scenes indexed: {len(scenes)}")
    return scenes


def find_s1_scene_for_point(lon, lat, s1_index, ref_datetime=None, max_days_diff=12):
    """
    Finds the Sentinel-1 scene that spatially covers the point AND is closest
    in time to ref_datetime (the HLS granule acquisition time).
    Only returns a scene if the temporal gap is within max_days_diff.
    """
    if s1_index is None:
        return None

    best_scene = None
    best_diff = None

    for scene in s1_index:
        if scene["crs_wkt"] is None:
            continue

        scene_crs = rasterio.crs.CRS.from_wkt(scene["crs_wkt"])

        xs, ys = rio_transform("EPSG:4326", scene_crs, [lon], [lat])
        x, y = xs[0], ys[0]

        left, bottom, right, top = scene["bounds"]

        if not (left <= x <= right and bottom <= y <= top):
            continue

        # If we have a reference datetime, pick the temporally closest scene
        if ref_datetime is not None and scene["acq_time"] is not None:
            diff = abs((scene["acq_time"] - ref_datetime).days)
            if diff > max_days_diff:
                continue
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_scene = scene
        else:
            # No date info available — take first spatial match
            return scene

    return best_scene


def reproject_s1_band_to_hls_patch(s1_tif_path, dst_transform, dst_crs, to_db=False):
    """
    Reproject one Sentinel-1 band into the exact same 3x3 HLS patch grid.

    Sentinel-1 is continuous radar data, so use bilinear resampling.
    """
    s1_patch = np.full((3, 3), np.nan, dtype=np.float32)

    with rasterio.open(s1_tif_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=s1_patch,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    s1_patch = s1_patch.astype(np.float32)

    if to_db:
        with np.errstate(divide="ignore", invalid="ignore"):
            s1_patch[s1_patch <= 0] = np.nan
            s1_patch = 10.0 * np.log10(s1_patch)

    return s1_patch




def load_gedi_shots_from_file(h5_path, min_quality=1):
    #the path of h5 files
    h5_path = Path(h5_path)
    rows = []

    # Extract DOY from filename: e.g. GEDI04_A_2021007070555_... → DOY 007 = Jan 7
    doy_match = re.search(r'GEDI04_A_\d{4}(\d{3})', h5_path.name)
    gedi_doy = int(doy_match.group(1)) if doy_match else -1

    # f.keys() is returning all the beam group like BEAMXXXX
    with h5py.File(h5_path, "r") as f:
        for beam_name in f.keys():
            if not beam_name.startswith("BEAM"):
                continue

            beam = f[beam_name]
            # we are getting 1 D numpy array from the beam group so lat[i] , agbd[i] will give us values for same shot number 
            try:
                lat = beam["lat_lowestmode"][:]# [:] this is basically convert the whole thing to 1D numpy array
                lon = beam["lon_lowestmode"][:]
                agbd = beam["agbd"][:]              # biomass
                quality = beam["l4_quality_flag"][:]
                shot_number = beam["shot_number"][:]
            except KeyError:
                print(f"Warning: missing vars in {h5_path.name}, {beam_name}")
                continue

            #quality >= min_quality → keeps shots with quality flag high enough
            # (if min_quality=1, it keeps quality==1)

            # np.isfinite(agbd) → removes shots where biomass is NaN, inf, -inf

            # So mask is a 1D boolean array, same length as lat, lon, agbd, etc.

            # Example:

            # mask = [True, False, True, True, ...]

            # 2) Apply the mask to every array
            # lat = lat[mask]
            # lon = lon[mask]
            # agbd = agbd[mask]
            # shot_number = shot_number[mask]
            # quality = quality[mask]


            # This is boolean indexing in NumPy.

            # It means: “keep only the rows where mask is True.”

            mask = (quality >= min_quality) & np.isfinite(agbd)
            lat = lat[mask]
            lon = lon[mask]
            agbd = agbd[mask]
            shot_number = shot_number[mask]
            quality = quality[mask]


            # inserting everything to a python dictionary 
            #   So rows becomes a list like:

            # [
            #   {"shot_number": ..., "lat": ..., "lon": ..., "agbd": ...},
            #   {"shot_number": ..., "lat": ..., "lon": ..., "agbd": ...},
            #   ...
            # ]
            for la, lo, a, q, sn in zip(lat, lon, agbd, quality, shot_number):
                rows.append(
                    {
                        "gedi_file": h5_path.name,
                        "beam": beam_name,
                        "shot_number": int(sn),
                        "lat": float(la),
                        "lon": float(lo),
                        "agbd": float(a),
                        "l4_quality_flag": int(q),
                        "gedi_doy": gedi_doy,
                    }
                )
    # Pandas reads the keys of the dicts as column names and the values as row values, creating a exel like table.
    return pd.DataFrame(rows)


def load_all_gedi_from_folder(gedi_folder, min_quality=1):
    """
    Loops over all *.h5 files in a folder and concatenates their shots
    into one big DataFrame.
    """
    # min_quality=1 - loading the gedi files whose quality is just 1 
    gedi_folder = Path(gedi_folder)
    dfs = []
    for h5_file in sorted(gedi_folder.glob("*.h5")):
        print(f"Reading {h5_file.name} ...")
        df_file = load_gedi_shots_from_file(h5_file, min_quality=min_quality)
        dfs.append(df_file)

    if len(dfs) == 0:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

def extract_3x3_patch_for_all_bands(
    granule_bands,
    row,
    col,
    sensor,
    nlcd_src=None,
    s1_scene=None,
    s1_to_db=False,
):
    """
    Read 3x3 HLS patch, add NLCD, and add Sentinel-1 VV/VH.

    Final output:
        8 HLS bands + 1 NLCD + 2 Sentinel-1 bands = 11 bands
    """
    if sensor == "S30":
        band_codes = ["B02", "B03", "B04", "B8A", "B11", "B12", "EVI", "NDVI"]
    else:  # L30
        band_codes = ["B02", "B03", "B04", "B05", "B06", "B07", "EVI", "NDVI"]

    patches = []
    used_bands = []
    window = None
    transform = None
    crs = None

    # 1. HLS bands
    for b in band_codes:
        if b not in granule_bands:
            return None, None, None, None

        path = granule_bands[b]

        with rasterio.open(path) as src:
            h, w = src.height, src.width

            if not (1 <= row < h - 1 and 1 <= col < w - 1):
                return None, None, None, None

            if window is None:
                window = Window(col_off=col - 1, row_off=row - 1, width=3, height=3)
                transform = src.window_transform(window)
                crs = src.crs

            patch = src.read(1, window=window).astype(np.float32)
            patches.append(patch)
            used_bands.append(b)

    if len(patches) == 0:
        return None, None, None, None

    # 2. NLCD band
    if nlcd_src is not None and window is not None and transform is not None and crs is not None:
        nlcd_patch = np.full((3, 3), fill_value=-1, dtype=np.float32)

        reproject(
            source=rasterio.band(nlcd_src, 1),
            destination=nlcd_patch,
            src_transform=nlcd_src.transform,
            src_crs=nlcd_src.crs,
            src_nodata=nlcd_src.nodata,
            dst_transform=transform,
            dst_crs=crs,
            dst_nodata=-1,
            resampling=Resampling.nearest,
        )

        patches.append(nlcd_patch.astype(np.float32))
        used_bands.append("NLCD")

    # 3. Sentinel-1 VV and VH bands
    if s1_scene is not None:
        vv_patch = reproject_s1_band_to_hls_patch(
            s1_scene["vv_path"],
            dst_transform=transform,
            dst_crs=crs,
            to_db=s1_to_db,
        )

        vh_patch = reproject_s1_band_to_hls_patch(
            s1_scene["vh_path"],
            dst_transform=transform,
            dst_crs=crs,
            to_db=s1_to_db,
        )

        # If the Sentinel-1 scene does not actually provide data at this patch, skip sample.
        if np.all(np.isnan(vv_patch)) or np.all(np.isnan(vh_patch)):
            return None, None, None, None

        patches.append(vv_patch)
        used_bands.append("S1_VV")

        patches.append(vh_patch)
        used_bands.append("S1_VH")

    data_cube = np.stack(patches, axis=0).astype(np.float32)

    return data_cube, used_bands, transform, crs

def index_hls_bands(hls_root):
    """
    Groups all HLS TIFs in a folder by granule ID.

    Example filename:
      HLS.S30.T10TEK.2021057T190939.v2.0.B02.tif

    We want:
      granule_id = HLS.S30.T10TEK.2021057T190939.v2.0
      band_code  = B02  (or Fmask, SZA, etc.)
    """
    hls_root = Path(hls_root)
    granules = defaultdict(dict)

    for tif in hls_root.glob("*.tif"):
        stem = tif.stem  # filename without ".tif"
        stem = stem.replace("_cropped", "")
        # parts = stem.split(".")
        parts = stem.replace("_", ".").split(".")

        # Expect something like:
        # ['HLS','S30','T10TEK','2021057T190939','v2','0','B02']
        if len(parts) < 7:
            continue

        granule_id = ".".join(parts[:-1])   # 'HLS.S30.T10TEK.2021057T190939.v2.0'
        band_code  = parts[-1]              # 'B02' or 'Fmask' or 'SZA', ...

        granules[granule_id][band_code] = tif

    return granules



# Given one GEDI shot (lon, lat), figure out which HLS raster file (granule) contains it, and what pixel (row, col) it falls on.
def find_granule_for_point(lon, lat, granule_dict, preferred_band="B02"):
    """
    Simple: loop over granules and return the first whose bounding box
    contains the (lon, lat) point (reprojected to that granule CRS),
    and the corresponding (row, col) in its raster grid.
    """
        #     granule_dict is like:

        # {
        #   "HLS.L30.T10TEK.2021096...": {"B02": "...tif", "B03": "...tif", ...},
        #   "HLS.L30.T10TEK.2021103...": {"B02": "...tif", "B03": "...tif", ...},
        # }
        # {"B02": "/.../B02.tif", "B03": "/.../B03.tif", ...}

    for granule_id, bands in granule_dict.items():
        # Reference band to check geometry
        if preferred_band in bands:
            ref_path = bands[preferred_band]
            # getting the tif file ref path 
        else:
            ref_path = list(bands.values())[0]

        with rasterio.open(ref_path) as src:
            # Reproject lon/lat (EPSG:4326) → raster CRS
            #             GEDI is in lat/lon. But HLS rasters are usually in a projected CRS (UTM).
            # So you must reproject the point into the raster coordinate system before checking bounds or computing pixel indices.
            xs, ys = rio_transform("EPSG:4326", src.crs, [lon], [lat])
            x, y = xs[0], ys[0]

            # Check if point lies in raster bounds
            if (src.bounds.left <= x <= src.bounds.right and
                src.bounds.bottom <= y <= src.bounds.top):

                row, col = src.index(x, y)
                return granule_id, row, col

    return None, None, None

def _init_worker(granule_dict, nlcd_path, s1_index, max_s1_days_diff, s1_to_db):
    """
    Runs once per worker process.
    Keeps HLS index, NLCD source, and Sentinel-1 index available inside each worker.
    """
    global _GRANULE_DICT, _NLCD_SRC, _S1_INDEX, _MAX_S1_DAYS_DIFF, _S1_TO_DB

    _GRANULE_DICT = granule_dict
    _S1_INDEX = s1_index
    _MAX_S1_DAYS_DIFF = max_s1_days_diff
    _S1_TO_DB = s1_to_db

    _NLCD_SRC = None
    if nlcd_path is not None:
        _NLCD_SRC = rasterio.open(nlcd_path)
        atexit.register(_NLCD_SRC.close)

def _process_one_shot(task):
    """
    Worker: does everything except writing the output patch file.
    Returns: None (skip) OR a tuple needed for writing + CSV row.
    """
    i, row = task

    try:
        lat = float(row["lat"])
        lon = float(row["lon"])
        agbd = float(row["agbd"])
        shot_number = int(row["shot_number"])
        gedi_file = row["gedi_file"]

        # label scaling (your current logic)
        HLS_PIXEL_AREA_M2 = 30.0 * 30.0
        CENTER_OVERLAP_M2 = 400.0
        frac = CENTER_OVERLAP_M2 / HLS_PIXEL_AREA_M2
        agbd_center = agbd * frac

        granule_id, r, c = find_granule_for_point(
            lon, lat, _GRANULE_DICT, preferred_band="B02"
        )
        if granule_id is None:
            return None
        
        sensor = "S30" if granule_id.split(".")[1] == "S30" else "L30"

        # Match Sentinel-1 scene using the HLS granule date and GEDI location
        hls_time = parse_hls_datetime(granule_id)

        s1_scene = None
        s1_days_diff = None

        if _S1_INDEX is not None:
            s1_scene = find_s1_scene_for_point(
                lon=lon,
                lat=lat,
                s1_index=_S1_INDEX,
                ref_datetime=hls_time,
                max_days_diff=_MAX_S1_DAYS_DIFF,
            )

            # If Sentinel-1 is required but no matching scene is found, skip this sample.
            if s1_scene is None:
                print("SKIP: no matching Sentinel-1 scene", lon, lat, hls_time)
                return None

        cube, used_bands, transform, crs = extract_3x3_patch_for_all_bands(
            _GRANULE_DICT[granule_id],
            r,
            c,
            sensor,
            nlcd_src=_NLCD_SRC,
            s1_scene=s1_scene,
            s1_to_db=_S1_TO_DB,
        )

        if cube is None:
            return None
        
        
       

        patch_id = f"{shot_number}_{granule_id}"

        # Make CRS robust for pickling
        crs_wkt = crs.to_wkt() if crs is not None else None

        meta = {
            "gedi_file": gedi_file,
            "shot_number": shot_number,
            "lat": lat,
            "lon": lon,
            "agbd_raw": agbd,
            "agbd_center": float(agbd_center),
            "agbd_log": float(np.log1p(agbd_center)),
            "center_frac": float(frac),
            "hls_granule": granule_id,
            "row": int(r),
            "col": int(c),
            "sentinel1_scene": s1_scene["safe_name"] if s1_scene is not None else None,
            "sentinel1_time": s1_scene["acq_time"].isoformat() if s1_scene is not None and s1_scene["acq_time"] is not None else None,
            "sentinel1_days_diff": None,
            # patch_tifs + bands will be filled by main process (deterministic)
        }

        return (i, patch_id, cube, used_bands, transform, crs_wkt, meta)

    except Exception as e:
        print(f"Error processing shot {i}: {e}", flush=True)
        # Don’t crash the whole job; just skip and optionally print
        # (You can log e + i if you want)
        return None


def build_gedi_hls_patches_multiprocess(
    gedi_folder,
    hls_root,
    patches_root="field_boundary_patch_tifs",
    out_csv_path="gedi_hls_metadata.csv",
    min_quality=1,
    nlcd_path=None,

    s1_root=None,
    max_s1_days_diff=12,
    s1_to_db=False,

    max_workers=None,
    chunksize=200,
):
    """
    Same outputs as build_gedi_hls_patches(), but parallelizes per-shot extraction.
    Writes patches + CSV in the main process to keep output deterministic.
    """

    gedi_folder = Path(gedi_folder)
    hls_root = Path(hls_root)
    patches_root = Path(patches_root)
    out_csv_path = Path(out_csv_path)

    # CLEAN START (same behavior as your script)
    if patches_root.exists():
        shutil.rmtree(patches_root)
    patches_root.mkdir(parents=True, exist_ok=True)

    if out_csv_path.exists():
        out_csv_path.unlink()

    # 1) Load GEDI shots (same as your script)
    print("Loading GEDI shots from folder...")
    gedi_df = load_all_gedi_from_folder(gedi_folder, min_quality=min_quality)
    print(f"Total good-quality shots across all files: {len(gedi_df)}")

    # DOY filter: keep only shots from April–August (DOY 91–240) to match HLS seasonal window
    # HLS data covers April–August only; winter/fall GEDI shots matched to summer imagery add noise
    MIN_DOY, MAX_DOY = 91, 240
    n_before = len(gedi_df)
    gedi_df = gedi_df[(gedi_df["gedi_doy"] >= MIN_DOY) & (gedi_df["gedi_doy"] <= MAX_DOY)].reset_index(drop=True)
    print(f"After DOY filter ({MIN_DOY}–{MAX_DOY}): {len(gedi_df)} shots kept, {n_before - len(gedi_df)} removed")

    # 2) Field polygon filtering (same as your script)
    field = GEOJSON_FILE
    field_geom = field.geometry.unary_union

    gedi_gdf = gpd.GeoDataFrame(
        gedi_df,
        geometry=gpd.points_from_xy(gedi_df.lon, gedi_df.lat),
        crs="EPSG:4326",
    )
    gedi_in_field = gedi_gdf[gedi_gdf.within(field_geom)].reset_index(drop=True)
    print("Shots inside field polygon:", len(gedi_in_field))
    gedi_df = gedi_in_field.drop(columns="geometry")

    if len(gedi_df) == 0:
        print("No GEDI shots found. Check folder / file names.")
        return

    # 3) Index HLS granules (same)
    print("Indexing HLS granules and bands...")
    granule_dict = index_hls_bands(hls_root)
    print(f"Total HLS granules found: {len(granule_dict)}")

    # 4. Index Sentinel-1 scenes
    s1_index = None

    if s1_root is not None:
        print("Indexing Sentinel-1 scenes...")
        s1_index = index_sentinel1_scenes(s1_root)

        if len(s1_index) == 0:
            raise RuntimeError("No valid Sentinel-1 VV/VH scenes found. Check S1_ROOT path.")

    # Decide workers (you have 32 logical CPUs; don’t max it out by default)
    if max_workers is None:
        cpu = os.cpu_count() or 1
        max_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "14"))

    # Prepare tasks in stable order
    records = gedi_df.to_dict("records")
    tasks = list(enumerate(records))

    meta_rows = []

    # Use spawn context (safer with GDAL/rasterio on clusters)
    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(granule_dict, nlcd_path, s1_index, max_s1_days_diff, s1_to_db),
    ) as ex:

        # executor.map preserves input order => deterministic output order
        results_iter = ex.map(_process_one_shot, tasks, chunksize=chunksize)

        for res in tqdm(results_iter, total=len(tasks)):
            if res is None:
                continue

            i, patch_id, cube, used_bands, transform, crs_wkt, meta = res

            patch_path = patches_root / f"patch_{patch_id}.tif"

            # Rebuild CRS object
            crs = rasterio.crs.CRS.from_wkt(crs_wkt) if crs_wkt else None

            # Write patch GeoTIFF in MAIN process (deterministic)
            C, H, W = cube.shape
            with rasterio.open(
                patch_path,
                "w",
                driver="GTiff",
                height=H,
                width=W,
                count=C,
                dtype=cube.dtype,
                crs=crs,
                transform=transform,
            ) as dst:
                for band_idx in range(C):
                    dst.write(cube[band_idx, :, :], band_idx + 1)

            # Fill CSV fields that depend on the final written file / ordering
            meta["patch_tifs"] = str(patch_path)
            meta["bands"] = ",".join(used_bands)

            meta_rows.append(meta)

    # Save CSV (same)
    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(out_csv_path, index=False)
    print(f"Saved metadata CSV: {out_csv_path}")
    print(f"Total samples with patches: {len(meta_df)}")


def main():

    print("HLS exists:", Path(HLS_FOLDER).exists())
    print("HLS tif count:", len(list(Path(HLS_FOLDER).glob("*.tif"))))
    print("First 5 tif files:", list(Path(HLS_FOLDER).glob("*.tif"))[:5])
    # this method is doing all the jobs
    build_gedi_hls_patches_multiprocess(
    gedi_folder=GEDI_FOLDER,
    hls_root=HLS_FOLDER,
    patches_root=PATCHES_ROOT,
    out_csv_path=OUT_CSV_PATH,
    min_quality=1,
    nlcd_path=NLCD_PATH,

    s1_root=S1_ROOT,
    max_s1_days_diff=MAX_S1_DAYS_DIFF,
    s1_to_db=S1_TO_DB,

    max_workers=MAX_WORKERS,     # start here on your 32 logical CPUs
    chunksize=CHUNK_SIZE
    
    )


if __name__ == "__main__":
    main()
