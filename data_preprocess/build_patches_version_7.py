


# Brief: 
# Building 5x5 patches for HLS (8 bands) + NLCD (1 band) + Sentinel-1 (VV, VH) = 11 bands total. center pixel
# (1) Sentinel-1 SAR enabled
#     Aug 18-31 2021, so day-gap matching against April-August GEDI/HLS shots caused

#     99.9% of HLS granule dates have a same-day S1 acquisition, 100% within 1 day.
# (2) MAX_S1_DAYS_DIFF tightened from 12 -> 3 days

#     rather than allowing stale multi-week SAR to be matched to a footprint.
# (3) Auto-unzip: index_sentinel1_scenes only sees already-extracted *.SAFE folders.

#     incremental CSV flush every 500 patches.

# gedi filter 
# min quality =1  
# degrade = 0



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
from rasterio.crs import CRS

import shutil
import os
import atexit
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import re
import zipfile
from datetime import datetime

# -------------------------
# Globals set in each worker
# -------------------------
_GRANULE_DICT = None
_NLCD_SRC = None
_S1_INDEX = None
_MAX_S1_DAYS_DIFF = None
_S1_TO_DB = False

# 5x5 patch: half-width = 2 pixels on each side of center
PATCH_SIZE = 5
HALF = PATCH_SIZE // 2   # = 2

# Flush CSV every this many rows to survive preemption
_CSV_FLUSH_EVERY = 500

############################ all the paths here — edit as needed
GEDI_FOLDER = r"/s/chopin/e/proj/hyperspec/masfiq/dataset/gedi_l4a_california_north_10_2021_whole_year"

HLS_FOLDER = r"/s/chopin/e/proj/hyperspec/masfiq/dataset/hls_data_California_north_10_2021_AprilToAugust"

S1_ROOT = r"/s/chopin/e/proj/hyperspec/masfiq/dataset/sentinel_1_data_california_north_10_2021_April_to_August"

MAX_S1_DAYS_DIFF = 3

# Keep False: patches store raw linear backscatter, not dB. Matches how the archive's
# original 13-scene subset was used at train time (Sentinel1Lookup's own log1p-based
# normalization there assumed non-negative linear values).
S1_TO_DB = False

NLCD_PATH = r"/s/chopin/e/proj/hyperspec/masfiq/dataset/Annual_NLCD_LndCov_2021_CU_C1V1/Annual_NLCD_LndCov_2021_CU_C1V1.tif"

PATCHES_ROOT = "/s/chopin/e/proj/hyperspec/masfiq/dataset/patch_tifs_california_north_10_2021_AprilToAugust_version_6"

OUT_CSV_PATH = "/s/chopin/e/proj/hyperspec/masfiq/csv_files/gedi_california_north_10_2021_AprilToAugust_version_6.csv"

MAX_WORKERS = 8
CHUNK_SIZE = 10

GEOJSON_FILE = gpd.read_file("/s/chopin/e/proj/hyperspec/masfiq/biomass_estimation_test/geojson_files/california_north_10.geojson")

##################################################
def parse_hls_datetime(granule_id):
    for part in granule_id.split("."):
        if re.fullmatch(r"\d{7}T\d{6}", part):
            return datetime.strptime(part, "%Y%jT%H%M%S")
    return None


def parse_s1_datetime_from_safe_name(safe_name):
    match = re.search(r"_(\d{8}T\d{6})_", safe_name)
    if match is None:
        return None
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")


def find_vv_vh_files(measurement_folder):
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


def ensure_sentinel1_unzipped(s1_root):
    """
    The bulk-download archive lands as *.zip (one SAFE product per zip). Only
    already-extracted *.SAFE folders are visible to index_sentinel1_scenes, so a
    partially-unzipped archive would silently look like the whole archive and most
    GEDI shots would fail to find a matching S1 scene. Unzip anything missing its
    .SAFE folder before indexing. Idempotent: already-extracted archives are skipped.
    """
    s1_root = Path(s1_root)
    zips = sorted(s1_root.glob("*.zip"))
    if not zips:
        return

    to_extract = [z for z in zips if not (s1_root / f"{z.stem}.SAFE").exists()]
    if not to_extract:
        print(f"[S1] All {len(zips)} archives already unzipped.")
        return

    print(f"[S1] Unzipping {len(to_extract)} of {len(zips)} Sentinel-1 archives ...")
    for z in tqdm(to_extract, desc="Unzipping S1 archives"):
        try:
            with zipfile.ZipFile(z, "r") as zf:
                zf.extractall(s1_root)
        except (zipfile.BadZipFile, OSError) as e:
            print(f"[S1] WARNING: could not unzip {z.name}: {e}")


def _get_spatial_ref(src):
    """
    Resolve (crs, transform) for a source dataset, falling back to GCP-based
    georeferencing when no direct CRS is present. Sentinel-1 GRD-HD COG products
    from ASF are the common case that needs this fallback: src.crs is None and
    src.transform is the identity matrix, with only a sparse GCP grid available.
    """
    if src.crs is not None and str(src.crs).strip() != "":
        return src.crs, src.transform
    gcps, gcp_crs = src.gcps
    if gcp_crs is not None and len(gcps) > 0:
        return gcp_crs, rasterio.transform.from_gcps(gcps)
    b = src.bounds
    if (-180 <= b.left <= 180 and -180 <= b.right <= 180 and
            -90 <= b.bottom <= 90 and -90 <= b.top <= 90):
        return CRS.from_epsg(4326), src.transform
    raise ValueError(f"No usable CRS or GCP CRS for {src.name}")


def index_sentinel1_scenes(s1_root):
    s1_root = Path(s1_root)

    ensure_sentinel1_unzipped(s1_root)

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
                scene_crs, _ = _get_spatial_ref(src)

                # Bounds for the fast point-in-scene pre-filter. When georeferencing
                # comes from GCPs, take the min/max of the actual GCP lon/lat directly
                # -- more robust for a coarse footprint check than round-tripping
                # through the approximate affine that from_gcps() would produce.
                gcps, gcp_crs = src.gcps
                if gcp_crs is not None and len(gcps) > 0 and scene_crs == gcp_crs:
                    xs = [g.x for g in gcps]
                    ys = [g.y for g in gcps]
                    bounds = (min(xs), min(ys), max(xs), max(ys))
                else:
                    bounds = tuple(src.bounds)

                scene = {
                    "safe_name": safe_dir.name,
                    "safe_path": str(safe_dir),
                    "vv_path": str(vv_file),
                    "vh_path": str(vh_file),
                    "acq_time": acq_time,
                    "crs_wkt": scene_crs.to_wkt() if scene_crs is not None else None,
                    "bounds": bounds,
                }
                scenes.append(scene)
        except Exception as e:
            print(f"Could not read Sentinel-1 scene {safe_dir.name}: {e}")
    print(f"Total Sentinel-1 scenes indexed: {len(scenes)}")
    return scenes


def find_s1_scene_for_point(lon, lat, s1_index, ref_datetime=None, max_days_diff=12):
    """
    Returns (scene, days_diff). days_diff is None when no reference date is available
    or no spatial match was found.
    """
    if s1_index is None:
        return None, None
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
        if ref_datetime is not None and scene["acq_time"] is not None:
            diff = abs((scene["acq_time"] - ref_datetime).days)
            if diff > max_days_diff:
                continue
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_scene = scene
        else:
            return scene, None
    return best_scene, best_diff


def reproject_s1_band_to_hls_patch(s1_tif_path, dst_transform, dst_crs, to_db=False):
    """Reproject one Sentinel-1 band into the 5x5 HLS patch grid."""
    s1_patch = np.full((PATCH_SIZE, PATCH_SIZE), np.nan, dtype=np.float32)
    with rasterio.open(s1_tif_path) as src:
        if src.crs is not None and str(src.crs).strip() != "":
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
        else:
            # No direct CRS: pass GCPs straight to GDAL's warp machinery rather than
            # approximating a single affine transform first. GDAL's native GCP
            # transformer handles the swath's geometric distortion far more accurately
            # than a first-order affine fit across a 250km-wide scene.
            gcps, gcp_crs = src.gcps
            reproject(
                source=rasterio.band(src, 1),
                destination=s1_patch,
                gcps=gcps,
                src_crs=gcp_crs,
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

# input gedi h5 file
# output a panda dataframe  

def load_gedi_shots_from_file(h5_path, min_quality=1):
    h5_path = Path(h5_path)
    rows = []
    doy_match = re.search(r'GEDI04_A_\d{4}(\d{3})', h5_path.name)
    gedi_doy = int(doy_match.group(1)) if doy_match else -1
    with h5py.File(h5_path, "r") as f:
        for beam_name in f.keys():
            if not beam_name.startswith("BEAM"):
                continue
            beam = f[beam_name]
            try:
                lat = beam["lat_lowestmode"][:]
                lon = beam["lon_lowestmode"][:]
                agbd = beam["agbd"][:]
                quality = beam["l4_quality_flag"][:]
                shot_number = beam["shot_number"][:]
                degrade = beam["degrade_flag"][:]
                # v7: recorded, NOT filtered on. See the note above the rows.append below.
                agbd_se = beam["agbd_se"][:]
                sensitivity = beam["sensitivity"][:]
            except KeyError:
                print(f"Warning: missing vars in {h5_path.name}, {beam_name}")
                continue
            # v7 filter changes vs v6:
            #   agbd > 0      replaces np.isfinite(agbd). GEDI marks missing biomass with
            #                 -9999, not NaN, and np.isfinite(-9999) is True, so the old
            #                 check removed exactly zero rows -- only the quality flag was
            #                 keeping fill values out. This is the guard it looked like.
            #   degrade == 0  new. Measured over 10 sampled granules, 35.3% of the shots
            #                 passing l4_quality_flag alone carry a non-zero degrade_flag
            #                 (25.0% are 80, 8.3% are 30, 2.0% are 70), meaning GEDI's own
            #                 pointing/orbit solution was degraded and the 25m footprint may
            #                 not be where the file says it is. Requiring both flags is
            #                 standard practice in the L4A literature.
            mask = (quality >= min_quality) & (agbd > 0) & (degrade == 0)

            lat = lat[mask]
            lon = lon[mask]
            agbd = agbd[mask]
            shot_number = shot_number[mask]
            quality = quality[mask]
            agbd_se = agbd_se[mask]
            sensitivity = sensitivity[mask]

            # agbd_se and sensitivity are RECORDED ONLY -- they are never filtered on and
            # must never become model inputs. Both exist only where GEDI actually flew, so
            # feeding either to the network would leak information about the label and
            # leave an empty slot when predicting anywhere GEDI did not sample.
            #
            #   agbd_se      Standard error of GEDI's own biomass prediction, in Mg/ha
            #                (median ~11.1 inside this ROI). Two legitimate uses: as an
            #                error floor when reporting accuracy ("MAE 37.9 against a label
            #                uncertainty of 11.1"), and as a per-sample loss weight. Note
            #                plain inverse-variance weighting (1/se^2) BACKFIRES here: se
            #                tracks prediction stratum, not data quality (shrub ~3, forest
            #                ~11), so it would hand 67.7% of total influence to the 0-25
            #                Mg/ha band and drop the 200+ band to 4.5%. Weight within
            #                stratum if weighting at all.
            #
            #   sensitivity  Max canopy cover at which this shot could still detect the
            #                ground return. Diagnostic only: lets us test whether large
            #                residuals coincide with weak measurements, and lets us compare
            #                shot difficulty between the north train ROI and the south test
            #                strip. Do NOT filter on it -- it correlates strongly with
            #                biomass (mean AGBD 47 at 0.95-0.96 vs 210 at 0.99+), so a
            #                >0.98 cut would delete 82% of shots and skew the set high.
            for la, lo, a, q, sn, se, sens in zip(
                lat, lon, agbd, quality, shot_number, agbd_se, sensitivity
            ):
                rows.append(
                    {
                        "gedi_file": h5_path.name,
                        "beam": beam_name,
                        "shot_number": int(sn),
                        "lat": float(la),
                        "lon": float(lo),
                        "agbd": float(a),
                        "agbd_se": float(se),
                        "sensitivity": float(sens),
                        "l4_quality_flag": int(q),
                        "gedi_doy": gedi_doy,
                    }
                )
    return pd.DataFrame(rows)


# getting all gedi shots from a file 

def load_all_gedi_from_folder(gedi_folder, min_quality=1):
    gedi_folder = Path(gedi_folder)
    dfs = []
    for h5_file in sorted(gedi_folder.glob("*.h5")):
        print(f"Reading {h5_file.name} ...")
        # passing each h5 file and getting shot from them 
        df_file = load_gedi_shots_from_file(h5_file, min_quality=min_quality)
        dfs.append(df_file)
    if len(dfs) == 0:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def extract_5x5_patch_for_all_bands(
    granule_bands,
    row,
    col,
    sensor,
    nlcd_src=None,
    s1_scene=None,
    s1_to_db=False,
):
    """
    Read 5x5 HLS patch centered on (row, col) and optionally add NLCD + Sentinel-1.
    We still record the CENTER pixel biomass — only the spatial window is wider.

    Final output (with SAR): 8 HLS bands + 1 NLCD + 2 Sentinel-1 (VV, VH) = 11 bands,
    each 5x5. Band order: HLS bands, then NLCD, then S1_VV, S1_VH — matches the
    FIXED_BANDS ordering already used by the training scripts
    (..., "EVI", "NDVI", "NLCD", "S1_VV", "S1_VH").
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

    # 1. HLS bands — 5x5 window
    for b in band_codes:
        if b not in granule_bands:
            return None, None, None, None

        path = granule_bands[b]

        with rasterio.open(path) as src:
            h, w = src.height, src.width

            # Need HALF pixels of margin on every side
            if not (HALF <= row < h - HALF and HALF <= col < w - HALF):
                return None, None, None, None

            if window is None:
                window = Window(
                    col_off=col - HALF,
                    row_off=row - HALF,
                    width=PATCH_SIZE,
                    height=PATCH_SIZE,
                )
                transform = src.window_transform(window)
                crs = src.crs

            patch = src.read(1, window=window).astype(np.float32)
            patches.append(patch)
            used_bands.append(b)

    if len(patches) == 0:
        return None, None, None, None

    # 2. NLCD band — reprojected to same 5x5 grid
    if nlcd_src is not None and window is not None and transform is not None and crs is not None:
        nlcd_patch = np.full((PATCH_SIZE, PATCH_SIZE), fill_value=-1, dtype=np.float32)
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

    # 3. Sentinel-1 VV and VH
    if s1_scene is not None:
        vv_patch = reproject_s1_band_to_hls_patch(
            s1_scene["vv_path"], dst_transform=transform, dst_crs=crs, to_db=s1_to_db,
        )
        vh_patch = reproject_s1_band_to_hls_patch(
            s1_scene["vh_path"], dst_transform=transform, dst_crs=crs, to_db=s1_to_db,
        )
        if np.all(np.isnan(vv_patch)) or np.all(np.isnan(vh_patch)):
            return None, None, None, None
        patches.append(vv_patch)
        used_bands.append("S1_VV")
        patches.append(vh_patch)
        used_bands.append("S1_VH")

    data_cube = np.stack(patches, axis=0).astype(np.float32)
    return data_cube, used_bands, transform, crs


def index_hls_bands(hls_root):
    hls_root = Path(hls_root)
    granules = defaultdict(dict)
    for tif in hls_root.glob("*.tif"):
        stem = tif.stem
        stem = stem.replace("_cropped", "")
        parts = stem.replace("_", ".").split(".")
        if len(parts) < 7:
            continue
        granule_id = ".".join(parts[:-1])
        band_code = parts[-1]
        granules[granule_id][band_code] = tif
    return granules


def find_granule_for_point(lon, lat, granule_dict, preferred_band="B02"):
    for granule_id, bands in granule_dict.items():
        if preferred_band in bands:
            ref_path = bands[preferred_band]
        else:
            ref_path = list(bands.values())[0]
        with rasterio.open(ref_path) as src:
            xs, ys = rio_transform("EPSG:4326", src.crs, [lon], [lat])
            x, y = xs[0], ys[0]
            if (src.bounds.left <= x <= src.bounds.right and
                    src.bounds.bottom <= y <= src.bounds.top):
                row, col = src.index(x, y)
                return granule_id, row, col
    return None, None, None


def _init_worker(granule_dict, nlcd_path, s1_index, max_s1_days_diff, s1_to_db):
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
    i, row = task
    try:
        lat = float(row["lat"])
        lon = float(row["lon"])
        agbd = float(row["agbd"])
        shot_number = int(row["shot_number"])
        gedi_file = row["gedi_file"]
        # v7: carried through from load_gedi_shots_from_file. Recorded only, never filtered
        # on and never a model input -- see the note in that function.
        agbd_se = float(row.get("agbd_se", float("nan")))
        sensitivity = float(row.get("sensitivity", float("nan")))

        # Still predicting center pixel biomass, same as v3/v4/v5
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

        hls_time = parse_hls_datetime(granule_id)

        s1_scene = None
        s1_days_diff = None
        if _S1_INDEX is not None:
            s1_scene, s1_days_diff = find_s1_scene_for_point(
                lon=lon, lat=lat, s1_index=_S1_INDEX,
                ref_datetime=hls_time, max_days_diff=_MAX_S1_DAYS_DIFF,
            )
            if s1_scene is None:
                print("SKIP: no matching Sentinel-1 scene", lon, lat, hls_time)
                return None

        cube, used_bands, transform, crs = extract_5x5_patch_for_all_bands(
            _GRANULE_DICT[granule_id],
            r, c, sensor,
            nlcd_src=_NLCD_SRC,
            s1_scene=s1_scene,
            s1_to_db=_S1_TO_DB,
        )

        if cube is None:
            return None

        patch_id = f"{shot_number}_{granule_id}"
        crs_wkt = crs.to_wkt() if crs is not None else None

        meta = {
            "gedi_file": gedi_file,
            "shot_number": shot_number,
            "lat": lat,
            "lon": lon,
            "agbd_raw": agbd,
            "agbd_center": float(agbd_center),
            "agbd_log": float(np.log1p(agbd_center)),
            "agbd_sqrt": float(np.sqrt(agbd_center)),
            "center_frac": float(frac),
            # GEDI's own uncertainty on agbd_raw, in Mg/ha (median ~11.1 in this ROI).
            "agbd_se": agbd_se,
            # The SAME uncertainty rescaled by center_frac so it is in the units the model
            # is actually trained and scored in. agbd_center = agbd_raw * frac, and scaling
            # is linear, so the matching standard error is agbd_se * frac. Comparing a
            # raw agbd_se against an MAE measured on agbd_center mixes two scales and
            # overstates the label-noise share by 1/frac = 2.25x.
            "agbd_se_center": agbd_se * float(frac),
            # Max canopy cover at which this shot could still find the ground. Diagnostic
            # only: never filter on it, it correlates strongly with biomass.
            "sensitivity": sensitivity,
            "hls_granule": granule_id,
            "row": int(r),
            "col": int(c),
            "sentinel1_scene": s1_scene["safe_name"] if s1_scene is not None else None,
            "sentinel1_time": s1_scene["acq_time"].isoformat() if s1_scene is not None and s1_scene["acq_time"] is not None else None,
            "sentinel1_days_diff": s1_days_diff,
        }

        return (i, patch_id, cube, used_bands, transform, crs_wkt, meta)

    except Exception as e:
        print(f"Error processing shot {i}: {e}", flush=True)
        return None


def build_gedi_hls_patches_multiprocess(
    gedi_folder,
    hls_root,
    patches_root="patch_tifs_version_6",
    out_csv_path="gedi_hls_metadata_version_6.csv",
    min_quality=1,
    nlcd_path=None,
    s1_root=None,
    max_s1_days_diff=12,
    s1_to_db=False,
    max_workers=None,
    chunksize=20,
):
    gedi_folder = Path(gedi_folder)
    hls_root = Path(hls_root)
    patches_root = Path(patches_root)
    out_csv_path = Path(out_csv_path)

    # Resume support: find shots already processed so we skip them
    existing_shots: set = set()

    # if the root exists  and scan the output folder and pull the shot number out of each filename
    # patch_130340000300182824_HLS.L30.T10TDM.2021213T190305.v2.0.tif
    #       └────────┬───────┘
    #                └─ the regex patch_(\d+)_ captures this
    if patches_root.exists():
        for p in patches_root.glob("patch_*.tif"):
            m = re.match(r"patch_(\d+)_", p.name)
            if m:
                existing_shots.add(int(m.group(1)))
    if existing_shots:
        print(f"Resuming: found {len(existing_shots)} existing patches, will skip them")
    patches_root.mkdir(parents=True, exist_ok=True)

    # 1) Load GEDI shots
    print("Loading GEDI shots from folder...")



    #     # edi_file      beam         shot_number        lat         lon       agbd  l4_quality_flag  gedi_doy
    # GEDI04_A_2021003083948_...V002.h5  BEAM0000  116730000200075534  44.502458 -114.480250  20.413515                1         3
    # GEDI04_A_2021003083948_...V002.h5  BEAM0000  116730000200075546  44.505643 -114.472909  30.998112                1         3
    gedi_df = load_all_gedi_from_folder(gedi_folder, min_quality=min_quality)
    print(f"Total good-quality shots across all files: {len(gedi_df)}")

    # DOY filter: April–August (DOY 91–240) to match HLS seasonal window
    MIN_DOY, MAX_DOY = 91, 240
    n_before = len(gedi_df)
    gedi_df = gedi_df[
        (gedi_df["gedi_doy"] >= MIN_DOY) & (gedi_df["gedi_doy"] <= MAX_DOY)
    ].reset_index(drop=True)
    print(f"After DOY filter ({MIN_DOY}–{MAX_DOY}): {len(gedi_df)} shots kept, {n_before - len(gedi_df)} removed")

    # 2) Field polygon filtering
    field_geom = GEOJSON_FILE.geometry.unary_union
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

    # 3) Index HLS granules
    print("Indexing HLS granules and bands...")
    granule_dict = index_hls_bands(hls_root)
    print(f"Total HLS granules found: {len(granule_dict)}")

    # 4) Index Sentinel-1 scenes (unzips missing archives first — see ensure_sentinel1_unzipped)
    s1_index = None
    if s1_root is not None:
        print("Indexing Sentinel-1 scenes...")
        s1_index = index_sentinel1_scenes(s1_root)
        if len(s1_index) == 0:
            raise RuntimeError("No valid Sentinel-1 VV/VH scenes found. Check S1_ROOT path.")

    if max_workers is None:
        max_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "14"))

    records = gedi_df.to_dict("records")

    # Skip shots already processed
    tasks = [
        (i, r) for i, r in enumerate(records)
        if int(r["shot_number"]) not in existing_shots
    ]
    print(f"Tasks to process: {len(tasks)} (skipped {len(records) - len(tasks)} already done)")

    meta_rows = []
    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(granule_dict, nlcd_path, s1_index, max_s1_days_diff, s1_to_db),
    ) as ex:

        results_iter = ex.map(_process_one_shot, tasks, chunksize=chunksize)

        for res in tqdm(results_iter, total=len(tasks)):
            if res is None:
                continue

            i, patch_id, cube, used_bands, transform, crs_wkt, meta = res

            patch_path = patches_root / f"patch_{patch_id}.tif"
            crs = rasterio.crs.CRS.from_wkt(crs_wkt) if crs_wkt else None

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

            meta["patch_tifs"] = str(patch_path)
            meta["bands"] = ",".join(used_bands)
            meta_rows.append(meta)

            # Incremental CSV flush to survive preemption
            if len(meta_rows) >= _CSV_FLUSH_EVERY:
                df_chunk = pd.DataFrame(meta_rows)
                write_header = not out_csv_path.exists()
                df_chunk.to_csv(out_csv_path, mode="a", header=write_header, index=False)
                meta_rows = []

    # Flush remaining rows
    if meta_rows:
        df_chunk = pd.DataFrame(meta_rows)
        write_header = not out_csv_path.exists()
        df_chunk.to_csv(out_csv_path, mode="a", header=write_header, index=False)

    total = sum(1 for _ in out_csv_path.open()) - 1 if out_csv_path.exists() else 0
    print(f"Saved metadata CSV: {out_csv_path}")
    print(f"Total samples with patches: {total}")


def main():
    #measuring the hlls pixel
    print(f"Patch size: {PATCH_SIZE}x{PATCH_SIZE} ({PATCH_SIZE * 30}m x {PATCH_SIZE * 30}m)")

    # check if hls exists 
    print("HLS exists:", Path(HLS_FOLDER).exists())
    #getting all the ti files
    print("HLS tif count:", len(list(Path(HLS_FOLDER).glob("*.tif"))))
    print("First 5 tif files:", list(Path(HLS_FOLDER).glob("*.tif"))[:5])

    # check s1 root 
    print("S1 root exists:", Path(S1_ROOT).exists() if S1_ROOT else None)

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
        max_workers=MAX_WORKERS,
        chunksize=CHUNK_SIZE,
    )


if __name__ == "__main__":
    main()
