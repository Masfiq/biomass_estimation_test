# Building 3x3 patches for HLS (8 bands) + NLCD (1 band) = 9 bands total. No SAR bands in this version.
# v3 changes vs v2:
# (1) DOY filter keeps only April–August GEDI shots (DOY 91–240) to match HLS seasonal window
# (2) agbd_log column added to CSV (log1p of agbd_center) alongside agbd_raw and agbd_center
# (3) Sentinel-1 removed — S1 matching caused 0 samples due to scene coverage issues; will be added in v4

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import geopandas as gpd

import rasterio
from rasterio.windows import Window
from rasterio.warp import transform as rio_transform, reproject, transform_bounds
from rasterio.transform import rowcol as rasterio_rowcol
from rasterio.enums import Resampling

import shutil
import os
import atexit
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import re

# -------------------------
# Globals set in each worker
# -------------------------
_GRANULE_DICT = None
_GRANULE_META = None   # pre-cached bounds/CRS per granule — avoids file opens in the per-shot loop
_NLCD_SRC = None

############################ all the paths here # edit
GEDI_FOLDER = r"/s/chopin/e/proj/hyperspec/masfiq/dataset/gedi_l4a_california_north_10_2021_whole_year"

HLS_FOLDER = r"/s/chopin/e/proj/hyperspec/masfiq/dataset/hls_data_California_north_10_2021_AprilToAugust"

NLCD_PATH = r"/s/chopin/e/proj/hyperspec/masfiq/dataset/Annual_NLCD_LndCov_2021_CU_C1V1/Annual_NLCD_LndCov_2021_CU_C1V1.tif"

PATCHES_ROOT = "/s/chopin/e/proj/hyperspec/masfiq/dataset/patch_tifs_california_north_10_2021_AprilToAugust_version_3"

OUT_CSV_PATH = "/s/chopin/e/proj/hyperspec/masfiq/csv_files/gedi_california_north_10_2021_AprilToAugust_version_3.csv"

MAX_WORKERS = 4

CHUNK_SIZE = 10

GEOJSON_FILE = gpd.read_file("/s/chopin/e/proj/hyperspec/masfiq/biomass_estimation_test/geojson_files/california_north_10.geojson")

##################################################
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
            except KeyError:
                print(f"Warning: missing vars in {h5_path.name}, {beam_name}")
                continue

            mask = (quality >= min_quality) & np.isfinite(agbd)
            lat = lat[mask]
            lon = lon[mask]
            agbd = agbd[mask]
            shot_number = shot_number[mask]
            quality = quality[mask]

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
    return pd.DataFrame(rows)


def load_all_gedi_from_folder(gedi_folder, min_quality=1):
    gedi_folder = Path(gedi_folder)
    dfs = []
    for h5_file in sorted(gedi_folder.glob("*.h5")):
        print(f"Reading {h5_file.name} ...")
        df_file = load_gedi_shots_from_file(h5_file, min_quality=min_quality)
        dfs.append(df_file)

    if len(dfs) == 0:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def extract_3x3_patch_for_all_bands(granule_bands, row, col, sensor, nlcd_src=None):
    """
    Read 3x3 HLS patch and add NLCD.
    Final output: 8 HLS bands + 1 NLCD = 9 bands
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

    data_cube = np.stack(patches, axis=0).astype(np.float32)

    return data_cube, used_bands, transform, crs


def index_hls_bands(hls_root):
    """
    Groups all HLS TIFs in a folder by granule ID and pre-caches each granule's
    bounds (in WGS84) and CRS so that find_granule_for_point needs zero file I/O.
    Returns (granule_dict, granule_meta).
    """
    hls_root = Path(hls_root)
    granules = defaultdict(dict)

    for tif in hls_root.glob("*.tif"):
        stem = tif.stem.replace("_cropped", "")
        parts = stem.replace("_", ".").split(".")
        if len(parts) < 7:
            continue
        granule_id = ".".join(parts[:-1])
        band_code  = parts[-1]
        granules[granule_id][band_code] = tif

    # Pre-read bounds and CRS for every granule — one file open per granule, done once.
    granule_meta = {}
    preferred_band = "B02"
    for granule_id, bands in granules.items():
        ref_path = bands.get(preferred_band) or next(iter(bands.values()))
        try:
            with rasterio.open(ref_path) as src:
                bounds_4326 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
                granule_meta[granule_id] = {
                    "crs_wkt": src.crs.to_wkt(),
                    "transform": src.transform,
                    "bounds_4326": bounds_4326,  # (west, south, east, north)
                }
        except Exception as e:
            print(f"Warning: could not read bounds for {granule_id}: {e}")

    return dict(granules), granule_meta


def find_granule_for_point(lon, lat, granule_dict, granule_meta):
    """
    Find the granule containing (lon, lat). Uses pre-cached bounds — no file I/O.
    """
    for granule_id, meta in granule_meta.items():
        if granule_id not in granule_dict:
            continue
        west, south, east, north = meta["bounds_4326"]
        # Fast pre-filter in WGS84 — pure memory, no I/O
        if not (west <= lon <= east and south <= lat <= north):
            continue
        # Point is inside — convert to native CRS to get pixel row/col
        crs = rasterio.crs.CRS.from_wkt(meta["crs_wkt"])
        xs, ys = rio_transform("EPSG:4326", crs, [lon], [lat])
        row, col = rasterio_rowcol(meta["transform"], xs[0], ys[0])
        return granule_id, int(row), int(col)

    return None, None, None


def _init_worker(granule_dict, granule_meta, nlcd_path):
    global _GRANULE_DICT, _GRANULE_META, _NLCD_SRC

    _GRANULE_DICT = granule_dict
    _GRANULE_META = granule_meta

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

        HLS_PIXEL_AREA_M2 = 30.0 * 30.0
        CENTER_OVERLAP_M2 = 400.0
        frac = CENTER_OVERLAP_M2 / HLS_PIXEL_AREA_M2
        agbd_center = agbd * frac

        granule_id, r, c = find_granule_for_point(
            lon, lat, _GRANULE_DICT, _GRANULE_META
        )
        if granule_id is None:
            return None

        sensor = "S30" if granule_id.split(".")[1] == "S30" else "L30"

        cube, used_bands, transform, crs = extract_3x3_patch_for_all_bands(
            _GRANULE_DICT[granule_id],
            r,
            c,
            sensor,
            nlcd_src=_NLCD_SRC,
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
            "center_frac": float(frac),
            "hls_granule": granule_id,
            "row": int(r),
            "col": int(c),
            # patch_tifs + bands will be filled by main process
        }

        return (i, patch_id, cube, used_bands, transform, crs_wkt, meta)

    except Exception as e:
        print(f"Error processing shot {i}: {e}", flush=True)
        return None


def build_gedi_hls_patches_multiprocess(
    gedi_folder,
    hls_root,
    patches_root="field_boundary_patch_tifs",
    out_csv_path="gedi_hls_metadata.csv",
    min_quality=1,
    nlcd_path=None,
    max_workers=None,
    chunksize=20,
):
    """
    Parallelizes per-shot extraction.
    Writes patches + CSV in the main process to keep output deterministic.
    """
    gedi_folder = Path(gedi_folder)
    hls_root = Path(hls_root)
    patches_root = Path(patches_root)
    out_csv_path = Path(out_csv_path)

    # Pre-scan existing patches to support resume after preemption
    existing_shots: set[int] = set()
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
    gedi_df = load_all_gedi_from_folder(gedi_folder, min_quality=min_quality)
    print(f"Total good-quality shots across all files: {len(gedi_df)}")

    # DOY filter: keep only April–August (DOY 91–240) to match HLS seasonal window
    MIN_DOY, MAX_DOY = 91, 240
    n_before = len(gedi_df)
    gedi_df = gedi_df[(gedi_df["gedi_doy"] >= MIN_DOY) & (gedi_df["gedi_doy"] <= MAX_DOY)].reset_index(drop=True)
    print(f"After DOY filter ({MIN_DOY}–{MAX_DOY}): {len(gedi_df)} shots kept, {n_before - len(gedi_df)} removed")

    # 2) Field polygon filtering
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

    # 3) Index HLS granules and pre-cache bounds (avoids per-shot file opens)
    print("Indexing HLS granules and bands...")
    granule_dict, granule_meta = index_hls_bands(hls_root)
    print(f"Total HLS granules found: {len(granule_dict)}")

    if max_workers is None:
        max_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "14"))

    records = gedi_df.to_dict("records")
    tasks = [(i, r) for i, r in enumerate(records) if int(r["shot_number"]) not in existing_shots]
    print(f"Tasks to process: {len(tasks)} (skipped {len(records) - len(tasks)} already done)")

    meta_rows = []
    _CSV_FLUSH_EVERY = 500

    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(granule_dict, granule_meta, nlcd_path),
    ) as ex:

        results_iter = ex.map(_process_one_shot, tasks, chunksize=chunksize)

        for res in tqdm(results_iter, total=len(tasks)):
            if res is None:
                continue

            _, patch_id, cube, used_bands, transform, crs_wkt, meta = res

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

            # Flush to CSV periodically so preemption doesn't lose all metadata
            if len(meta_rows) >= _CSV_FLUSH_EVERY:
                df_chunk = pd.DataFrame(meta_rows)
                write_header = not out_csv_path.exists()
                df_chunk.to_csv(out_csv_path, mode="a", header=write_header, index=False)
                meta_rows = []

    # Flush any remaining rows
    if meta_rows:
        df_chunk = pd.DataFrame(meta_rows)
        write_header = not out_csv_path.exists()
        df_chunk.to_csv(out_csv_path, mode="a", header=write_header, index=False)
        meta_rows = []

    total_csv = len(pd.read_csv(out_csv_path)) if out_csv_path.exists() else 0
    print(f"Saved metadata CSV: {out_csv_path}")
    print(f"Total samples with patches: {total_csv}")


def main():
    print("HLS exists:", Path(HLS_FOLDER).exists())
    print("HLS tif count:", len(list(Path(HLS_FOLDER).glob("*.tif"))))
    print("First 5 tif files:", list(Path(HLS_FOLDER).glob("*.tif"))[:5])

    build_gedi_hls_patches_multiprocess(
        gedi_folder=GEDI_FOLDER,
        hls_root=HLS_FOLDER,
        patches_root=PATCHES_ROOT,
        out_csv_path=OUT_CSV_PATH,
        min_quality=1,
        nlcd_path=NLCD_PATH,
        max_workers=MAX_WORKERS,
        chunksize=CHUNK_SIZE,
    )


if __name__ == "__main__":
    main()
