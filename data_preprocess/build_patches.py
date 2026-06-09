# buiding 3 x 3 patches for hcl(8 bands) and NLCD(1 band ) and stack them for the input for CNN model 

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


#multprocessing version of the following code 
import os
import atexit
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

# -------------------------
# Globals set in each worker
# -------------------------
_GRANULE_DICT = None
_NLCD_SRC = None

############################ all the paths here # edit 
GEDI_FOLDER = r"/s/chopin/e/proj/hyperspec/masfiq/dataset/gedi_l4a_california_north_10_2021_whole_year"

#GEDI_FOLDER = r"/s/chopin/e/proj/hyperspec/masfiq/dataset/gedi_l4a_field_boundary_2021_whole_year"



HLS_FOLDER = r"/s/chopin/e/proj/hyperspec/masfiq/dataset/hls_data_California_north_10_2021_AprilToAugust"

#HLS_FOLDER = r"/s/chopin/e/proj/hyperspec/masfiq/dataset/hls_data_field_boundary_2021_AprilToAugust"


NLCD_PATH = r"/s/chopin/e/proj/hyperspec/masfiq/dataset/Annual_NLCD_LndCov_2021_CU_C1V1/Annual_NLCD_LndCov_2021_CU_C1V1.tif"

PATCHES_ROOT = "/s/chopin/e/proj/hyperspec/masfiq/dataset/patch_tifs_california_north_10_2021_AprilToAugust"

#PATCHES_ROOT = "/s/chopin/e/proj/hyperspec/masfiq/dataset/patch_tifs_field_boundary_2021_AprilToAugust"

OUT_CSV_PATH="/s/chopin/e/proj/hyperspec/masfiq/csv_files/gedi_California_california_north_10_2021_whole_year_metadata.csv"

#OUT_CSV_PATH="/s/chopin/e/proj/hyperspec/masfiq/csv_files/gedi_field_boundary_2021_whole_year_metadata.csv"

# diff between path= r"path" and path = "path"
# nlcd_path = "C:\new\test.tif"   # \n becomes newline, \t becomes tab (bad)
# nlcd_path = r"C:\new\test.tif"  # safe

#max_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "24"))
MAX_WORKERS = 4 # means we are using 18 workers , in the slurm scrpt we set cpus-per-task=20, so the 18 workers will use the 20 cpus 

CHUNK_SIZE = 10 # each worker grabs 200 shots, processes them, then asks for the next 200.

GEOJSON_FILE =  gpd.read_file("/s/chopin/e/proj/hyperspec/masfiq/biomass_estimation_test/geojson_files/california_north_10.geojson")

#GEOJSON_FILE =  gpd.read_file("/s/chopin/e/proj/hyperspec/masfiq/biomass_estimation_test/geojson_files/Field_Boundary.geojson")

##################################################





def load_gedi_shots_from_file(h5_path, min_quality=1):
    #the path of h5 files 
    h5_path = Path(h5_path)
    rows = []

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
                    }
                )
    # Pandas reads the keys of the dicts as column names and the values as row values, creating a exel like table.
    return pd.DataFrame(rows)

# get all the gedi shots from the folder and return a df of them 
def load_all_gedi_from_folder(gedi_folder, min_quality=1):
   
    # min_quality=1 - loading the gedi files whose quality is just 1 
    gedi_folder = Path(gedi_folder)
    dfs = []
    # making the dataframe for every h5 file and appending them to dfs
    # so one dataframe per h5 file is appending into the dfs , then we concat it to get one big dataframe
    for h5_file in sorted(gedi_folder.glob("*.h5")):
        print(f"Reading {h5_file.name} ...")
        df_file = load_gedi_shots_from_file(h5_file, min_quality=min_quality)
        dfs.append(df_file)

        #     The ignore_index=True part resets the row numbers.

        # For example, without ignore_index=True, each small DataFrame may have its own index starting from 0:

        # df_file_1 index: 0, 1, 2
        # df_file_2 index: 0, 1, 2
        # df_file_3 index: 0, 1, 2

        # After concatenation, that could produce repeated row indices.
        

    if len(dfs) == 0:
        print("Empty dataframe")
        return pd.DataFrame() # we can not concat an empty dataframe so to handle error we did this 
    return pd.concat(dfs, ignore_index=True)

def extract_3x3_patch_for_all_bands(granule_bands, row, col, sensor, nlcd_src=None):
    """
    Given the band paths for one granule and a center (row, col),
    read a 3x3 patch for each band and stack them into (C, 3, 3),
    and also return the GeoTransform + CRS for that 3x3 window.
    """
    if sensor == "S30":
        band_codes = ["B02", "B03", "B04", "B8A", "B11", "B12", "EVI", "NDVI"]
    else:  # L30
        band_codes = ["B02", "B03", "B04", "B05", "B06", "B07", "EVI", "NDVI"]

    patches = [] 
    used_bands = [] # these are numpy array 
    window = None
    transform = None
    crs = None

    # if the any band of band_codes are not in granule bands  we skip the file
    for b in band_codes:
        if b not in granule_bands:
            continue

        path = granule_bands[b]
        with rasterio.open(path) as src:
            # we are getting the tif file height and width
            h, w = src.height, src.width

            # ensure we can get a full 3x3 window
            # h the pixles index from top to down
            # w  - the pixel index from left to right 
            if not (1 <= row < h - 1 and 1 <= col < w - 1):
                return None, None, None, None

            if window is None:
                window = Window(col_off=col - 1, row_off=row - 1, width=3, height=3)
                transform = src.window_transform(window)
                crs = src.crs

            patch = src.read(1, window=window)  # (3, 3)
            patches.append(patch)
            used_bands.append(b)

    if len(patches) == 0:
        return None, None, None, None
    
    #     So patches is a Python list like:

    # patches = [
    #   patch_B02,   # shape (3,3)
    #   patch_B03,   # shape (3,3)
    #   patch_B04,   # shape (3,3)
    #   ...
    # ]
    #     np.stack(..., axis=0) creates a new dimension at the front and stacks them:

    # data_cube[0, :, :] = patch_B02
    # data_cube[1, :, :] = patch_B03
    # data_cube[2, :, :] = patch_B04
    # ...


    # So the output shape becomes:

    # first dimension = how many patches you stacked = C

    # next dimensions = the patch size = 3 × 3

    # That’s why it’s (C, 3, 3).
    # ---- NEW: add NLCD as an extra band (categorical -> nearest resampling) ----
    if nlcd_src is not None and window is not None and transform is not None and crs is not None:
        nlcd_patch = np.full((3, 3), fill_value=-1, dtype=np.int16)  # -1 = unknown/nodata

        reproject(
            source=rasterio.band(nlcd_src, 1),
            destination=nlcd_patch,
            src_transform=nlcd_src.transform,
            src_crs=nlcd_src.crs,
            src_nodata=nlcd_src.nodata,
            dst_transform=transform,   # HLS 3x3 window transform
            dst_crs=crs,               # HLS CRS
            dst_nodata=-1,
            resampling=Resampling.nearest
        )

        patches.append(nlcd_patch)
        used_bands.append("NLCD")

    data_cube = np.stack(patches, axis=0)  # (C, 3, 3)
    return data_cube, used_bands, transform, crs

from collections import defaultdict
from pathlib import Path

def index_hls_bands(hls_root):
    
    # Groups all HLS TIFs in a folder by granule ID.

    # Example filename:
    #   HLS.S30.T10TEK.2021057T190939.v2.0.B02.tif

    # We want:
    #   granule_id = HLS.S30.T10TEK.2021057T190939.v2.0
    #   band_code  = B02  (or Fmask, SZA, etc.)
    
    hls_root = Path(hls_root)
    granules = defaultdict(dict) # empty dictioary {}

    # getting all the files with .tif
    for tif in hls_root.glob("*.tif"):
        stem = tif.stem  # gets the filename without ".tif"
        stem = stem.replace("_cropped", "") # getting rid of cropped with empty space
        # parts = stem.split(".")
        parts = stem.replace("_", ".").split(".") # replaces every underscore _ with a dot . then split it with . 

        # Expect something like:
        # ['HLS','S30','T10TEK','2021057T190939','v2','0','B02']
        if len(parts) < 7:
            continue

        granule_id = ".".join(parts[:-1])   # 'HLS.S30.T10TEK.2021057T190939.v2.0'
        band_code  = parts[-1]              # 'B02' or 'Fmask' or 'SZA', ...

        granules[granule_id][band_code] = tif
        # the  granules dictionary single key becomes something like this 
        #         granules = {
        #     "HLS.S30.T10TEK.2021057T190939.v2.0": {
        #         "B02": Path("/.../HLS.S30.T10TEK.2021057T190939.v2.0.B02.tif"),
        #         "B03": Path("/.../HLS.S30.T10TEK.2021057T190939.v2.0.B03.tif"),
        #         "B04": Path("/.../HLS.S30.T10TEK.2021057T190939.v2.0.B04.tif"),
        #         "NDVI": Path("/.../HLS.S30.T10TEK.2021057T190939.v2.0.NDVI.tif")
        #     }
        # }


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

def _init_worker(granule_dict, nlcd_path):
    
    # Runs once per worker process.
    # Keeps granule_dict in memory and opens NLCD once per worker.
    
    global _GRANULE_DICT, _NLCD_SRC
    _GRANULE_DICT = granule_dict

    _NLCD_SRC = None
    if nlcd_path is not None:
        _NLCD_SRC = rasterio.open(nlcd_path)
        atexit.register(_NLCD_SRC.close)

def _process_one_shot(task):
    
    # Worker: does everything except writing the output patch file.
    # Returns: None (skip) OR a tuple needed for writing + CSV row.
    
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

        cube, used_bands, transform, crs = extract_3x3_patch_for_all_bands(
            _GRANULE_DICT[granule_id], r, c, sensor, nlcd_src=_NLCD_SRC
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
            "center_frac": float(frac),
            "hls_granule": granule_id,
            "row": int(r),
            "col": int(c),
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
    max_workers=None,
    chunksize=200,
):

    gedi_folder = Path(gedi_folder)
    hls_root = Path(hls_root)
    patches_root = Path(patches_root)
    out_csv_path = Path(out_csv_path)

    # CLEAN START (same behavior as your script
    
    if patches_root.exists():
        import shutil
        shutil.rmtree(patches_root)#if root file exists delete the entire dir
    patches_root.mkdir(parents=True, exist_ok=True)# the make dir 
    # checks whether the output metadata CSV already exists. If it exists, unlink() deletes that CSV file.
    if out_csv_path.exists():
        out_csv_path.unlink()


    # 1) Load GEDI shots (same as your script)
    print("Loading GEDI shots from folder...")
    gedi_df = load_all_gedi_from_folder(gedi_folder, min_quality=min_quality)
    # gedi df has exel like column  row , so the length means all the rows 
    print(f"Total good-quality shots across all files: {len(gedi_df)}")

    # 2) Field polygon filtering (same as your script)
    import geopandas as gpd
    field = GEOJSON_FILE
    #takes all geometries inside the GeoJSON and combines them into one single geometry object.

    # For example, if your GeoJSON has one polygon:

    # Polygon A

    # then field_geom is basically that same polygon.

    # If your GeoJSON has multiple polygons:

    # Polygon A
    # Polygon B
    # Polygon C

    # then field_geom becomes one combined geometry:

    # Combined boundary = Polygon A + Polygon B + Polygon C
    field_geom = field.geometry.unary_union

    #This block converts your normal GEDI table into a spatial table so GeoPandas can do geographic operations on it.
    geometry=gpd.points_from_xy(gedi_df.lon, gedi_df.lat)
    # we are adding a geometry column in the gedi dataframe
    # creates a point geometry for every GEDI shot using its longitude and latitude.
    # For example:
    # lon = -123.45
    # lat = 41.25
    # becomes:
    # POINT(-123.45 41.25)
    # So after this line, gedi_gdf becomes a GeoDataFrame like:
    # shot_number | lat   | lon     | agbd | geometry
    # 12345       | 41.25 | -123.45 | 80.5 | POINT(-123.45 41.25)
    # 12346       | 41.26 | -123.46 | 75.2 | POINT(-123.46 41.26)
    #     crs="EPSG:4326" tells GeoPandas that these coordinates are in normal latitude/longitude format using WGS84. GEDI coordinates are usually stored this way.
    gedi_gdf = gpd.GeoDataFrame(
        gedi_df,
        geometry=gpd.points_from_xy(gedi_df.lon, gedi_df.lat),
        crs="EPSG:4326",
    )
    # gedi_gdf.within(field_geom) checks every GEDI point and asks:

    # Is this GEDI shot inside the GeoJSON boundary?

    # It returns a boolean result like:

    # True
    # False
    # True
    # True
    # False
    # gedi_gdf[gedi_gdf.within(field_geom)] keeps only the rows where the result is True
    # reset_index(drop=True) resets the row numbers after filtering.
    gedi_in_field = gedi_gdf[gedi_gdf.within(field_geom)].reset_index(drop=True)
    print("Shots inside field polygon:", len(gedi_in_field))
    # removes the geometry column that we added earlier 
    gedi_df = gedi_in_field.drop(columns="geometry")

    if len(gedi_df) == 0:
        print("No GEDI shots found. Check folder / file names.")
        return

    # 3) Index HLS granules (same)
    print("Indexing HLS granules and bands...")
    granule_dict = index_hls_bands(hls_root)
    print(f"Total HLS granules found: {len(granule_dict)}")

    # Decide workers (you have 32 logical CPUs; don’t max it out by default)
    #     If I did not provide max_workers,
    # then check how many CPUs SLURM assigned.
    # Use that number as max_workers.
    # If SLURM_CPUS_PER_TASK is not found, use 14.
    if max_workers is None:
        cpu = os.cpu_count() or 1
        max_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "14"))

    # Prepare tasks in stable order
    records = gedi_df.to_dict("records")
    tasks = list(enumerate(records))

    # records = gedi_df.to_dict("records") This converts the GEDI DataFrame into a list of Python dictionaries like this 
        #     records = [
        #     {
        #         "gedi_file": "file1.h5",
        #         "shot_number": 12345,
        #         "lat": 41.2,
        #         "lon": -123.4,
        #         "agbd": 80.5
        #     },
        #     {
        #         "gedi_file": "file1.h5",
        #         "shot_number": 12346,
        #         "lat": 41.3,
        #         "lon": -123.5,
        #         "agbd": 91.2
        #     }
        # ]
        # tasks = list(enumerate(records)) adds an index number to each record.
        #         So tasks becomes something like:

        # tasks = [
        #     (0, {"gedi_file": "file1.h5", "shot_number": 12345, "lat": 41.2, "lon": -123.4, "agbd": 80.5}),
        #     (1, {"gedi_file": "file1.h5", "shot_number": 12346, "lat": 41.3, "lon": -123.5, "agbd": 91.2})
        # ]

    meta_rows = [] # This creates an empty list to store metadata for every successful patch.

    # Use spawn context (safer with GDAL/rasterio on clusters)
    ctx = mp.get_context("spawn")
    # ctx = mp.get_context("spawn")

    # This sets the multiprocessing start method to "spawn".

    # In simple terms it means each worker process starts fresh, instead of copying the current Python process. This is often safer when using libraries like rasterio, GDAL, and geospatial file reading on clusters.

    #ProcessPoolExecutor(...)

    #  ProcessPoolExecutorThis creates a pool of worker processes. Instead of processing GEDI shots one by one in the main program, the code can send different GEDI shots to different workers at the same time. if max worker =4 it will create 4 worker processes, Each worker can process a different GEDI shot. 

    #initializer=_init_worker says before each worker starts processing GEDI shots, run the _init_worker() function once.
    # initargs=(granule_dict, nlcd_path) These are the arguments passed into _init_worker


    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(granule_dict, nlcd_path),
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

    # this folder is just checking if the hls folder exists or not , and how many tif files in the folder and prints the first 5 tif files , I did this when I was getting error for a typo in the hls path 
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
    max_workers=MAX_WORKERS,     
    chunksize=CHUNK_SIZE
    
    )


if __name__ == "__main__":
    main()
