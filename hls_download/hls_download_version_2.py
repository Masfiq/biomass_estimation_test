#this is the version 2 of the hls download py file 
# For the single-band part, that one created a new ProcessPoolExecutor for every band inside for target_band in bands: . That’s slow and it can look like “no output” for a long time.
    
# This one does the following 
# uses one ProcessPoolExecutor
# processes one granule per worker, and inside that worker it writes EVI + NDVI + all bands (so each granule’s URLs are read once, not 13 separate times)
# prints progress continuously as tasks finish
# uses SLURM_CPUS_PER_TASK the same way Falcon’s sample does

# --- DEFAULT PATHS / SETTINGS (edit these) ---
#DEFAULT_GEOJSON = "/s/chopin/e/proj/hyperspec/masfiq/biomass_estimation_test/geojson_files/california_north_10.geojson"

DEFAULT_GEOJSON = "/s/chopin/e/proj/hyperspec/masfiq/biomass_estimation_test/geojson_files/california_top_half.geojson"

# DEFAULT_OUT_DIR = "/s/chopin/e/proj/hyperspec/masfiq/dataset/hls_data_field_boundary_2021_AprilToAugust"

DEFAULT_OUT_DIR = "/s/chopin/e/proj/hyperspec/masfiq/dataset/hls_data_california_top_half_2021_AprilToAugust"
DEFAULT_START   = "2021-04-01T00:00:00" # year month day time  #April to August 
DEFAULT_END     = "2021-08-30T23:59:59"
DEFAULT_SKIP_EXISTING = True

#################################


import os
import time
import pathlib
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import geopandas as gp
from osgeo import gdal
import xarray as xr
import rioxarray as rxr
import earthaccess
from shapely import wkt
import dask

from rioxarray.exceptions import NoDataInBounds


def scaling(band):
    scale_factor = band.attrs.get("scale_factor", 1)
    if scale_factor == 1:
        return band
    out = band.copy()
    out.data = band.data * scale_factor
    out.attrs["scale_factor"] = 1
    return out


def calc_evi(red, blue, nir):
    evi = red.copy()
    denom = (nir + 6.0 * red - 7.5 * blue + 1.0)
    denom = denom.where(denom != 0)  # avoid divide-by-zero

    evi_data = 2.5 * ((nir - red) / denom)
    evi.data = evi_data.data  # keep dask-backed data

    evi = xr.where(np.isfinite(evi), evi, np.nan, keep_attrs=True)
    evi.attrs["long_name"] = "EVI"
    evi.attrs["scale_factor"] = 1
    return evi


def create_quality_mask(quality_data, bit_nums=(1, 2, 3, 4, 5)):
    mask_array = np.zeros((quality_data.shape[0], quality_data.shape[1]), dtype=bool)
    quality_data = np.nan_to_num(quality_data, 255).astype(np.int16)
    for bit in bit_nums:
        mask_temp = (quality_data & (1 << bit)) > 0
        mask_array = np.logical_or(mask_array, mask_temp)
    return mask_array


def _open_band_retry(url, chunk_size, tries=8, base_sleep=0.5):
    """
    Retries open_rasterio metadata reads. GDAL_HTTP_MAX_RETRY also helps later reads.
    """
    last = None
    for k in range(tries):
        try:
            return rxr.open_rasterio(url, chunks=chunk_size, masked=True).squeeze("band", drop=True)
        except Exception as e:
            last = e
            time.sleep(min(20, base_sleep * (2 ** k)))
    raise last


def process_granule(job):
    """
    One granule -> write EVI + NDVI + all reflectance bands for that sensor.
    Returns (j, msg).
    """
    (j, h, out_dir, field_wkt, field_crs, write_driver_band, write_driver_idx, skip_existing) = job

    dask.config.set(scheduler="single-threaded")

    # Rebuild ROI geometry inside the worker (picklable)
    geom = wkt.loads(field_wkt)
    field_local = gp.GeoDataFrame(geometry=[geom], crs=field_crs)

    first_url = h[0]
    is_s30 = ("/HLSS30.020/" in first_url) or ("HLS.S30" in first_url)

    # Bands by sensor
    if is_s30:
        nir_name, red_name, blue_name = "B8A", "B04", "B02"
        reflectance_bands = ["B02", "B03", "B04", "B8A", "B11", "B12"]
    else:
        nir_name, red_name, blue_name = "B05", "B04", "B02"
        reflectance_bands = ["B02", "B03", "B04", "B05", "B06", "B07"]

    # Build suffix->url map from this granule's links
    suffix_to_url = {}
    for u in h:
        if not u.lower().endswith(".tif"):
            continue
        try:
            suffix = u.rsplit(".", 2)[-2]
        except Exception:
            continue
        suffix_to_url[suffix] = u

    needed = set(reflectance_bands + ["Fmask", nir_name, red_name, blue_name])
    missing = [b for b in needed if b not in suffix_to_url]
    if "Fmask" not in suffix_to_url:
        return (j, f"[{j}] missing Fmask; skipping.")

    base = first_url.split("/")[-1].split("v2.0")[0] + "v2.0"

    # If everything already exists, skip early (fast)
    if skip_existing:
        all_targets = []
        all_targets.append(os.path.join(out_dir, f"{base}_EVI_cropped.tif"))
        all_targets.append(os.path.join(out_dir, f"{base}_NDVI_cropped.tif"))
        for b in reflectance_bands:
            all_targets.append(os.path.join(out_dir, f"{base}_{b}_cropped.tif"))
        if all(os.path.exists(p) for p in all_targets):
            return (j, f"[{j}] all outputs exist; skipping {base}")

    chunk_size = dict(band=1, x=512, y=512)

    # Open fmask first (for mask + CRS), then open only what exists
    #fmask = _open_band_retry(suffix_to_url["Fmask"], chunk_size)
    try:
        fmask = _open_band_retry(suffix_to_url["Fmask"], chunk_size)
    except Exception as e:
        return (j, f"[{j}] {base}: SKIP could not open Fmask ({type(e).__name__}: {e})")
    ref_for_crs = fmask
    if ref_for_crs.rio.crs is None:
        # fallback: open one reflectance band
        for b in reflectance_bands:
            if b in suffix_to_url:
                ref_for_crs = _open_band_retry(suffix_to_url[b], chunk_size)
                break

    fsUTM = field_local.to_crs(ref_for_crs.rio.crs)

    # Small speed win: clip_box first (less IO), then polygon clip
    minx, miny, maxx, maxy = fsUTM.total_bounds
    fmask_small = fmask.rio.clip_box(minx, miny, maxx, maxy)
    #fmask_crop = fmask_small.rio.clip(fsUTM.geometry.values, fsUTM.crs, all_touched=True)
    try:
        #fmask_crop = fmask_small.rio.clip(fsUTM.geometry.values, fsUTM.crs, all_touched=True)
        fmask_crop = fmask.rio.clip(fsUTM.geometry.values, fsUTM.crs, all_touched=True)
    except NoDataInBounds:
        return (j, f"[{j}] {base}: SKIP NoDataInBounds (field does not intersect FMASK)")

    # Build quality mask once
    mask_layer = create_quality_mask(fmask_crop.data)

    written = 0

    # Helper: open+clip+scale+mask one band
    def load_band(bname):
        if bname not in suffix_to_url:
            return None
        da = _open_band_retry(suffix_to_url[bname], chunk_size)
        # reflectance scaling
        da.attrs["scale_factor"] = 0.0001
        da_small = da.rio.clip_box(minx, miny, maxx, maxy)
        da_crop = da_small.rio.clip(fsUTM.geometry.values, fsUTM.crs, all_touched=True)
        da_scaled = scaling(da_crop)
        return da_scaled.where(~mask_layer)

    # Write reflectance bands
    for b in reflectance_bands:
        out_path = os.path.join(out_dir, f"{base}_{b}_cropped.tif")
        if skip_existing and os.path.exists(out_path):
            continue
        da_masked = load_band(b)
        if da_masked is None:
            continue

        if write_driver_band == "GTiff":
            da_masked.rio.to_raster(
                out_path,
                driver="GTiff",
                compress="deflate",
                tiled=True,
                blockxsize=512,
                blockysize=512,
                BIGTIFF="IF_SAFER",
            )
        else:
            da_masked.rio.to_raster(out_path, driver=write_driver_band)
        written += 1

    # NDVI + EVI need NIR/RED/BLUE
    nir = load_band(nir_name)
    red = load_band(red_name)
    blue = load_band(blue_name)

    # NDVI
    ndvi_path = os.path.join(out_dir, f"{base}_NDVI_cropped.tif")
    if nir is not None and red is not None and (not (skip_existing and os.path.exists(ndvi_path))):
        den = (nir + red).where((nir + red) != 0)  # avoid divide-by-zero
        ndvi = (nir - red) / den

        ndvi = xr.where(np.isfinite(ndvi), ndvi, np.nan, keep_attrs=True)
        ndvi.attrs["long_name"] = "NDVI"
        ndvi.rio.to_raster(ndvi_path, driver=write_driver_idx)
        written += 1

    # EVI
    evi_path = os.path.join(out_dir, f"{base}_EVI_cropped.tif")
    if nir is not None and red is not None and blue is not None and (not (skip_existing and os.path.exists(evi_path))):
        evi = calc_evi(red, blue, nir)
        evi.rio.to_raster(evi_path, driver=write_driver_idx)
        written += 1

    if missing:
        return (j, f"[{j}] {base}: wrote {written} files (missing some bands: {missing})")
    return (j, f"[{j}] {base}: wrote {written} files")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geojson", default=DEFAULT_GEOJSON, help="ROI geojson path")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="Output directory")
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)


    # default True, but allow turning it off if you want
    ap.add_argument("--skip_existing", action="store_true", default=DEFAULT_SKIP_EXISTING)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.set_defaults(skip_existing=DEFAULT_SKIP_EXISTING)

    ap.add_argument("--workers", type=int, default=0, help="0 = use SLURM_CPUS_PER_TASK")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Falcon sample uses SLURM_CPUS_PER_TASK :contentReference[oaicite:5]{index=5}
    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    workers = args.workers if args.workers > 0 else slurm_cpus

    # IMPORTANT: Too many workers can trigger 503 throttling from LP DAAC.
    # Keep a safety cap unless you KNOW it’s stable.
    workers = max(1, min(workers, 15))
    workers = max(1, min(workers, 8, slurm_cpus))

    # Make sure prints appear in .out while job is running
    print(f"Using workers={workers} (SLURM_CPUS_PER_TASK={slurm_cpus})", flush=True)

    # Login must be non-interactive on compute nodes; persist=True assumes you already set it up.
    earthaccess.login(persist=True)

    field = gp.read_file(args.geojson)
    bbox = tuple(list(field.total_bounds))
    temporal = (args.start, args.end)

    results = earthaccess.search_data(
        short_name=["HLSL30", "HLSS30"],
        bounding_box=bbox,
        temporal=temporal,
    )
    print(f"Total granules found: {len(results)}", flush=True)

    hls_results_urls = [granule.data_links() for granule in results]

    # GDAL / vsicurl tuning (you already had these) :contentReference[oaicite:6]{index=6}
    gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    gdal.SetConfigOption("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", "TIF")
    gdal.SetConfigOption("GDAL_HTTP_UNSAFESSL", "YES")
    gdal.SetConfigOption("GDAL_HTTP_MAX_RETRY", "10")
    gdal.SetConfigOption("GDAL_HTTP_RETRY_DELAY", "0.5")

    # Ensure URS cookies are available (your script already does this) :contentReference[oaicite:7]{index=7}
    home = pathlib.Path.home()
    os.environ["GDAL_HTTP_COOKIEFILE"] = str(home / ".urs_cookies")
    os.environ["GDAL_HTTP_COOKIEJAR"] = str(home / ".urs_cookies")

    _FIELD_WKT = field.geometry.unary_union.wkt
    _FIELD_CRS = field.crs.to_string()

    jobs = [
        (
            j,
            h,
            args.out_dir,
            _FIELD_WKT,
            _FIELD_CRS,
            "GTiff",  # reflectance bands
            "COG",    # indices
            args.skip_existing,
        )
        for j, h in enumerate(hls_results_urls)
    ]

    done = 0
    total = len(jobs)

    print(f"Submitting {total} granules to the pool...", flush=True)

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(process_granule, job) for job in jobs]

        for f in as_completed(futures):
            j, msg = f.result()
            done += 1
            print(f"[{done}/{total}] {msg}", flush=True)


if __name__ == "__main__":
    main()