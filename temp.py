# this is the multiprocessing version of the single band code 
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import geopandas as gp 
from shapely import wkt

# ============================
# Download ONE band at a time
# ============================

##########################################
    # S30 
    #  "blue": "B02",
    #         "green": "B03",
    #         "red": "B04",
    #         "nir": "B8A", bad ase
    #         "swir1": "B11",
    #         "swir2": "B12", bad ase
    #     }
    # else:
    #     band_map = {
    #         "blue": "B02",
    #         "green": "B03",
    #         "red": "B04",
    #         "nir": "B05",
    #         "swir1": "B06",
    #         "swir2": "B07",


target_band = "B04"   # e.g., "B02", "B05", "B06", "B8A", "B11", etc. :contentReference[oaicite:1]{index=1}
out_dir = "/s/chopin/e/proj/hyperspec/masfiq/dataset/hls_data_California_North_12_2021_AprilToAugust"
write_driver = "GTiff"
avoid_L30_thermal_B11 = True



# --- make geometry picklable for multiprocessing ---
_FIELD_WKT = field.geometry.unary_union.wkt
_FIELD_CRS = field.crs.to_string()

def process_one_granule(args):
    """
    Worker for ONE granule (one 'h' list of urls).
    Returns (j, message).
    """
    (j, h, target_band, out_dir, write_driver, avoid_L30_thermal_B11, field_wkt, field_crs) = args

    # Rebuild field GeoDataFrame in the worker
    geom = wkt.loads(field_wkt)
    field_local = gp.GeoDataFrame(geometry=[geom], crs=field_crs)

    # --- your original per-loop logic ---
    band = None
    fmask = None

    first_url = h[0]
    is_L30 = ("HLS.L30" in first_url) or ("HLSL30" in first_url)

    if avoid_L30_thermal_B11 and is_L30 and target_band == "B11":
        return (j, "Skipping L30 B11 (thermal). If you want SWIR1 reflectance for L30, use target_band='B06'.")

    outName = first_url.split("/")[-1].split("v2.0")[0] + f"v2.0_{target_band}_cropped.tif"
    outPath = os.path.join(out_dir, outName)

    if os.path.exists(outPath):
        return (j, f"{outName} already exists; skipping.")

    bands_needed = [target_band, "Fmask"]
    band_links = [a for a in h if any(b in a for b in bands_needed)]
    chunk_size = dict(band=1, x=512, y=512)

    for e in band_links:
        suffix = e.rsplit(".", 2)[-2]

        if suffix == target_band:
            band = rxr.open_rasterio(e, chunks=chunk_size, masked=True).squeeze("band", drop=True)

            # reflectance scaling (don’t do for thermal)
            if not (is_L30 and target_band in ["B10", "B11"]):
                band.attrs["scale_factor"] = 0.0001

        elif suffix == "Fmask":
            fmask = rxr.open_rasterio(e, chunks=chunk_size, masked=True).squeeze("band", drop=True)

    if band is None or fmask is None:
        return (j, f"Missing {target_band} or Fmask in this granule; skipping -> {first_url}")

    fsUTM = field_local.to_crs(band.rio.crs)

    band_cropped  = band.rio.clip(fsUTM.geometry.values, fsUTM.crs, all_touched=True)
    fmask_cropped = fmask.rio.clip(fsUTM.geometry.values, fsUTM.crs, all_touched=True)

    if band_cropped.shape != fmask_cropped.shape:
        return (j, f"Shape mismatch after clip (band {band_cropped.shape} vs fmask {fmask_cropped.shape}); skipping -> {first_url}")

    band_cropped_scaled = scaling(band_cropped) if band_cropped.attrs.get("scale_factor", 1) != 1 else band_cropped

    mask_layer = create_quality_mask(fmask_cropped.data)
    mask_da = xr.DataArray(mask_layer, coords=band_cropped_scaled.coords, dims=band_cropped_scaled.dims)

    band_masked = band_cropped_scaled.where(~mask_da)

    # NOTE: This forces a full compute (slow). Keep it only if you really need it.
    if int(band_masked.count().compute()) == 0:
        return (j, "All masked/empty after clipping; skipping.")

    if write_driver == "GTiff":
        band_masked.rio.to_raster(
            outPath,
            driver="GTiff",
            compress="deflate",
            tiled=True,
            blockxsize=512,
            blockysize=512,
            BIGTIFF="IF_SAFER",
        )
    else:
        band_masked.rio.to_raster(outPath, driver=write_driver)

    return (j, f"Processed {j+1}/{len(hls_results_urls)} -> {outName}")


if __name__ == "__main__":
    os.makedirs(out_dir, exist_ok=True)

    # Start conservative. You can try 6–10 later if stable.
    # max_workers = min(os.cpu_count() or 4, 6)
    max_workers = 13   

    jobs = [
        (j, h, target_band, out_dir, write_driver, avoid_L30_thermal_B11, _FIELD_WKT, _FIELD_CRS)
        for j, h in enumerate(hls_results_urls)
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(process_one_granule, job) for job in jobs]

        for f in as_completed(futures):
            j, msg = f.result()
            print(msg)