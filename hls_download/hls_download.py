#I am writing this file which can download all the bands of the hls files in one go using multiprocessing  , I am optiizing it using for the falcon cluster 


#first we will do all the imports 
import os
from datetime import datetime
import numpy as np
import pandas as pd
import geopandas as gp 
from skimage import io
import matplotlib.pyplot as plt
from osgeo import gdal
import xarray as xr
import rioxarray as rxr
import hvplot.xarray
import hvplot.pandas

import earthaccess
import pathlib

from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from shapely import wkt




# Define function to scale 
#my note
#eads the scale factor you stored earlier (like 0.0001)

# Multiplies the data by it (this is the actual scaling)

# Sets scale_factor to 1 afterward so you don’t accidentally scale twice
def scaling(band):
    scale_factor = band.attrs['scale_factor'] 
    band_out = band.copy()
    band_out.data = band.data*scale_factor
    band_out.attrs['scale_factor'] = 1
    return(band_out)




def calc_evi(red, blue, nir):
      # Create EVI xarray.DataArray that has the same coordinates and metadata
      evi = red.copy()
      # Calculate the EVI
      evi_data = 2.5 * ((nir.data - red.data) / (nir.data + 6.0 * red.data - 7.5 * blue.data + 1.0))
      # Replace the Red xarray.DataArray data with the new EVI data
      evi.data = evi_data
      # exclude the inf values
      evi = xr.where(evi != np.inf, evi, np.nan, keep_attrs=True)
      # change the long_name in the attributes
      evi.attrs['long_name'] = 'EVI'
      evi.attrs['scale_factor'] = 1
      return evi

# we do not need to declare the bit_nums as it is already declared in the function by defaults 
#
def create_quality_mask(quality_data, bit_nums: list = [1, 2, 3, 4, 5]):
    """
    Uses the Fmask layer and bit numbers to create a binary mask of good pixels.
    By default, bits 1-5 are used.
    """
    mask_array = np.zeros((quality_data.shape[0], quality_data.shape[1]))
    # Remove/Mask Fill Values and Convert to Integer
    quality_data = np.nan_to_num(quality_data, 255).astype(np.int8)
    for bit in bit_nums:
        # Create a Single Binary Mask Layer
        mask_temp = np.array(quality_data) & 1 << bit > 0
        mask_array = np.logical_or(mask_array, mask_temp)
    return mask_array

def evi_download(hls_results_urls, output_dir):
    #EVI indicies
# you need to create the folder before downloading at the end of this prtion where it is mentioned
# from here we can join this code with the previous cell code to run it from a single python file 
# This following code is evi indicies 

    

 

    for j, h in enumerate(hls_results_urls):
        
        outName = h[0].split('/')[-1].split('v2.0')[0] +'v2.0_EVI_cropped.tif'
        print(outName)

        output_path = os.path.join(output_dir, outName)

        evi_band_links = []
        if h[0].split('/')[4] == 'HLSS30.020':
            evi_bands = ['B8A', 'B04', 'B02', 'Fmask'] # NIR RED BLUE
        else:
            evi_bands = ['B05', 'B04', 'B02', 'Fmask'] # NIR RED BLUE
        
        for a in h: 
            if any(b in a for b in evi_bands):
                evi_band_links.append(a)

        ###########################################
        # change here the file location, you have to create the folder before downloading
        ###########################################
        # Check if file already exists in output directory, if yes--skip that file and move to the next observation
        # if os.path.exists(f'/s/chopin/e/proj/hyperspec/masfiq/dataset/hls_data_field_geojson_2021_AprilToAugust_evi/{outName}'):
        #     print(f"{outName} has already been processed and is available in this directory, moving to next file.")
        #     continue

        if os.path.exists(output_path):
            print(f"{outName} has already been processed and is available in this directory, moving to next file.")
            continue

        
        
        # Use vsicurl to load the data directly into memory (be patient, may take a few seconds)
        chunk_size = dict(band=1, x=512, y=512) # Tiles have 1 band and are divided into 512x512 pixel chunks
        for e in evi_band_links:
            print(e)
            if e.rsplit('.', 2)[-2] == evi_bands[0]:      # NIR index
                nir = rxr.open_rasterio(e, chunks=chunk_size, masked= True).squeeze('band', drop=True)
                nir.attrs['scale_factor'] = 0.0001                         # hard coded the scale_factor attribute 
            elif e.rsplit('.', 2)[-2] == evi_bands[1]:    # red index
                red = rxr.open_rasterio(e, chunks=chunk_size, masked= True).squeeze('band', drop=True)
                red.attrs['scale_factor'] = 0.0001                         # hard coded the scale_factor attribute
            elif e.rsplit('.', 2)[-2] == evi_bands[2]:    # blue index
                blue = rxr.open_rasterio(e, chunks=chunk_size, masked= True).squeeze('band', drop=True)
                blue.attrs['scale_factor'] = 0.0001                        # hard coded the scale_factor attribute
            elif e.rsplit('.', 2)[-2] == evi_bands[3]:    # Fmask index
                fmask = rxr.open_rasterio(e, chunks=chunk_size, masked= True).squeeze('band', drop=True)
            
        fsUTM = field.to_crs(nir.spatial_ref.crs_wkt)

        # Crop to our ROI and apply scaling and masking
        nir_cropped = nir.rio.clip(fsUTM.geometry.values, fsUTM.crs, all_touched=True)
        red_cropped = red.rio.clip(fsUTM.geometry.values, fsUTM.crs, all_touched=True)
        blue_cropped = blue.rio.clip(fsUTM.geometry.values, fsUTM.crs, all_touched=True)
        fmask_cropped = fmask.rio.clip(fsUTM.geometry.values, fsUTM.crs, all_touched=True)
        
        print('Cropped')      
        
        # Fix Scaling
        nir_cropped_scaled = scaling(nir_cropped)
        red_cropped_scaled = scaling(red_cropped)
        blue_cropped_scaled = scaling(blue_cropped)

        # Generate EVI
        
        evi_cropped = calc_evi(red_cropped_scaled, blue_cropped_scaled, nir_cropped_scaled)

        print('EVI Calculated')
        
        # Apply Quality Filter
        mask_layer = create_quality_mask(fmask_cropped.data)
        evi_cropped = evi_cropped.where(~mask_layer)

        # Remove any observations that are entirely fill value
        if np.nansum(evi_cropped.data) == 0.0:
            print(f"File: {h[0].split('/')[-1].rsplit('.', 1)[0]} was entirely fill values and will not be exported.")
            continue

        #evi_cropped.rio.to_raster(raster_path=f'/s/chopin/e/proj/hyperspec/masfiq/dataset/hls_data_field_geojson_2021_AprilToAugust_evi/{outName}', driver='COG')

        evi_cropped.rio.to_raster(raster_path=output_path, driver='COG')

        
        
        print(f"Processed file {j+1} of {len(hls_results_urls)}")
        


def ndvi_download(hls_results_urls, output_dir):
    #NDVI download 
# This following code is for ndvi indicies 
 

    for j, h in enumerate(hls_results_urls):
        
        outName = h[0].split('/')[-1].split('v2.0')[0] +'v2.0_NDVI_cropped.tif'
        print(outName)

        output_path = os.path.join(output_dir, outName)

        ndvi_band_links = []
        if h[0].split('/')[4] == 'HLSS30.020':     # Sentinel-2
            ndvi_bands = ['B8A', 'B04', 'Fmask']  # NIR, RED
        else:                                     # Landsat
            ndvi_bands = ['B05', 'B04', 'Fmask']  # NIR, RED
        
        for a in h: 
            if any(b in a for b in ndvi_bands):
                ndvi_band_links.append(a)

        ###########################################
        # change here the file location, you have to create the folder before downloading
        ###########################################
        # Check if file already exists in output directory, if yes--skip that file and move to the next observation
        # if os.path.exists(f'//s/chopin/e/proj/hyperspec/masfiq/dataset/hls_data_California_North_10_2021_AprilToAugust_evi/{outName}'):
        #     print(f"{outName} has already been processed and is available in this directory, moving to next file.")
        #     continue

        if os.path.exists(output_path):
            print(f"{outName} has already been processed and is available in this directory, moving to next file.")
            continue

        
        
        # Use vsicurl to load the data directly into memory (be patient, may take a few seconds)
        chunk_size = dict(band=1, x=512, y=512) # Tiles have 1 band and are divided into 512x512 pixel chunks
        for e in ndvi_band_links:
            print(e)
            if e.rsplit('.', 2)[-2] == ndvi_bands[0]:      # NIR index
                nir = rxr.open_rasterio(e, chunks=chunk_size, masked= True).squeeze('band', drop=True)
                nir.attrs['scale_factor'] = 0.0001                         # hard coded the scale_factor attribute 
            elif e.rsplit('.', 2)[-2] == ndvi_bands[1]:    # red index
                red = rxr.open_rasterio(e, chunks=chunk_size, masked= True).squeeze('band', drop=True)
                red.attrs['scale_factor'] = 0.0001                         # hard coded the scale_factor attribute
            elif e.rsplit('.', 2)[-2] == ndvi_bands[2]:    # Fmask index
                fmask = rxr.open_rasterio(e, chunks=chunk_size, masked= True).squeeze('band', drop=True)
            
        fsUTM = field.to_crs(nir.spatial_ref.crs_wkt)

        # Crop to our ROI and apply scaling and masking
        nir_cropped = nir.rio.clip(fsUTM.geometry.values, fsUTM.crs, all_touched=True)
        red_cropped = red.rio.clip(fsUTM.geometry.values, fsUTM.crs, all_touched=True)
        fmask_cropped = fmask.rio.clip(fsUTM.geometry.values, fsUTM.crs, all_touched=True)
        
        print('Cropped')      
        
        # Fix Scaling
        nir_cropped_scaled = scaling(nir_cropped)
        red_cropped_scaled = scaling(red_cropped)
        

        # Generate EVI
        
        ndvi_cropped = (nir_cropped_scaled - red_cropped_scaled) / (nir_cropped_scaled + red_cropped_scaled)

        print('NDVI Calculated')
        
        # Apply Quality Filter
        mask_layer = create_quality_mask(fmask_cropped.data)
        ndvi_cropped = ndvi_cropped.where(~mask_layer)

        # Remove any observations that are entirely fill value
        if np.nansum(   ndvi_cropped.data) == 0.0:
            print(f"File: {h[0].split('/')[-1].rsplit('.', 1)[0]} was entirely fill values and will not be exported.")
            continue

        ###########################################
        # change here the file location, you have to create the folder before downloading
        ###########################################
        #ndvi_cropped.rio.to_raster(raster_path=f'/s/chopin/e/proj/hyperspec/masfiq/dataset/hls_data_California_North_10_2021_AprilToAugust_evi/{outName}', driver='COG')
        ndvi_cropped.rio.to_raster(raster_path=output_path, driver='COG')

        
        
        print(f"Processed file {j+1} of {len(hls_results_urls)}")


def single_band_download(args):
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


    ########################### change here
    #output dir , no need to create it manually, its automated, just name it whatever you want  
    output_dir = "/s/chopin/e/proj/hyperspec/masfiq/dataset/hls_data_California_Mainland_2021_AprilToAugust"
    # Create the folder if it does not already exist
    os.makedirs(output_dir, exist_ok=True)

     # earthaccesss login
    earthaccess.login(persist=True)

    ##################################### change here
 
    #read any geojson file and the bounding box of the geojson file
    #field = gp.read_file('geojson_files/Field_Boundary.geojson')
    field = gp.read_file('geojson_files/california_north_12.geojson')

    bbox = tuple(list(field.total_bounds))
    
    ###################################### change here 
    #set up the temporal resoulation year - month - day - hour - minute - second
    temporal = ("2021-04-01T00:00:00", "2021-08-30T23:59:59")#April to August

    # get all the granule in a list named results 
    results = earthaccess.search_data(
        short_name=['HLSL30','HLSS30'],
        bounding_box=bbox,
        temporal=temporal,
        # count=100
    )
    # printing how many granules we have in result list
    print("Total granules found(the length of results): " + str(len(results)))
    print()
    print()


    # get all the data links of the objects in the in hls_results_urls list 
    hls_results_urls = [granule.data_links() for granule in results]
    print(hls_results_urls[0:1]) # Show a subset of the list

    # check the actual image in jpg format which we will not use , we will only use the tif files as from that we can accesss the multi band data
    browse_urls = [granule.dataviz_links()[0] for granule in results] # 0 retrieves only the https links
    print("Check the images by clicking on the links below:\n")
    print(browse_urls[0:2])  # Show a subset of the list
    print()
    print()

    # GDAL configurations used to successfully access LP DAAC Cloud Assets via vsicurl 
    gdal.SetConfigOption('GDAL_HTTP_COOKIEFILE','~/cookies.txt')
    gdal.SetConfigOption('GDAL_HTTP_COOKIEJAR', '~/cookies.txt')
    gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN','EMPTY_DIR')
    gdal.SetConfigOption('CPL_VSIL_CURL_ALLOWED_EXTENSIONS','TIF')
    gdal.SetConfigOption('GDAL_HTTP_UNSAFESSL', 'YES')
    gdal.SetConfigOption('GDAL_HTTP_MAX_RETRY', '10')
    gdal.SetConfigOption('GDAL_HTTP_RETRY_DELAY', '0.5')


    # code for accessing the google earth engine 


    earthaccess.login(persist=True)  # creates/refreshes ~/.urs_cookies

    home = pathlib.Path.home()
    os.environ["NETRC"] = str(home / "_netrc")                 # Windows uses _netrc
    os.environ["GDAL_HTTP_COOKIEFILE"] = str(home / ".urs_cookies")
    os.environ["GDAL_HTTP_COOKIEJAR"]  = str(home / ".urs_cookies")

    
    url = "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSS30.020/HLS.S30.T10TEK.2021142T185921.v2.0/HLS.S30.T10TEK.2021142T185921.v2.0.Fmask.tif"
    ds = gdal.Open(f"/vsicurl/{url}")
    print(ds is not None)  # should print True

    print("if True is printed above then you are good to go\n")
    print()
    print()

    evi_download(hls_results_urls, output_dir)
    ndvi_download(hls_results_urls, output_dir)

    #multiprocessing the single band download 
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
    bands = ["B02" , "B03" , "B04" , "B05" , "B8A" , "B11", "B12" , "B02" , "B03" , "B04" , "B05" , "B06" , "B07" ]

    write_driver = "GTiff"
    avoid_L30_thermal_B11 = True



    # --- make geometry picklable for multiprocessing ---
    _FIELD_WKT = field.geometry.unary_union.wkt
    _FIELD_CRS = field.crs.to_string()
    

    for target_band in bands: 
        max_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "13"))

        jobs = [
            (j, h, target_band, output_dir, write_driver, avoid_L30_thermal_B11, _FIELD_WKT, _FIELD_CRS)
            for j, h in enumerate(hls_results_urls)
        ]

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(single_band_download, job) for job in jobs]

            for f in as_completed(futures):
                j, msg = f.result()
                print(msg)
        
    


     