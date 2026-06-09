import datetime as dt 
import pandas as pd
import geopandas as gpd
import earthaccess
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import orient
from pathlib import Path
import shutil

import earthaccess
################################################ edit 
# LOCAL_PATH = "/s/chopin/e/proj/hyperspec/masfiq/dataset/gedi_l4a_California_Mainland_2021_whole_year"
LOCAL_PATH  = "/s/chopin/e/proj/hyperspec/masfiq/dataset/gedi_l4a_california_north_10_2021_whole_year"

#LOCAL_PATH  = "/s/chopin/e/proj/hyperspec/masfiq/dataset/gedi_l4a_field_boundary_2021_whole_year"

# geojson file 

GEOJSON_FILE = gpd.read_file("/s/chopin/e/proj/hyperspec/masfiq/biomass_estimation_test/geojson_files/california_north_10.geojson").geometry

#GEOJSON_FILE = gpd.read_file("/s/chopin/e/proj/hyperspec/masfiq/biomass_estimation_test/geojson_files/Field_Boundary.geojson").geometry


# USA bounding box
# bound = (-125.0,  24.5, -66.5,  49.5)  

#change the dates for which we need to download the gedi files 
# time bound
start_date = dt.datetime(2021, 1, 1) # specify your own start date
end_date = dt.datetime(2021, 12, 30)  # specify your end start date



#####################################################
dir_path = Path(LOCAL_PATH)

if dir_path.exists():
    print("Director exists , deleting directory ...\n")
    shutil.rmtree(dir_path)

dir_path.mkdir(parents=True, exist_ok=True)

# granules = earthaccess.search_data(
#     count=-1, # needed to retrieve all granules
#     bounding_box = bound,
#     temporal=(start_date, end_date), # time bound
#     doi='10.3334/ORNLDAAC/2056' # GEDI L4A DOI 
# )
# print(f"Total granules found: {len(granules)}")


def convert_umm_geometry(gpoly):
    """converts UMM geometry to multipolygons"""
    multipolygons = []
    for gl in gpoly:
        ltln = gl["Boundary"]["Points"]
        points = [(p["Longitude"], p["Latitude"]) for p in ltln]
        multipolygons.append(Polygon(points))
    return MultiPolygon(multipolygons)

def convert_list_gdf(datag):
    """converts List[] to geopandas dataframe"""
    # create pandas dataframe from json
    df = pd.json_normalize([vars(granule)['render_dict'] for granule in datag])
    # keep only last string of the column names
    df.columns=df.columns.str.split('.').str[-1]
    # convert polygons to multipolygonal geometry
    df["geometry"] = df["GPolygons"].apply(convert_umm_geometry)
    # return geopandas dataframe
    return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

# only keep three columns
# gdf = convert_list_gdf(granules)[['GranuleUR', 'size', 'geometry']]

poly = GEOJSON_FILE
# poly = gpd.read_file("geojson_files/california_north_12.geojson").geometry
# poly = gpd.read_file("geojson_files/Field_Boundary.geojson").geometry
poly.explore(color='red',  fill=False)
# poly.explore(color='green',  fill=False)

# bounding lon, lat as a list of tuples
poly = poly.apply(orient, args=(1,))
xy = poly.simplify(0.01).get_coordinates()

granules = earthaccess.search_data(
    count=-1, # needed to retrieve all granules
    temporal=(start_date, end_date), # time bound
    doi='10.3334/ORNLDAAC/2056', # GEDI L4A DOI 
    polygon=list(zip(xy.x, xy.y))
)
print(f"Total granules found: {len(granules)}")

gdf = convert_list_gdf(granules)[['GranuleUR', 'size', 'geometry']]



# This will pop up a prompt in your terminal / notebook for your Earthdata username & password,
# then save them into ~/.netrc so future calls can go non-interactive.
auth = earthaccess.login(
    strategy="interactive",
    persist=True   # <-- write your creds to ~/.netrc
)

if not auth.authenticated:
    raise RuntimeError("Login failed; check your Earthdata credentials")


# works if the EDL login already been persisted to a netrc
auth = earthaccess.login(strategy="netrc") 
if not auth.authenticated:
    # ask for EDL credentials and persist them in a .netrc file
    auth = earthaccess.login(strategy="interactive", persist=True)


# downloaded_files = earthaccess.download(granules[:2], local_path="test_download")
# downloaded_files = earthaccess.download(granules[:500], local_path="gedi_l4a_california")
# downloaded_files = earthaccess.download(granules[:2], local_path="/../dataset/gedi_l4a_california_mainland_april_2021")
# download all files 
#downloaded_files = earthaccess.download(granules, local_path="../dataset/gedi_l4a_California_North_10_2021_whole_year")
downloaded_files = earthaccess.download(granules, local_path=LOCAL_PATH)