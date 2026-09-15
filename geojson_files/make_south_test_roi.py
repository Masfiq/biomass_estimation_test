# Cuts a held-out TEST region immediately south of california_north_10.
#
# Purpose: california_north_10 is the TRAIN region. The current train/val split is a
# random shuffle of shots inside that one strip, so validation shots sit a median of
# ~57 m from a training shot and their 5x5 patches overlap by 48-64%. That measures
# interpolation, not generalisation. A physically separate strip gives a genuine
# "never seen this ground" test set.
#
# Why directly SOUTH and the same width:
# the biomass gradient in this area runs east-west (coast ~101 Mg/ha -> Modoc high
# desert ~14 Mg/ha). A test region covering a DIFFERENT longitude range would confound
# "can it generalise spatially" with "can it extrapolate to a biomass regime it never
# saw". Keeping the same lon span means the test strip samples the same gradient.
#
# Why the buffer gap:
# spatial autocorrelation does not stop at a polygon edge. Without a gap, shots just
# inside the test boundary are still ~tens of metres from training shots, which is the
# exact leakage this is meant to eliminate.

import geopandas as gpd
from shapely.geometry import box

GEOJSON_DIR = "/s/chopin/e/proj/hyperspec/masfiq/biomass_estimation_test/geojson_files"

TRAIN_ROI = f"{GEOJSON_DIR}/california_north_10.geojson"
LAND_MASK = f"{GEOJSON_DIR}/california_mainland.geojson"
OUT_PATH  = f"{GEOJSON_DIR}/california_north_10_south_test.geojson"

# ---- tune these ------------------------------------------------------------------
BUFFER_DEG = 0.05    # gap below the train ROI, ~5.5 km, to break autocorrelation
HEIGHT_DEG = 0.51    # test strip height; 0.51 matches california_north_10
CLIP_TO_LAND = True  # drop ocean so the ROI is not mostly water on the west end
# ----------------------------------------------------------------------------------


def main():
    train = gpd.read_file(TRAIN_ROI).to_crs("EPSG:4326")
    minx, miny, maxx, maxy = train.total_bounds
    print(f"TRAIN roi : lon {minx:.4f} .. {maxx:.4f}   lat {miny:.4f} .. {maxy:.4f}")

    # same longitude span, stacked directly below with a buffer gap
    test_maxy = miny - BUFFER_DEG
    test_miny = test_maxy - HEIGHT_DEG
    test_box = box(minx, test_miny, maxx, test_maxy)

    test = gpd.GeoDataFrame(geometry=[test_box], crs="EPSG:4326")

    if CLIP_TO_LAND:
        land = gpd.read_file(LAND_MASK).to_crs("EPSG:4326")
        test = gpd.clip(test, land)
        if test.empty:
            raise RuntimeError("test ROI is empty after clipping to land")

    b = test.total_bounds
    area_km2 = test.to_crs("EPSG:5070").area.sum() / 1e6   # CONUS Albers, equal-area
    train_km2 = train.to_crs("EPSG:5070").area.sum() / 1e6

    print(f"TEST  roi : lon {b[0]:.4f} .. {b[2]:.4f}   lat {b[1]:.4f} .. {b[3]:.4f}")
    print(f"gap       : {BUFFER_DEG:.3f} deg (~{BUFFER_DEG * 111.32:.1f} km) between train and test")
    print(f"area      : train {train_km2:,.0f} km2   test {area_km2:,.0f} km2")

    test.to_file(OUT_PATH, driver="GeoJSON")
    print(f"\nwrote {OUT_PATH}")

    print("\nStill required before patches can be built for this ROI:")
    print("  HLS   - MUST DOWNLOAD. Existing HLS was subset to north_10; no imagery below lat 41.48.")
    print("  DEM   - Copernicus mosaic currently covers lat 41-43; add N40 tiles if the ROI drops below 41.0.")
    print("  GEDI  - already covered (existing granules reach lat 39.0).")
    print("  S1    - already covered (existing scenes reach lat 39.66).")
    print("  NLCD / Koppen / gridMET - CONUS-wide or global, already fine.")


if __name__ == "__main__":
    main()
