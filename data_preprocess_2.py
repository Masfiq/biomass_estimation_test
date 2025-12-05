# gedi_hls_quadtile_3x3.py
import os, glob, math
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform_bounds
from rasterio.transform import rowcol
from shapely.geometry import Point, box
from pyproj import Transformer
import h5py
import gc  # NEW



# CONFIG

ROOT       = r"/../dataset"  # ← change if needed
GEDI_DIR   = os.path.join(ROOT, "gedi_l4a_california_mainland_april_2021")
HLS_DIR    = os.path.join(ROOT, "hls_california")
PATCH_DIR  = os.path.join(ROOT, "out_patches")
META_DIR   = os.path.join(ROOT, "out_meta")

# GEDI footprint radius (meters). ~12.5 m for ~25 m diameter footprint
RADIUS_M = 12.5

# Filters
GOOD_L2 = True   # require l2_quality_flag == 1
GOOD_L4 = True   # require l4_quality_flag == 1

# Quadtile zoom (Web Mercator). 12 ≈ ~10 km cells; 14 ≈ ~2.5 km
ZOOM = 12
INCLUDE_NEIGHBOR_QK = True  # include 8 neighboring quadkeys to catch edge cases

# File search patterns
RECURSIVE_GEDI = False  # set True if GEDI H5 live in nested subfolders
RECURSIVE_HLS  = False  # set True if HLS TIF live in nested subfolders

# --- memory tuning ---
GDAL_CACHE_MB = 256            # keep GDAL cache modest
FLUSH_EVERY   = 5000           # write CSVs every N patches and clear RAM


# ======================
# UTILS: Quadkey helpers
# ======================
def _lonlat_to_tilexy(lon, lat, zoom):
    lat = max(min(lat, 85.05112878), -85.05112878)
    n = 2 ** zoom
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - (math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi)) / 2.0 * n
    return int(x), int(y)

def _tilexy_to_quadkey(x, y, zoom):
    qk = []
    for i in range(zoom, 0, -1):
        mask = 1 << (i - 1)
        digit = 0
        if (x & mask) != 0: digit += 1
        if (y & mask) != 0: digit += 2
        qk.append(str(digit))
    return ''.join(qk)

def lonlat_to_quadkey(lon, lat, zoom):
    x, y = _lonlat_to_tilexy(lon, lat, zoom)
    return _tilexy_to_quadkey(x, y, zoom), x, y

def neighbors_quadkeys(x, y, zoom):
    n = 2 ** zoom
    out = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0: 
                continue
            xx, yy = x + dx, y + dy
            if 0 <= xx < n and 0 <= yy < n:
                out.append(_tilexy_to_quadkey(xx, yy, zoom))
    return out

# ======================
# Index HLS tiles with quadkeys
# ======================
def index_hls_tiles_with_quadkeys(hls_dir, zoom, recursive=False):
    pattern = "**/*.tif" if recursive else "*.tif"
    tiles = []
    qk_to_tiles = {}   # quadkey -> list of tile dicts

    for path in glob.glob(os.path.join(hls_dir, pattern), recursive=recursive):
        try:
            with rasterio.open(path) as ds:
                crs = ds.crs
                b = ds.bounds
                b84 = transform_bounds(crs, "EPSG:4326", b.left, b.bottom, b.right, b.top, densify_pts=21)
                tinfo = {
                    "path": path,
                    "crs": crs,
                    "bounds": b,            # raster CRS
                    "bounds_wgs84": b84,    # lon/lat
                    "transform": ds.transform,
                    "width": ds.width,
                    "height": ds.height,
                    "count": ds.count,
                    "res": ds.transform.a
                }
                tiles.append(tinfo)

                # Cover this HLS tile's WGS84 bbox with quadkeys at `zoom`
                minlon, minlat, maxlon, maxlat = b84
                x0, y0 = _lonlat_to_tilexy(minlon, maxlat, zoom)  # top-left
                x1, y1 = _lonlat_to_tilexy(maxlon, minlat, zoom)  # bottom-right
                for x in range(min(x0, x1), max(x0, x1) + 1):
                    for y in range(min(y0, y1), max(y0, y1) + 1):
                        qk = _tilexy_to_quadkey(x, y, zoom)
                        qk_to_tiles.setdefault(qk, []).append(tinfo)
        except Exception as e:
            print(f"[WARN] Could not open {path}: {e}")

    return tiles, qk_to_tiles

def candidate_tiles_for_lonlat(lon, lat, zoom, qk_to_tiles, include_neighbors=True):
    qk, x, y = lonlat_to_quadkey(lon, lat, zoom)
    cands = list(qk_to_tiles.get(qk, []))
    if include_neighbors:
        for nqk in neighbors_quadkeys(x, y, zoom):
            if nqk in qk_to_tiles:
                cands.extend(qk_to_tiles[nqk])
    # de-dup by path
    seen, unique = set(), []
    for t in cands:
        p = t["path"]
        if p not in seen:
            unique.append(t)
            seen.add(p)
    return unique

# ======================
# GEDI shot iterator
# ======================



    
def iter_gedi_shots(h5_path, block_size=50000):
    """
    Stream GEDI shots without loading full arrays in memory.
    Yields (lon, lat, shot_number, l2q, l4q, beam_name).
    """
    with h5py.File(h5_path, "r") as f:
        for beam_name in [k for k in f.keys() if k.startswith("BEAM")]:
            g = f[beam_name]
            if "lon_lowestmode" not in g or "lat_lowestmode" not in g:
                continue

            d_lon = g["lon_lowestmode"]       # h5py.Dataset
            d_lat = g["lat_lowestmode"]
            d_l2q = g.get("l2_quality_flag", None)
            d_l4q = g.get("l4_quality_flag", None)
            if "geolocation" in g and "shot_number" in g["geolocation"]:
                d_shot = g["geolocation"]["shot_number"]
            else:
                d_shot = None

            n = d_lon.shape[0]
            for start in range(0, n, block_size):
                end = min(start + block_size, n)
                # slice into reasonably sized NumPy arrays
                lon  = d_lon[start:end][...]  # small block
                lat  = d_lat[start:end][...]
                l2q  = d_l2q[start:end][...] if d_l2q is not None else np.ones(end-start, dtype=np.uint8)
                l4q  = d_l4q[start:end][...] if d_l4q is not None else np.ones(end-start, dtype=np.uint8)
                shot = d_shot[start:end][...] if d_shot is not None else np.arange(start, end, dtype=np.int64)

                for i in range(end - start):
                    yield float(lon[i]), float(lat[i]), int(shot[i]), int(l2q[i]), int(l4q[i]), beam_name

                # free block arrays eagerly
                del lon, lat, l2q, l4q, shot
                gc.collect()


# ======================
# Patch extraction using candidate tiles (edge-friendly)
# ======================
import re

def _safe_filename(s: str) -> str:
    # replace characters illegal on Windows: \ / : * ? " < > |
    return re.sub(r'[\\/:*?"<>|]+', '_', s)

def extract_patch_edge_friendly(lon, lat, shot_id, candidates, radius_m, patch_rows, pix_rows):
    """
    Pick the candidate HLS tile with the largest intersection area with the GEDI circle,
    then choose the nearest in-bounds 3×3 window (clamped). Write patch & record overlaps.
    """
    import os, re
    import rasterio
    from rasterio.windows import Window
    from shapely.geometry import Point, box
    from pyproj import Transformer

    def _safe_filename(s: str) -> str:
        # replace characters illegal on Windows: \ / : * ? " < > |
        return re.sub(r'[\\/:*?"<>|]+', '_', s)

    # 1) choose best tile by intersection area
    best = None
    best_area = 0.0
    best_circle = None
    for t in candidates:
        transformer = Transformer.from_crs("EPSG:4326", t["crs"], always_xy=True)
        x_tile, y_tile = transformer.transform(lon, lat)
        circle = Point(x_tile, y_tile).buffer(radius_m, resolution=64)
        tp = box(t["bounds"].left, t["bounds"].bottom, t["bounds"].right, t["bounds"].top)
        ia = circle.intersection(tp).area
        if ia > best_area:
            best_area = ia
            best = t
            best_circle = circle

    if best is None or best_area <= 0:
        return False

    # 2) snap & clamp to nearest valid 3×3 window and compute overlaps
    with rasterio.open(best["path"]) as ds:
        transformer = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
        x_tile, y_tile = transformer.transform(lon, lat)

        inv = ~ds.transform
        colf, rowf = inv * (x_tile, y_tile)
        col_c = int(round(colf))
        row_c = int(round(rowf))
        if ds.width < 3 or ds.height < 3:
            return False
        # clamp to keep a full 3×3 inside
        col_c = max(1, min(ds.width  - 2, col_c))
        row_c = max(1, min(ds.height - 2, row_c))

        win = Window(col_off=col_c - 1, row_off=row_c - 1, width=3, height=3)
        patch = ds.read(window=win)  # (bands, 3, 3)

        area_circle = float(best_circle.area)

        def cell_poly(rr, cc):
            x0, y0 = rasterio.transform.xy(ds.transform, rr,   cc,   offset='ul')
            x1, y1 = rasterio.transform.xy(ds.transform, rr+1, cc+1, offset='ul')
            return box(x0, y1, x1, y0)

        # per-pixel records (9 cells)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                rr = row_c + dy
                cc = col_c + dx
                poly = cell_poly(rr, cc)
                inter_area = poly.intersection(best_circle).area
                frac = inter_area / area_circle if area_circle > 0 else 0.0
                vals = patch[:, dy+1, dx+1].tolist()
                rec = {
                    "shot_id": shot_id,
                    "tile": os.path.basename(best["path"]),
                    "center_row": row_c, "center_col": col_c,
                    "row": rr, "col": cc,
                    "overlap_frac": frac
                }
                for b in range(ds.count):
                    rec[f"band{b+1}"] = vals[b]
                pix_rows.append(rec)

        # write 3×3 GeoTIFF (preserve georeferencing) — safe filename + explicit driver
        patch_transform = rasterio.windows.transform(win, ds.transform)
        name_core = os.path.splitext(os.path.basename(best["path"]))[0]
        safe_shot = _safe_filename(shot_id)
        out_name = f"{safe_shot}__{name_core}__r{row_c}_c{col_c}_3x3.tif"
        out_path = os.path.join(PATCH_DIR, out_name)
        os.makedirs(PATCH_DIR, exist_ok=True)

        profile = ds.profile.copy()
        profile.update({
            "driver": "GTiff",
            "height": 3,
            "width": 3,
            "transform": patch_transform,
            # Optional: compression & safety
            # "compress": "LZW",
            # "BIGTIFF": "IF_SAFER",
        })
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(patch)

        patch_rows.append({
            "shot_id": shot_id,
            "tile": os.path.basename(best["path"]),
            "patch_path": out_path,
            "center_x": x_tile, "center_y": y_tile,
            "center_row": row_c, "center_col": col_c,
            "bands": ds.count,
            "pix_res_m": best["res"]
        })
        return True


# ======================
# MAIN
# ======================
def main():
    os.makedirs(PATCH_DIR, exist_ok=True)
    os.makedirs(META_DIR, exist_ok=True)

    # Index HLS with quadkeys
    tiles, qk_to_tiles = index_hls_tiles_with_quadkeys(HLS_DIR, ZOOM, recursive=RECURSIVE_HLS)
    print(f"[HLS] tiles indexed: {len(tiles)} | quadkey cells: {len(qk_to_tiles)}")

    # Find GEDI files
    gedi_pattern = "**/*.h5" if RECURSIVE_GEDI else "*.h5"
    h5_files = sorted(glob.glob(os.path.join(GEDI_DIR, gedi_pattern), recursive=RECURSIVE_GEDI))
    print(f"[GEDI] files found: {len(h5_files)}")
    if not h5_files:
        return

    # CSV paths
    patch_csv = os.path.join(META_DIR, "patch_index.csv")
    pix_csv   = os.path.join(META_DIR, "pixel_intersections.csv")
    # If files exist, we'll append; if not, first write will include header
    wrote_patch_header = os.path.exists(patch_csv)
    wrote_pix_header   = os.path.exists(pix_csv)

    patch_rows, pix_rows = [], []
    since_flush = 0

    grand_total = 0
    grand_kept  = 0

    # Keep GDAL cache modest to reduce working set
    with rasterio.Env(GDAL_CACHEMAX=GDAL_CACHE_MB):
        for h5 in h5_files:
            base = os.path.basename(h5)
            print(f"\n[GEDI] Processing {base}")
            shots_total = 0
            shots_kept  = 0

            for lon, lat, shot_no, l2q, l4q, beam_name in iter_gedi_shots(h5):
                shots_total += 1
                grand_total += 1

                if GOOD_L2 and l2q != 1: 
                    continue
                if GOOD_L4 and l4q != 1:
                    continue

                shot_id = f"{base}:{beam_name}:{shot_no}"

                candidates = candidate_tiles_for_lonlat(
                    lon, lat, ZOOM, qk_to_tiles, include_neighbors=INCLUDE_NEIGHBOR_QK
                )
                if not candidates:
                    continue

                ok = extract_patch_edge_friendly(
                    lon, lat, shot_id, candidates, RADIUS_M, patch_rows, pix_rows
                )
                if ok:
                    shots_kept += 1
                    grand_kept += 1
                    since_flush += 1

                # -------- periodic flush to CSV to free RAM --------
                if since_flush >= FLUSH_EVERY:
                    if patch_rows:
                        dfp = pd.DataFrame(patch_rows)
                        dfp.to_csv(patch_csv, mode='a', index=False, header=not wrote_patch_header)
                        wrote_patch_header = True
                    if pix_rows:
                        dfx = pd.DataFrame(pix_rows)
                        dfx.to_csv(pix_csv, mode='a', index=False, header=not wrote_pix_header)
                        wrote_pix_header = True
                    patch_rows.clear()
                    pix_rows.clear()
                    since_flush = 0
                    gc.collect()
                # ----------------------------------------------------

            print(f"[{base}] shots total={shots_total}, kept={shots_kept}")

        # final flush
        if patch_rows:
            pd.DataFrame(patch_rows).to_csv(patch_csv, mode='a', index=False, header=not wrote_patch_header)
        if pix_rows:
            pd.DataFrame(pix_rows).to_csv(pix_csv, mode='a', index=False, header=not wrote_pix_header)
        patch_rows.clear(); pix_rows.clear()
        gc.collect()

    print(f"\n[SUMMARY] GEDI shots total={grand_total}, kept={grand_kept}")
    # Count rows without loading whole CSVs (optional)
    try:
        pc = sum(1 for _ in open(patch_csv, 'rb')) - 1 if os.path.exists(patch_csv) else 0
        xc = sum(1 for _ in open(pix_csv, 'rb')) - 1 if os.path.exists(pix_csv) else 0
        print(f"[OUTPUT] patch rows: {pc} | pixel rows: {xc} (~9× patches)")
    except Exception:
        pass

if __name__ == "__main__":
    main()
