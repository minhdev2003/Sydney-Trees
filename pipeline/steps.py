"""Pipeline steps for the Sydney tree count.

One function per box in the Part 2 diagram. The notebook calls them directly and
`pipeline/dag.py` wraps the same functions as Airflow tasks. Every network call is
cached under `data/cache/`, so a second run works offline and takes seconds.

Local stand-ins for production services:
    data/s3/            -> the S3 bucket (same key layout)
    data/store.duckdb   -> PostGIS (the SQL in `store_run` runs on both)
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from pyproj import Transformer
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
S3 = ROOT / "data" / "s3"
DB_PATH = ROOT / "data" / "store.duckdb"
OUTPUTS = ROOT / "outputs"

METRIC_CRS = "EPSG:7856"  # GDA2020 / MGA zone 56, metres
WEB_CRS = "EPSG:3857"

# Between Redfern station and Redfern Park; a 600 m box here crosses Redfern, Surry Hills and Chippendale.
DEMO_CENTRE = (151.2030, -33.8902)  # lon, lat

STAC_URL = "https://earth-search.aws.element84.com/v1"
SIX_EXPORT_URL = "https://maps.six.nsw.gov.au/arcgis/rest/services/public/NSW_Imagery/MapServer/export"
SIX_TILE_URL = "https://maps.six.nsw.gov.au/arcgis/rest/services/public/NSW_Imagery/MapServer/tile/{z}/{y}/{x}"
REGISTER_URL = "https://services1.arcgis.com/cNVyNtjGVZybOQWZ/arcgis/rest/services/Trees/FeatureServer/0/query"
SUBURBS_URL = (
    "https://portal.spatial.nsw.gov.au/server/rest/services/"
    "NSW_Administrative_Boundaries_Theme/FeatureServer/2/query"
)

# Sentinel-2 scene classification (SCL) values treated as unusable.
SCL_MASK = {0: "no data", 1: "saturated", 3: "cloud shadow", 8: "cloud (medium)", 9: "cloud (high)", 10: "cirrus"}

HTTP = requests.Session()
HTTP.headers["User-Agent"] = "lynxx-trees-demo/0.1"


def _to_metric() -> Transformer:
    return Transformer.from_crs("EPSG:4326", METRIC_CRS, always_xy=True)


def _from_metric() -> Transformer:
    return Transformer.from_crs(METRIC_CRS, "EPSG:4326", always_xy=True)


def demo_area(half_size_m: float = 300, centre: tuple[float, float] = DEMO_CENTRE) -> tuple[float, float, float, float]:
    """Square lon/lat bbox of `2 * half_size_m` metres a side around `centre`."""
    x, y = _to_metric().transform(*centre)
    back = _from_metric()
    minx, miny = back.transform(x - half_size_m, y - half_size_m)
    maxx, maxy = back.transform(x + half_size_m, y + half_size_m)
    return (round(minx, 6), round(miny, 6), round(maxx, 6), round(maxy, 6))


def _key(bbox) -> str:
    return hashlib.md5(",".join(f"{v:.6f}" for v in bbox).encode()).hexdigest()[:8]


def _get(url: str, params: dict, retries: int = 4, **kw) -> requests.Response:
    for attempt in range(retries):
        try:
            r = HTTP.get(url, params=params, timeout=60, **kw)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * 2**attempt)
    raise RuntimeError("unreachable")


# --------------------------------------------------------------------------- 1. Data sources

def search_sentinel(bbox, month: str = "2026-09", max_cloud: float = 80):
    """STAC search of Earth Search `sentinel-2-l2a` for one calendar month (HTTPS/JSON)."""
    import pystac
    from pystac_client import Client

    cache = CACHE / f"stac_sentinel-2-l2a_{month}_{_key(bbox)}.json"
    if cache.exists():
        return pystac.ItemCollection.from_file(str(cache))

    start = pd.Timestamp(f"{month}-01")
    end = start + pd.offsets.MonthEnd(0)
    search = Client.open(STAC_URL).search(
        collections=["sentinel-2-l2a"],
        bbox=list(bbox),
        datetime=f"{start:%Y-%m-%d}/{end:%Y-%m-%d}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
        sortby=[{"field": "properties.datetime", "direction": "asc"}],
    )
    items = search.item_collection()
    cache.parent.mkdir(parents=True, exist_ok=True)
    items.save_object(str(cache))
    return items


def load_register(bbox) -> gpd.GeoDataFrame:
    """City of Sydney tree register points inside `bbox` (ArcGIS REST, paged GeoJSON)."""
    cache = CACHE / f"register_{'_'.join(map(str, bbox))}.geojson"
    if cache.exists():
        return gpd.read_file(cache)

    features, offset, page = [], 0, 2000
    while True:
        r = _get(REGISTER_URL, {
            "where": "1=1",
            "geometry": ",".join(map(str, bbox)),
            "geometryType": "esriGeometryEnvelope",
            "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "asset_id,TreeType,SpeciesName,CommonName,TreeHeight,TreeCanopyNS,Tree_Status,Tree_Age,DBH_in_cm",
            "outSR": 4326,
            "resultOffset": offset,
            "resultRecordCount": page,
            "orderByFields": "OBJECTID",
            "f": "geojson",
        })
        batch = r.json()["features"]
        features += batch
        if len(batch) < page:
            break
        offset += page

    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    cache.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(cache, driver="GeoJSON")
    return gdf


def load_suburbs(bbox) -> gpd.GeoDataFrame:
    """NSW suburb polygons intersecting `bbox`, plus the part of each inside the demo area."""
    cache = CACHE / f"suburbs_{'_'.join(map(str, bbox))}.geojson"
    if cache.exists():
        gdf = gpd.read_file(cache)
    else:
        r = _get(SUBURBS_URL, {
            "where": "1=1",
            "geometry": ",".join(map(str, bbox)),
            "geometryType": "esriGeometryEnvelope",
            "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "suburbname,postcode",
            "outSR": 4326,
            "f": "geojson",
        })
        gdf = gpd.GeoDataFrame.from_features(r.json()["features"], crs="EPSG:4326")
        cache.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(cache, driver="GeoJSON")

    gdf = gdf.rename(columns={"suburbname": "suburb"})
    gdf["suburb"] = gdf["suburb"].str.title()
    clipped = gdf.clip(box(*bbox))
    clipped = clipped[~clipped.is_empty].copy()
    clipped["area_ha"] = (clipped.to_crs(METRIC_CRS).area / 10_000).round(2)
    return clipped[clipped["area_ha"] > 0.5][["suburb", "postcode", "area_ha", "geometry"]].reset_index(drop=True)


# --------------------------------------------------------------------------- 2. Ingestion

@dataclass
class NdviResult:
    path: Path
    composite: np.ndarray
    transform: object
    crs: object
    scenes: list[dict] = field(default_factory=list)


def _read_s2_window(item, bounds_4326) -> dict:
    """Read red, nir and SCL for a small window straight from the remote COGs (HTTP range requests)."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform_bounds
    from rasterio.windows import bounds as window_bounds
    from rasterio.windows import from_bounds

    from affine import Affine
    from rasterio.crs import CRS

    cache = CACHE / "sentinel2" / f"{item.id}_{_key(bounds_4326)}.npz"
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files} | {"transform": Affine(*z["transform"]),
                                             "crs": CRS.from_wkt(str(z["crs"]))}

    env = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="1",
               CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif")
    with rasterio.Env(**env):
        with rasterio.open(item.assets["red"].href) as src:
            native = transform_bounds("EPSG:4326", src.crs, *bounds_4326)
            win = from_bounds(*native, src.transform).round_offsets().round_lengths()
            red = src.read(1, window=win)
            transform, crs = src.window_transform(win), src.crs
            win_bounds = window_bounds(win, src.transform)
        with rasterio.open(item.assets["nir"].href) as src:
            nir = src.read(1, window=from_bounds(*win_bounds, src.transform).round_offsets().round_lengths())
        with rasterio.open(item.assets["scl"].href) as src:
            scl = src.read(1, window=from_bounds(*win_bounds, src.transform), out_shape=red.shape,
                           resampling=Resampling.nearest)

    def scale_offset(asset):
        band = (item.assets[asset].extra_fields.get("raster:bands") or [{}])[0]
        return band.get("scale", 1e-4), band.get("offset", 0.0)

    out = {"red": red, "nir": nir, "scl": scl,
           "red_so": np.array(scale_offset("red")), "nir_so": np.array(scale_offset("nir"))}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out, transform=np.array(transform[:6]), crs=np.array(crs.to_wkt()))
    return out | {"transform": transform, "crs": crs}


def make_ndvi_cog(items, bbox, month: str = "2026-09") -> NdviResult:
    """Cloud-mask every scene with SCL, compute NDVI, take the monthly median and write a COG to S3."""
    import rasterio

    scenes, stack, ref = [], [], None
    for item in items:
        w = _read_s2_window(item, bbox)
        if ref is None:
            ref = w
        if w["red"].shape != ref["red"].shape or str(w["crs"]) != str(ref["crs"]):
            continue
        red = w["red"] * w["red_so"][0] + w["red_so"][1]
        nir = w["nir"] * w["nir_so"][0] + w["nir_so"][1]
        with np.errstate(divide="ignore", invalid="ignore"):
            ndvi = ((nir - red) / (nir + red)).astype("float32")
        ndvi[(w["red"] == 0) | (w["nir"] == 0)] = np.nan
        bad = np.isin(w["scl"], list(SCL_MASK))
        masked = np.where(bad, np.nan, ndvi)
        stack.append(masked)
        scenes.append({"id": item.id, "date": item.datetime.date().isoformat(),
                       "tile_cloud_pct": round(item.properties.get("eo:cloud_cover", np.nan), 1),
                       "local_masked_pct": round(100 * bad.mean(), 1),
                       "ndvi_raw": ndvi, "ndvi_masked": masked})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        composite = np.nanmedian(np.stack(stack), axis=0).astype("float32")

    path = S3 / "raw" / "sentinel2" / month / "ndvi_median.tif"
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(driver="COG", dtype="float32", count=1, width=composite.shape[1], height=composite.shape[0],
                   crs=ref["crs"], transform=ref["transform"], nodata=np.nan, compress="DEFLATE")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(composite, 1)
    return NdviResult(path, composite, ref["transform"], ref["crs"], scenes)


def chip_grid(bbox, chip_m: float = 100, margin_m: float = 10) -> gpd.GeoDataFrame:
    """Split the demo area into square chips (EPSG:3857) with an overlap margin on each side.

    Web Mercator units shrink on the ground by cos(latitude), so sizes are scaled to keep
    chips `chip_m` ground metres wide.
    """
    to_web = Transformer.from_crs("EPSG:4326", WEB_CRS, always_xy=True)
    minx, miny = to_web.transform(bbox[0], bbox[1])
    maxx, maxy = to_web.transform(bbox[2], bbox[3])
    k = 1 / math.cos(math.radians((bbox[1] + bbox[3]) / 2))
    size, margin = chip_m * k, margin_m * k
    nx, ny = round((maxx - minx) / size), round((maxy - miny) / size)
    rows = []
    for j in range(ny):
        for i in range(nx):
            core = (minx + i * size, maxy - (j + 1) * size, minx + (i + 1) * size, maxy - j * size)
            rows.append({"chip_id": f"r{j}c{i}", "export_bounds": tuple(c + d for c, d in
                         zip(core, (-margin, -margin, margin, margin))), "geometry": box(*core)})
    return gpd.GeoDataFrame(rows, crs=WEB_CRS)


def fetch_aerial_chips(bbox, chip_m: float = 100, margin_m: float = 10, gsd_m: float = 0.10) -> gpd.GeoDataFrame:
    """Download NSW aerial imagery for each chip from the SIX Maps MapServer `/export` endpoint."""
    grid = chip_grid(bbox, chip_m, margin_m)
    px = round((chip_m + 2 * margin_m) / gsd_m)
    folder = CACHE / "aerial" / _key(bbox)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for chip in grid.itertuples():
        path = folder / f"{chip.chip_id}.jpg"
        if not path.exists():
            params = {"bbox": ",".join(f"{v:.3f}" for v in chip.export_bounds), "bboxSR": 3857, "imageSR": 3857,
                      "size": f"{px},{px}", "format": "jpg", "f": "image"}
            for attempt in range(6):
                r = _get(SIX_EXPORT_URL, params)
                if r.headers.get("Content-Type", "").startswith("image/"):
                    path.write_bytes(r.content)
                    break
                time.sleep(2 + attempt)
            else:
                raise RuntimeError(f"SIX Maps export failed for {chip.chip_id}")
        paths.append(str(path))
    grid["path"] = paths
    grid["px"] = px
    return grid


# --------------------------------------------------------------------------- 3. Detection

_MODEL = None

# DeepForest was trained on 400 px patches of forest imagery, where crowns are smaller than
# Sydney's street and park trees. At 10 cm, 800 px patches (scaled to the model's input size)
# give one box per crown; 400 px patches split large crowns into fragments.
PATCH_SIZE = 800


def _model():
    global _MODEL
    if _MODEL is None:
        from deepforest import main
        _MODEL = main.deepforest()
    return _MODEL


def predict_chip(path: str, patch_size: int = PATCH_SIZE) -> pd.DataFrame:
    """Raw DeepForest boxes for one chip, in pixel coordinates. Cached per chip and patch size."""
    cache = CACHE / "detections" / Path(path).parent.name / f"{Path(path).stem}_p{patch_size}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    df = _model().predict_tile(path=path, patch_size=patch_size, patch_overlap=0.1)
    df = pd.DataFrame(df)[["xmin", "ymin", "xmax", "ymax", "score"]] if df is not None and len(df) else \
        pd.DataFrame(columns=["xmin", "ymin", "xmax", "ymax", "score"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def detect_trees(chips: gpd.GeoDataFrame, score_threshold: float = 0.15,
                 patch_size: int = PATCH_SIZE) -> gpd.GeoDataFrame:
    """Run the detector on every chip and return one crown polygon per tree (EPSG:4326).

    A crown is kept only if its centre falls in the chip's core, so trees in the
    overlap margin are counted once.
    """
    rows = []
    for chip in chips.itertuples():
        df = predict_chip(chip.path, patch_size)
        df = df[df["score"] >= score_threshold]
        ex = chip.export_bounds
        res_x, res_y = (ex[2] - ex[0]) / chip.px, (ex[3] - ex[1]) / chip.px
        core = chip.geometry.bounds
        for d in df.itertuples():
            x0, x1 = ex[0] + d.xmin * res_x, ex[0] + d.xmax * res_x
            y0, y1 = ex[3] - d.ymax * res_y, ex[3] - d.ymin * res_y
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if core[0] <= cx < core[2] and core[1] <= cy < core[3]:
                rows.append({"chip_id": chip.chip_id, "score": round(float(d.score), 3), "geometry": box(x0, y0, x1, y1)})

    trees = gpd.GeoDataFrame(rows, columns=["chip_id", "score", "geometry"], crs=WEB_CRS)
    metric = trees.to_crs(METRIC_CRS)
    trees["crown_area_m2"] = metric.area.round(1)
    trees["crown_diameter_m"] = np.sqrt(metric.area).round(1)
    trees = trees.to_crs("EPSG:4326")
    trees.insert(0, "tree_id", [f"T{i:06d}" for i in range(len(trees))])
    return trees


# --------------------------------------------------------------------------- 3b. Lidar check

ELVIS_DOWNLOADABLES_URL = "https://api.elevation.fsdf.org.au/elevation/downloadables"
LIDAR_DIR = CACHE / "lidar"
LIDAR_CRS = "EPSG:7856"  # fallback if a tile has no CRS in its header; Sydney202005 is GDA2020 / MGA 56
LIDAR_MAX_TREE_M = 45     # taller "vegetation" is noise or rooftop planting on towers
LIDAR_GROUND = 2
LIDAR_VEGETATION = (3, 4, 5)  # low, medium and high vegetation (ASPRS classes)
LIDAR_BUILDING = 6


def find_lidar_tiles(bbox, survey_prefix: str = "Sydney") -> pd.DataFrame:
    """Classified point-cloud tiles covering `bbox`, newest survey first, from the ELVIS catalogue.

    The files sit in a Requester Pays S3 bucket, so they are either ordered free through
    https://elevation.fsdf.org.au (emailed link) or fetched with AWS credentials.
    """
    cache = CACHE / f"elvis_{_key(bbox)}.json"
    if cache.exists():
        data = json.loads(cache.read_text())
    else:
        x0, y0, x1, y1 = bbox
        polygon = f"POLYGON(({x0} {y0},{x1} {y0},{x1} {y1},{x0} {y1},{x0} {y0}))"
        from urllib.parse import quote

        r = _get(f"{ELVIS_DOWNLOADABLES_URL}?polygon={quote(polygon)}", {},  # the API rejects "+" for spaces
                 headers={"Origin": "https://elevation.fsdf.org.au", "Referer": "https://elevation.fsdf.org.au/"})
        data = r.json()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data))

    rows = []
    for source in data.get("available_data", []):
        for product, resolutions in (source.get("downloadables") or {}).items():
            if product != "Point Clouds":
                continue
            for files in (resolutions or {}).values():
                for f in files or []:
                    name = f.get("file_name", "")
                    if name.startswith(survey_prefix) and name.endswith(".laz"):
                        rows.append({"survey": name.split("-")[0], "file_name": name,
                                     "size_mb": round(float(f.get("file_size") or 0) / 1e6), "url": f.get("file_url")})
    tiles = pd.DataFrame(rows, columns=["survey", "file_name", "size_mb", "url"])
    if tiles.empty:
        return tiles
    newest = tiles["survey"].max()
    return tiles[tiles["survey"] == newest].sort_values("file_name").reset_index(drop=True)


@dataclass
class LidarResult:
    path: Path
    survey: str
    height: np.ndarray    # vegetation height above ground (m), 0 where there is no vegetation
    building: np.ndarray  # True where building points were returned
    transform: object
    crs: str


def lidar_rasters(bbox, tiles: pd.DataFrame, res_m: float = 1.0, folder: Path = LIDAR_DIR) -> LidarResult | None:
    """Rasterise classified lidar into a vegetation height model and a building mask (2-band COG).

    Ground height per cell is the lowest ground return, with gaps filled from neighbours.
    Vegetation height is the highest vegetation return minus ground. Returns None, with
    instructions, if the tiles have not been downloaded yet.
    """
    import rasterio
    from rasterio.fill import fillnodata
    from rasterio.transform import from_origin
    from rasterio.warp import transform_bounds

    survey = tiles["survey"].iloc[0] if len(tiles) else "unknown"
    out = S3 / "raw" / "lidar" / survey / f"chm_{_key(bbox)}.tif"
    if out.exists():
        with rasterio.open(out) as src:
            return LidarResult(out, survey, src.read(1), src.read(2).astype(bool), src.transform, str(src.crs))

    # ELVIS downloads unpack into nested folders ("NSW Government - Spatial Services/Point Clouds/AHD/").
    found = {p.name: p for p in folder.rglob("*.laz")} if folder.exists() else {}
    paths = [found.get(name, folder / name) for name in tiles["file_name"]]
    missing = [p.name for p in paths if not p.exists()]
    if not paths or missing:
        print(f"Lidar tiles not found in {folder}. Order them free from https://elevation.fsdf.org.au "
              "(draw the demo area, choose NSW Spatial Services > Point Clouds > the newest Sydney survey), "
              "then save these files there:\n  " + "\n  ".join(missing or ["(no tiles listed)"]))
        return None

    import laspy

    with laspy.open(paths[0]) as reader:
        header_crs = reader.header.parse_crs()
    crs = header_crs.to_string() if header_crs else LIDAR_CRS

    minx, miny, maxx, maxy = transform_bounds("EPSG:4326", crs, *bbox)
    minx, miny = math.floor(minx) - 10, math.floor(miny) - 10
    maxx, maxy = math.ceil(maxx) + 10, math.ceil(maxy) + 10
    width, height = round((maxx - minx) / res_m), round((maxy - miny) / res_m)
    ground = np.full((height, width), np.inf, dtype="float32")
    veg_top = np.full((height, width), -np.inf, dtype="float32")
    building = np.zeros((height, width), dtype=bool)

    for path in paths:
        with laspy.open(path) as reader:
            for pts in reader.chunk_iterator(5_000_000):
                x, y, z, cls = np.asarray(pts.x), np.asarray(pts.y), np.asarray(pts.z), np.asarray(pts.classification)
                inside = (x >= minx) & (x < maxx) & (y > miny) & (y <= maxy)
                if not inside.any():
                    continue
                x, y, z, cls = x[inside], y[inside], z[inside].astype("float32"), cls[inside]
                col = ((x - minx) / res_m).astype(int)
                row = ((maxy - y) / res_m).astype(int)
                g = cls == LIDAR_GROUND
                np.minimum.at(ground, (row[g], col[g]), z[g])
                v = np.isin(cls, LIDAR_VEGETATION)
                np.maximum.at(veg_top, (row[v], col[v]), z[v])
                b = cls == LIDAR_BUILDING
                building[row[b], col[b]] = True

    has_ground = np.isfinite(ground)
    ground = fillnodata(np.where(has_ground, ground, 0).astype("float32"), mask=has_ground.astype("uint8"),
                        max_search_distance=200)
    veg_height = np.where(np.isfinite(veg_top), np.clip(veg_top - ground, 0, None), 0).astype("float32")
    veg_height[veg_height > LIDAR_MAX_TREE_M] = 0

    transform = from_origin(minx, maxy, res_m, res_m)
    out.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(driver="COG", dtype="float32", count=2, width=width, height=height, crs=crs,
                   transform=transform, compress="DEFLATE")
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(veg_height, 1)
        dst.write(building.astype("float32"), 2)
        dst.set_band_description(1, "vegetation height above ground (m)")
        dst.set_band_description(2, "building returns (1 = yes)")
    return LidarResult(out, survey, veg_height, building, transform, crs)


def lidar_check(trees: gpd.GeoDataFrame, lidar: LidarResult, min_height_m: float = 3.0,
                min_canopy_pct: float = 25.0, canopy_m: float = 2.0) -> gpd.GeoDataFrame:
    """Add lidar evidence to each detected crown and flag the ones that are not trees.

    `canopy_pct` is the share of the box covered by vegetation taller than `canopy_m`.
    A crown passes (`lidar_ok`) if that share is at least `min_canopy_pct` and its tallest
    vegetation reaches `min_height_m`. Roofs fail because lidar classifies them as building,
    and shrubs fail on height. Trees overhanging a roof still pass, because lidar returns
    from the leaves are classified as vegetation.
    """
    from rasterio.windows import from_bounds

    boxes = trees.to_crs(lidar.crs)
    rows = []
    for geom in boxes.geometry:
        win = from_bounds(*geom.bounds, lidar.transform).round_offsets().round_lengths()
        r0, c0 = max(int(win.row_off), 0), max(int(win.col_off), 0)
        h = lidar.height[r0:r0 + int(win.height), c0:c0 + int(win.width)]
        b = lidar.building[r0:r0 + int(win.height), c0:c0 + int(win.width)]
        if h.size == 0:
            rows.append((0.0, 0.0, 0.0))
            continue
        rows.append((float(h.max()), 100 * float((h >= canopy_m).mean()), 100 * float(b.mean())))

    out = trees.copy()
    out["lidar_height_m"] = [round(r[0], 1) for r in rows]
    out["canopy_pct"] = [round(r[1]) for r in rows]
    out["building_pct"] = [round(r[2]) for r in rows]
    out["lidar_ok"] = (out["lidar_height_m"] >= min_height_m) & (out["canopy_pct"] >= min_canopy_pct)
    return out


# --------------------------------------------------------------------------- 4. Validation and QA

@dataclass
class Validation:
    precision: float
    recall: float
    f1: float
    matched: int
    register_trees: int
    detections_in_zone: int
    match_m: float
    matches: gpd.GeoDataFrame
    zone: gpd.GeoSeries

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("precision", "recall", "f1", "matched", "register_trees", "detections_in_zone", "match_m")}


def validate(trees: gpd.GeoDataFrame, register: gpd.GeoDataFrame, match_m: float = 5.0,
             zone_m: float = 6.0) -> Validation:
    """Compare detected crown centres with council register trees (public land only).

    Each register tree is matched to at most one detection within `match_m` metres
    (closest first). The register records the trunk and the model the crown centre, which
    can be a few metres apart for street trees leaning over the road.
    Recall uses every register tree. Precision is measured only inside
    the area the register covers (`zone_m` around register trees), because private
    trees are not in the register and would otherwise look like false positives.
    """
    reg = register[register["Tree_Status"].fillna("Tree").eq("Tree")].to_crs(METRIC_CRS)
    reg = reg[["asset_id", "geometry"]].reset_index(drop=True)
    det = trees.to_crs(METRIC_CRS)
    det = gpd.GeoDataFrame(det[["tree_id"]], geometry=det.centroid, crs=METRIC_CRS)

    pairs = gpd.sjoin_nearest(reg, det, how="inner", max_distance=match_m, distance_col="distance_m")
    pairs = pairs.sort_values("distance_m").drop_duplicates("tree_id").drop_duplicates("asset_id")

    zone = reg.buffer(zone_m).union_all()
    in_zone = int(det.within(zone).sum())
    tp = len(pairs)
    precision = tp / in_zone if in_zone else 0.0
    recall = tp / len(reg) if len(reg) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return Validation(round(precision, 3), round(recall, 3), round(f1, 3), tp, len(reg), in_zone, match_m,
                      pairs, gpd.GeoSeries([zone], crs=METRIC_CRS))


def qa_gate(metrics: Validation | dict, min_precision: float = 0.8, min_recall: float = 0.4) -> bool:
    """Return True if the run may be published. Airflow uses this as a ShortCircuitOperator.

    The defaults sit just below the baseline run, so the gate catches drift (bad imagery,
    a broken model release) rather than demanding accuracy the model has never had.
    """
    m = metrics.as_dict() if isinstance(metrics, Validation) else metrics
    failures = [f"{name} {m[name]:.2f} < {limit:.2f}" for name, limit in
                (("precision", min_precision), ("recall", min_recall)) if m[name] < limit]
    if failures:
        print("QA gate: BLOCKED -", "; ".join(failures), "- nothing is published")
        return False
    print(f"QA gate: PASS - precision {m['precision']:.2f} >= {min_precision:.2f}, "
          f"recall {m['recall']:.2f} >= {min_recall:.2f}")
    return True


# --------------------------------------------------------------------------- 5. Storage

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id       VARCHAR PRIMARY KEY,
    run_at       TIMESTAMP,
    imagery      VARCHAR,
    n_trees      INTEGER,
    precision    DOUBLE,
    recall       DOUBLE,
    passed       BOOLEAN
);
CREATE TABLE IF NOT EXISTS suburbs (
    suburb   VARCHAR PRIMARY KEY,
    postcode INTEGER,
    area_ha  DOUBLE,
    geom     GEOMETRY
);
CREATE TABLE IF NOT EXISTS trees (
    tree_id          VARCHAR,
    run_id           VARCHAR,
    score            DOUBLE,
    crown_area_m2    DOUBLE,
    crown_diameter_m DOUBLE,
    geom             GEOMETRY
);
"""

SUBURB_STATS_SQL = """
CREATE OR REPLACE VIEW suburb_stats AS
WITH latest AS (
    SELECT run_id FROM runs WHERE passed ORDER BY run_at DESC LIMIT 1
)
SELECT s.suburb,
       s.postcode,
       s.area_ha,
       COUNT(t.tree_id)                                         AS trees,
       ROUND(COUNT(t.tree_id) / s.area_ha, 1)                   AS trees_per_ha,
       ROUND(COALESCE(SUM(t.crown_area_m2), 0) / (s.area_ha * 100), 1) AS crown_cover_pct,
       (SELECT run_id FROM latest)                              AS run_id
FROM suburbs s
LEFT JOIN trees t
       ON t.run_id = (SELECT run_id FROM latest)
      AND ST_Within(ST_Centroid(t.geom), s.geom)
GROUP BY s.suburb, s.postcode, s.area_ha
ORDER BY trees DESC
"""


def connect(read_only: bool = False, path: Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path), read_only=read_only)
    con.install_extension("spatial")
    con.load_extension("spatial")
    return con


def store_run(run_id: str, trees: gpd.GeoDataFrame, suburbs: gpd.GeoDataFrame, metrics: Validation,
              passed: bool, imagery: str = "NSW aerial (SIX Maps)") -> dict:
    """Write the run to the spatial database and, if it passed QA, a GeoParquet snapshot to S3."""
    con = connect()
    try:
        con.execute(SCHEMA_SQL)
        con.execute("DELETE FROM trees WHERE run_id = ?", [run_id])
        con.execute("DELETE FROM runs WHERE run_id = ?", [run_id])
        con.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [run_id, datetime.now(timezone.utc).replace(tzinfo=None), imagery,
                     len(trees) if passed else None, metrics.precision, metrics.recall, passed])

        sub = pd.DataFrame(suburbs.drop(columns="geometry"))
        sub["wkb"] = suburbs.geometry.to_wkb()
        con.execute("DELETE FROM suburbs")
        con.execute("INSERT INTO suburbs SELECT suburb, postcode, area_ha, ST_GeomFromWKB(wkb) FROM sub")

        snapshot = None
        if passed:
            t = pd.DataFrame(trees[["tree_id", "score", "crown_area_m2", "crown_diameter_m"]])
            t["run_id"] = run_id
            t["wkb"] = trees.geometry.to_wkb()
            con.execute("""INSERT INTO trees SELECT tree_id, run_id, score, crown_area_m2, crown_diameter_m,
                           ST_GeomFromWKB(wkb) FROM t""")
            snapshot = S3 / "snapshots" / run_id / "trees.parquet"
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            trees.assign(run_id=run_id).to_parquet(snapshot)

        con.execute(SUBURB_STATS_SQL)
    finally:
        con.close()
    return {"db": DB_PATH, "snapshot": snapshot}


# --------------------------------------------------------------------------- 8. Open data

def export_open_data(run_id: str, trees: gpd.GeoDataFrame, suburb_stats: gpd.GeoDataFrame,
                     base_url: str = "https://data.example.org/sydney-trees") -> dict:
    """Write GeoJSON, CSV and GeoParquet for the run plus a `latest.json` pointer."""
    folder = OUTPUTS / "open-data" / run_id
    folder.mkdir(parents=True, exist_ok=True)

    centres = trees.to_crs(METRIC_CRS).centroid.to_crs("EPSG:4326")
    table = pd.DataFrame(trees.drop(columns="geometry")).assign(
        lon=centres.x.round(7), lat=centres.y.round(7), run_id=run_id)

    files = {
        "trees.geojson": lambda p: trees.assign(run_id=run_id).to_file(p, driver="GeoJSON",
                                                                        COORDINATE_PRECISION=7),
        "trees.csv": lambda p: table.to_csv(p, index=False),
        "trees.parquet": lambda p: trees.assign(run_id=run_id).to_parquet(p),
        "suburbs.geojson": lambda p: suburb_stats.to_file(p, driver="GeoJSON", COORDINATE_PRECISION=7),
        "suburbs.csv": lambda p: pd.DataFrame(suburb_stats.drop(columns="geometry")).to_csv(p, index=False),
    }
    for name, write in files.items():
        p = folder / name
        p.unlink(missing_ok=True)
        write(p)

    latest = {
        "run_id": run_id,
        "published_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_trees": len(trees),
        "files": {name: f"{base_url}/{run_id}/{name}" for name in files},
        "licence": "CC BY 4.0",
    }
    (OUTPUTS / "open-data" / "latest.json").write_text(json.dumps(latest, indent=2))
    return {"folder": folder, "latest": latest}


def suburb_stats_gdf(con: duckdb.DuckDBPyConnection | None = None) -> gpd.GeoDataFrame:
    """The `suburb_stats` view joined back to suburb geometry, as a GeoDataFrame."""
    own = con is None
    con = con or connect(read_only=True)
    try:
        df = con.execute("""SELECT st.*, ST_AsWKB(s.geom) AS wkb FROM suburb_stats st
                            JOIN suburbs s USING (suburb)""").df()
    finally:
        if own:
            con.close()
    geom = gpd.GeoSeries.from_wkb(df.pop("wkb").map(bytes), crs="EPSG:4326")
    return gpd.GeoDataFrame(df, geometry=geom)


def to_geojson(gdf: gpd.GeoDataFrame, precision: int = 7) -> dict:
    """Compact GeoJSON dict (rounded coordinates) for embedding in HTML."""
    data = json.loads(gdf.to_crs("EPSG:4326").to_json(drop_id=True))

    def rnd(c):
        return [rnd(x) for x in c] if isinstance(c[0], (list, tuple)) else [round(v, precision) for v in c]

    for f in data["features"]:
        f["geometry"]["coordinates"] = rnd(f["geometry"]["coordinates"])
    return data
