"""Generate notebooks/part2_walkthrough.ipynb from the cells below.

    uv run python scripts/build_notebook.py
    uv run jupyter nbconvert --to notebook --execute --inplace notebooks/part2_walkthrough.ipynb
"""

from pathlib import Path

import nbformat as nbf

cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


# --------------------------------------------------------------------------- intro

md("""
# Part 2 walkthrough: the production system, run small

This notebook follows the Part 2 system diagram from top to bottom. It runs real code on
a **600 m x 600 m area of Redfern** (between Redfern station and Redfern Park, crossing
Redfern, Surry Hills and Chippendale) and shows what each box produces.

![Part 2 system diagram](../docs/part2_diagram.png)

Each section starts by naming the diagram box, the production technology and the protocol
between systems. Production-only services are either replaced by a local equivalent with
the same interface, or shown as code without being run:

| Production | In this notebook |
|---|---|
| Apache Airflow | Same step functions called directly; the DAG is printed, not run |
| AWS Batch GPU job | DeepForest on CPU (36 chips, cached after the first run) |
| PostGIS | DuckDB with the spatial extension, same SQL |
| S3 | `data/s3/`, same key layout |
| FastAPI on containers behind CloudFront | The same FastAPI app called in-process with `TestClient` |
| Martin / pg_tileserv | The API's `/tiles/{z}/{x}/{y}.mvt` route |
| React + MapLibre GL | A standalone MapLibre GL page, `outputs/map.html` |

Every download is cached in `data/cache/`, so re-running the notebook takes seconds and
works offline.
""")

code("""
import json
import logging
import sys
import warnings
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display
from shapely.geometry import box

from pipeline import steps

from matplotlib_inline.backend_inline import set_matplotlib_formats

set_matplotlib_formats("jpeg", pil_kwargs={"quality": 82})
pd.set_option("display.width", 140)
cx.set_cache_dir(str(steps.CACHE / "tiles"))

RUN_ID = MONTH = "2026-09"            # monthly run for September 2026
BBOX = steps.demo_area()              # 600 m x 600 m, lon/lat
CONTEXT_BBOX = steps.demo_area(1000)  # 2 km x 2 km for Sentinel-2 context
DEMO = gpd.GeoSeries([box(*BBOX)], crs=4326)


def aerial(ax, zoom=18):
    cx.add_basemap(ax, source=steps.SIX_TILE_URL, zoom=zoom, attribution="Imagery (c) NSW Spatial Services")
    ax.set_axis_off()


print("demo bbox:", BBOX)
""")

# --------------------------------------------------------------------------- 1

md("""
## 1. Data sources

**Diagram box:** Satellite imagery, Aerial and Lidar, Ground truth.
**Protocol:** HTTPS for everything: STAC (JSON) for Sentinel-2, the ArcGIS REST MapServer
`/export` endpoint for aerial imagery, and ArcGIS REST FeatureServer `/query` (GeoJSON) for the
council tree register and suburb boundaries. Endpoints and limits are listed in
[docs/data-access.md](../docs/data-access.md).

**Runs locally:** yes, these are the real public services.

First, the Sentinel-2 scenes for the month from the Earth Search STAC API (searched with `pystac-client`):
""")

code("""
items = steps.search_sentinel(CONTEXT_BBOX, MONTH)
scenes = pd.DataFrame([{"scene": i.id, "date": i.datetime.date(), "tile cloud %": round(i.properties["eo:cloud_cover"], 1)}
                       for i in items])
print(f"{len(items)} Sentinel-2 L2A scenes in {MONTH} with < 80% cloud over the tile")
scenes
""")

md("""
Next, NSW aerial imagery from SIX Maps. The area is split into 36 chips of 100 m, each
requested with a 10 m margin at 10 cm per pixel (1,200 x 1,200 px). That is close to the
resolution the detection model was trained on.
""")

code("""
chips = steps.fetch_aerial_chips(BBOX)
print(f"{len(chips)} chips, {chips.px.iloc[0]} x {chips.px.iloc[0]} px each, cached in {Path(chips.path.iloc[0]).parent.relative_to(ROOT)}")

sample = chips.set_index("chip_id").loc["r2c2"]
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(plt.imread(sample.path))
ax.set_title("Chip r2c2: 120 m x 120 m at 10 cm per pixel (NSW aerial imagery)")
ax.set_axis_off()
""")

md("""
Finally, the ground truth: City of Sydney tree register points and NSW suburb boundaries
(clipped to the demo area).
""")

code("""
register = steps.load_register(BBOX)
suburbs = steps.load_suburbs(BBOX)
print(f"{len(register)} register trees in the demo area")
display(register["TreeType"].value_counts().rename("register trees").to_frame())
display(suburbs.drop(columns="geometry"))

fig, ax = plt.subplots(figsize=(8, 8))
suburbs.to_crs(3857).boundary.plot(ax=ax, color="white", linewidth=2)
chips.boundary.plot(ax=ax, color="white", linewidth=0.4, alpha=0.6)
register.to_crs(3857).plot(ax=ax, column="TreeType", markersize=10, legend=True,
                           legend_kwds={"loc": "lower left"})
for s in suburbs.to_crs(3857).itertuples():
    p = s.geometry.representative_point()
    ax.annotate(s.suburb, (p.x, p.y), color="white", weight="bold", ha="center", fontsize=11)
aerial(ax, zoom=17)
ax.set_title("Register trees, suburbs and the 36-chip grid")
plt.show()
""")

# --------------------------------------------------------------------------- 2

md("""
## 2. Ingestion and orchestration

**Diagram box:** Ingestion and orchestration.
**Production:** Apache Airflow runs a monthly DAG. `rasterio` reads only the needed window of
each Sentinel-2 Cloud-Optimised GeoTIFF using HTTP range requests, so a 2 km window downloads
a few hundred KB rather than the whole 100 km tile. Clouds are masked with the Sentinel-2 Scene
Classification Layer (SCL), and the result is written to S3 as a COG.
**Runs locally:** the step functions run for real and write to `data/s3/`, a local folder with the same layout as the S3 bucket.

The scene table shows the cloud cover the catalogue reports for the whole tile, next to the share of
pixels actually masked in our 2 km window:
""")

code("""
ndvi = steps.make_ndvi_cog(items, CONTEXT_BBOX, MONTH)
table = pd.DataFrame([{k: s[k] for k in ("date", "tile_cloud_pct", "local_masked_pct")} for s in ndvi.scenes])
print("masked SCL classes:", ", ".join(steps.SCL_MASK.values()))
table
""")

code("""
import rasterio
from rasterio.plot import plotting_extent

demo_native = DEMO.to_crs(ndvi.crs)
shown = max((s for s in ndvi.scenes if s["local_masked_pct"] < 90), key=lambda s: s["local_masked_pct"])
extent = plotting_extent(ndvi.composite, ndvi.transform)

fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
panels = [(shown["ndvi_raw"], f"{shown['date']}\\nbefore masking"),
          (shown["ndvi_masked"], f"{shown['date']}\\nafter SCL mask ({shown['local_masked_pct']}% removed)"),
          (ndvi.composite, f"{MONTH}\\nmedian of {len(ndvi.scenes)} masked scenes")]
for ax, (img, title) in zip(axes, panels):
    im = ax.imshow(img, cmap="RdYlGn", vmin=-0.1, vmax=0.8, extent=extent)
    demo_native.boundary.plot(ax=ax, color="black", linewidth=1.5)
    ax.set_title(title)
    ax.set_axis_off()
fig.colorbar(im, ax=axes, shrink=0.8, label="NDVI")
plt.show()

with rasterio.open(ndvi.path) as src:
    print("written:", ndvi.path.relative_to(ROOT))
    print("driver:", src.driver, "| tiled:", src.profile.get("tiled"), "| layout:", src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT"),
          "| crs:", src.crs, "| size:", src.width, "x", src.height)
""")

md("""
Sentinel-2 at 10 m cannot separate individual trees, but it is free, arrives every five days
and is used in production to flag change (canopy lost or gained since last month) and to decide
where new aerial imagery is worth processing. The black square is the demo area; trees are counted
from the aerial imagery in section 3.

In production Airflow schedules these same functions. The DAG below imports `pipeline/steps.py`.
It is shown here but not run, because Airflow does not run natively on Windows. The QA gate is a
`ShortCircuitOperator`: if validation fails, `publish` is skipped and nothing reaches the public.
""")

code("""
print((ROOT / "pipeline" / "dag.py").read_text())
""")

# --------------------------------------------------------------------------- 3

md("""
## 3. Tree detection model

**Diagram box:** Tree detection model.
**Production:** a PyTorch model (DeepForest, a RetinaNet trained on 10 cm aerial imagery) runs as
an AWS Batch job on a rented GPU. It reads chips from S3 and writes one crown per tree.
**Runs locally:** the same model on CPU. Each chip took about 13 s on first run; the results are cached per chip.

Each chip is split into 800 px patches (80 m), which the model rescales to its 400 px input.
DeepForest was trained on forest imagery where crowns are smaller than Sydney's street and park
trees. With 400 px patches it split large crowns into several small boxes; 800 px gives one box per crown.
A detection is kept if its score is at least 0.15 (chosen in section 4) and its centre lies in the
chip's 100 m core, so trees in the overlap margin are not counted twice.
""")

code("""
SCORE_THRESHOLD = 0.15
trees = steps.detect_trees(chips, score_threshold=SCORE_THRESHOLD)
print(f"{len(trees):,} trees detected in {DEMO.to_crs(steps.METRIC_CRS).area.iloc[0] / 1e4:.0f} ha")
trees[["score", "crown_area_m2", "crown_diameter_m"]].describe().round(2).T
""")

code("""
raw = steps.predict_chip(sample.path)
kept = raw[raw.score >= SCORE_THRESHOLD]

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.imshow(plt.imread(sample.path))
for d in kept.itertuples():
    ax.add_patch(plt.Rectangle((d.xmin, d.ymin), d.xmax - d.xmin, d.ymax - d.ymin, fill=False,
                               edgecolor=plt.cm.autumn_r(d.score), linewidth=1.5))
margin = 100  # 10 m margin at 10 cm per pixel
ax.add_patch(plt.Rectangle((margin, margin), sample.px - 2 * margin, sample.px - 2 * margin, fill=False,
                           edgecolor="white", linestyle="--", linewidth=1))
ax.set_title(f"Chip r2c2: {len(kept)} boxes with score >= {SCORE_THRESHOLD} (dashed line = 100 m core)")
ax.set_axis_off()
""")

code("""
fig, ax = plt.subplots(figsize=(9, 9))
trees.to_crs(3857).boundary.plot(ax=ax, color="#ffeb3b", linewidth=0.6)
suburbs.to_crs(3857).boundary.plot(ax=ax, color="white", linewidth=2)
aerial(ax, zoom=18)
ax.set_title(f"All {len(trees):,} detected crowns")
plt.show()
""")

md("""
### 3b. Lidar check: removing roofs and shrubs

The detector only sees colour imagery, and some tile and iron roofs look like crowns to it (see the
roof boxes in the chip above). Lidar settles it. NSW Spatial Services publishes classified point clouds
through ELVIS (box 1, "Aerial and Lidar"), where every laser return is already labelled as ground,
vegetation or building.

**Production:** once per lidar survey, the tiles are rasterised into a 1 m vegetation height model (tallest
vegetation return minus ground) and a building mask, and stored on S3 as a COG. Every monthly run then
reads that COG.
**Runs locally:** yes, if the tiles are in `data/cache/lidar/`. The NSW bucket is Requester Pays, so the
tiles are ordered free from [ELVIS](https://elevation.fsdf.org.au) (emailed link) or fetched with AWS credentials.
Without them this step is skipped and every detection is kept.

A detection is kept if at least 25% of its box is vegetation over 2 m and the tallest vegetation reaches 3 m.
A tree overhanging a roof still passes, because the returns from its leaves are classified as vegetation.
The lidar is older than the imagery (May 2020 vs 2026). Trees planted since 2020 will be removed and trees
felled since can still pass, so in production the lidar is refreshed whenever NSW flies a new survey.
""")

code("""
lidar_tiles = steps.find_lidar_tiles(BBOX)
display(lidar_tiles[["survey", "file_name", "size_mb"]])
lidar = steps.lidar_rasters(BBOX, lidar_tiles)

trees_raw, checked = trees, None
if lidar is not None:
    checked = steps.lidar_check(trees_raw, lidar)
    trees = checked[checked["lidar_ok"]].drop(columns="lidar_ok").reset_index(drop=True)
    print(f"Lidar survey {lidar.survey}: kept {len(trees):,} of {len(trees_raw):,} detections "
          f"({(~checked['lidar_ok']).sum()} removed)")
    display(checked.groupby("lidar_ok")[["lidar_height_m", "canopy_pct", "building_pct"]].median()
            .rename(index={True: "kept", False: "removed"}).rename_axis("median per detection"))
else:
    print("Lidar check skipped: continuing with all detections")
""")

code("""
if lidar is not None:
    from rasterio.plot import plotting_extent

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    ax = axes[0]
    ext = plotting_extent(lidar.height, lidar.transform)
    ax.imshow(np.where(lidar.building, 1.0, np.nan), cmap="Greys", vmin=0, vmax=2, extent=ext)
    im = ax.imshow(np.where(lidar.height >= 2, lidar.height, np.nan), cmap="YlGn", vmin=0, vmax=25, extent=ext)
    DEMO.to_crs(lidar.crs).boundary.plot(ax=ax, color="black", linewidth=1.5)
    fig.colorbar(im, ax=ax, shrink=0.7, label="vegetation height above ground (m)")
    ax.set_title(f"Lidar {lidar.survey}: vegetation over 2 m (green), buildings (grey)")
    ax.set_axis_off()

    ax = axes[1]
    web = checked.to_crs(3857)
    web[web["lidar_ok"]].boundary.plot(ax=ax, color="#ffeb3b", linewidth=1.5)
    web[~web["lidar_ok"]].boundary.plot(ax=ax, color="#ff1744", linewidth=1.5)
    b = chips.set_index("chip_id").loc["r2c2"].geometry.bounds
    ax.set_xlim(b[0], b[2]); ax.set_ylim(b[1], b[3])
    aerial(ax, zoom=19)
    ax.set_title("Chip r2c2: kept (yellow) and removed by the lidar check (red)")
    plt.show()
""")

# --------------------------------------------------------------------------- 4

md("""
## 4. Validation and QA

**Diagram box:** Validation and QA.
**Production:** GeoPandas compares every run with the council register before anything is published.
**Runs locally:** yes.

Each register tree is matched to the nearest detected crown centre within 5 m (`sjoin_nearest`),
with each detection used once. The register records the trunk and the model the crown centre; with
crowns about 7 m across, those are often a few metres apart. **Recall** is the share of register
trees that were found. **Precision** is measured only near register trees, because the register covers
public land only: a correct detection of a backyard tree would otherwise count as a false positive.
In production, precision on private land would come from a sample of hand-counted plots.
""")

code("""
v = steps.validate(trees, register, match_m=5.0)
pd.Series(v.as_dict()).to_frame("value")
""")

code("""
def detected_at(threshold):
    d = steps.detect_trees(chips, score_threshold=threshold)
    return d[steps.lidar_check(d, lidar)["lidar_ok"]] if lidar is not None else d


sweep = []
for t in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4):
    detected = detected_at(t)
    m = steps.validate(detected, register).as_dict()
    sweep.append({"score threshold": t, "trees detected": len(detected), "near register": m["detections_in_zone"],
                  "precision": m["precision"], "recall": m["recall"], "f1": m["f1"]})
pd.DataFrame(sweep).set_index("score threshold")
""")

md("""
**Choosing the threshold.** On imagery alone, lowering the threshold quickly adds boxes that are roofs
and shrubs, which validation cannot see because they are away from register trees. The lidar check
removes those at every threshold (the sweep above applies it), so a lower threshold becomes worthwhile.
0.15 is used: it finds more register trees than 0.2 and keeps precision near register trees at about 0.9.
Going down to 0.1 adds recall but lets in more low-confidence boxes that the lidar check has to remove.

**Where recall is lost.** Recall depends strongly on tree size. Small, young street trees are often
hidden under a neighbour's canopy or are too small to see, and rows of touching crowns can merge
into one box:
""")

code("""
found = register["asset_id"].isin(v.matches["asset_id"])
height = pd.cut(register["TreeHeight"], [0, 5, 8, 12, 100], labels=["under 5 m", "5-8 m", "8-12 m", "12 m+"])
pd.DataFrame({"register trees": found.groupby(height).size(), "recall": found.groupby(height).mean().round(2)}).T
""")

md("""
This is the main limitation of an off-the-shelf model. The lidar check in section 3b removes false
detections but cannot add missed trees, and trees under 3 m are removed by design. In production, recall is
improved by fine-tuning the model on the council register. The published count is the detected count, and
the recall figures are published with every run so users can judge how complete it is.

**The QA gate.** The production thresholds (precision of at least 0.8, recall of at least 0.4) sit just below this
baseline run, so the gate catches drift such as bad imagery or a broken model release. The run passes.
With a deliberately strict threshold the same run is blocked and nothing would be published.
""")

code("""
passed = steps.qa_gate(v)
_ = steps.qa_gate(v, min_precision=0.9, min_recall=0.9)
""")

code("""
reg_m = register.to_crs(3857)
matched = reg_m["asset_id"].isin(v.matches["asset_id"])
centres = trees.to_crs(steps.METRIC_CRS).centroid.to_crs(3857)

fig, ax = plt.subplots(figsize=(9, 9))
gpd.GeoSeries(v.zone.to_crs(3857)).plot(ax=ax, color="white", alpha=0.15)
trees.to_crs(3857).boundary.plot(ax=ax, color="#ffeb3b", linewidth=0.5)
centres.plot(ax=ax, color="#ffeb3b", markersize=3)
reg_m[matched].plot(ax=ax, color="#00e676", markersize=14, label=f"register tree found ({matched.sum()})")
reg_m[~matched].plot(ax=ax, color="#ff1744", markersize=14, label=f"register tree missed ({(~matched).sum()})")
ax.legend(loc="lower left")
xmin, ymin, xmax, ymax = reg_m.total_bounds
cx_, cy_ = (xmin + xmax) / 2, (ymin + ymax) / 2
ax.set_xlim(cx_ - 180, cx_ + 180)
ax.set_ylim(cy_ - 180, cy_ + 180)
aerial(ax, zoom=19)
ax.set_title("Validation, zoomed in: register trees vs detected crowns (white = evaluation zone)")
plt.show()
""")

# --------------------------------------------------------------------------- 5

md("""
## 5. Storage

**Diagram box:** Spatial database and Object storage.
**Production:** PostGIS holds the trees, suburbs and run history; S3 holds a GeoParquet snapshot per month.
**Runs locally:** DuckDB with the spatial extension stands in for PostGIS, and `data/s3/` for S3.
The SQL uses only functions both have (`ST_Within`, `ST_Centroid`, `ST_MakeEnvelope`,
`ST_Intersects`), so it runs unchanged on PostGIS. There you would also add
`CREATE INDEX ON trees USING GIST (geom)`.

Blocked runs are still recorded in `runs` (for monitoring), but their trees are not stored.
""")

code("""
result = steps.store_run(RUN_ID, trees, suburbs, v, passed)
print("database:", result["db"].relative_to(ROOT))
print("snapshot:", result["snapshot"].relative_to(ROOT))
print(steps.SUBURB_STATS_SQL)
""")

code("""
con = steps.connect(read_only=True)
display(con.sql("SELECT 'runs' AS tbl, COUNT(*) AS n FROM runs UNION ALL SELECT 'suburbs', COUNT(*) FROM suburbs "
                "UNION ALL SELECT 'trees', COUNT(*) FROM trees").df())
display(con.sql("SELECT * FROM runs").df())
stats = con.sql("SELECT * FROM suburb_stats").df()
con.close()
stats
""")

code("""
snap = gpd.read_parquet(result["snapshot"])
import pyarrow.parquet as pq
geo = json.loads(pq.read_schema(result["snapshot"]).metadata[b"geo"])
print(f"GeoParquet snapshot: {len(snap):,} rows, geometry column '{geo['primary_column']}', "
      f"encoding {geo['columns'][geo['primary_column']]['encoding']}")
snap.head(3)
""")

# --------------------------------------------------------------------------- 6

md("""
## 6. API and tiles

**Diagram box:** API and tile service.
**Production:** FastAPI (REST/JSON and GeoJSON) on containers, plus Martin or pg_tileserv for
vector tiles, all behind the CloudFront CDN. The API sets `Cache-Control` headers, so the CDN
answers almost every request and the database is only queried once per URL per day.
**Runs locally:** the same FastAPI app (`api/main.py`), called in-process with FastAPI's `TestClient` instead of over the network.
""")

code("""
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)
print("routes:", ", ".join(r.path for r in app.routes if not r.path.startswith(("/docs", "/redoc", "/openapi"))))

r = client.get("/suburbs")
print("\\nGET /suburbs ->", r.status_code, "| Cache-Control:", r.headers["cache-control"])
print(json.dumps(r.json(), indent=2))
""")

md("""
Vector tiles are what the web map draws. The tile below is encoded by the API, then decoded again
to show what a browser receives: a compact binary tile with a `trees` layer and a `suburbs` layer.
""")

code("""
import mapbox_vector_tile
import mercantile

lon, lat = (BBOX[0] + BBOX[2]) / 2, (BBOX[1] + BBOX[3]) / 2
t = mercantile.tile(lon, lat, 17)
r = client.get(f"/tiles/{t.z}/{t.x}/{t.y}.mvt")
print(f"GET /tiles/{t.z}/{t.x}/{t.y}.mvt ->", r.status_code, "|", r.headers["content-type"], "|",
      f"{len(r.content):,} bytes | Cache-Control: {r.headers['cache-control']}")

decoded = mapbox_vector_tile.decode(r.content)
for name, layer in decoded.items():
    print(f"  layer '{name}': {len(layer['features'])} features, extent {layer['extent']}, "
          f"first properties {layer['features'][0]['properties']}")

fig, ax = plt.subplots(figsize=(6, 6))
for f in decoded["trees"]["features"]:
    for ring in f["geometry"]["coordinates"]:
        xs, ys = zip(*ring)
        ax.plot(xs, ys, color="#2e7d32", linewidth=0.8)
for f in decoded["suburbs"]["features"]:
    polys = f["geometry"]["coordinates"] if f["geometry"]["type"] == "MultiPolygon" else [f["geometry"]["coordinates"]]
    for poly in polys:
        xs, ys = zip(*poly[0])
        ax.plot(xs, ys, color="black", linewidth=1.5)
ax.set_xlim(0, 4096); ax.set_ylim(0, 4096); ax.set_aspect("equal")
ax.set_title(f"Decoded tile {t.z}/{t.x}/{t.y} (tile coordinates)")
plt.show()
""")

# --------------------------------------------------------------------------- 7

md("""
## 7. Web map

**Diagram box:** Web map.
**Production:** a React app using MapLibre GL, reading vector tiles from the CDN.
**Runs locally:** `outputs/map.html`, a standalone MapLibre GL page with the same layers (a suburb
choropleth and tree crowns over NSW aerial imagery). It embeds the data, so it opens straight from disk.
Open it in a browser to pan, zoom and click a suburb. A static PNG of the same view is shown below so it is visible on GitHub.
""")

code("""
from pipeline.webmap import write_map_png, write_web_map

suburb_stats = steps.suburb_stats_gdf()
html = write_web_map(RUN_ID, trees, suburb_stats, ROOT / "outputs" / "map.html")
png = write_map_png(RUN_ID, trees, suburb_stats, ROOT / "outputs" / "map.png")
print("web map:", html.relative_to(ROOT), f"({html.stat().st_size / 1e3:.0f} KB)")
display(Image(filename=str(png), width=760))
""")

# --------------------------------------------------------------------------- 8

md("""
## 8. Open data

**Diagram box:** Open data.
**Production:** after every published run the files are written to a public S3 bucket behind the
CDN. Each run has a permanent URL, and `latest.json` always points at the newest run.
**Runs locally:** files are written to `outputs/open-data/`.
""")

code("""
export = steps.export_open_data(RUN_ID, trees, suburb_stats)
files = sorted(export["folder"].iterdir())
display(pd.DataFrame({"file": [f.name for f in files], "KB": [round(f.stat().st_size / 1e3, 1) for f in files]}))
print(json.dumps(export["latest"], indent=2))
""")

code("""
back = gpd.read_file(export["folder"] / "trees.geojson")
csv = pd.read_csv(export["folder"] / "trees.csv")
print(f"trees.geojson reads back: {len(back):,} features, CRS {back.crs}")
print(f"trees.csv reads back: {len(csv):,} rows")
csv.head()
""")

# --------------------------------------------------------------------------- 9

md("""
## 9. Public API

**Diagram box:** Public API.
**Production:** the same FastAPI service, documented with OpenAPI, over HTTPS. Other apps can ask for
"trees in this bounding box" as GeoJSON. GIS software such as QGIS and ArcGIS can connect through the
OGC API Features endpoints (the modern successor to WFS): in QGIS, *Layer > Add Layer > Add WFS / OGC API
Features Layer*, then enter the API's base URL.
**Runs locally:** the same calls through `TestClient`. The `curl` lines show the production equivalents.
""")

code("""
API = "https://api.example.org/sydney-trees"
small = (151.2025, -33.8915, 151.2035, -33.8905)
bbox = ",".join(map(str, small))

print(f"$ curl '{API}/trees?bbox={bbox}'")
r = client.get("/trees", params={"bbox": bbox})
fc = r.json()
print(r.status_code, r.headers["content-type"], "|", fc["numberMatched"], "trees in the box")
first = fc["features"][0]
print(json.dumps({"type": first["type"], "id": first["id"],
                  "geometry": {"type": first["geometry"]["type"],
                               "coordinates": [[[round(x, 6), round(y, 6)] for x, y in first["geometry"]["coordinates"][0]]]},
                  "properties": first["properties"]}, indent=1))
""")

code("""
print(f"$ curl '{API}/collections/trees/items?limit=2'")
r = client.get("/collections/trees/items", params={"limit": 2})
body = r.json()
print(r.status_code, r.headers["content-type"])
print(json.dumps({k: body[k] for k in ("type", "numberMatched", "numberReturned", "links")}, indent=2))

print(f"\\n$ curl '{API}/conformance'")
print(json.dumps(client.get("/conformance").json(), indent=2))
""")

# --------------------------------------------------------------------------- wrap-up

md("""
## Wrap-up: from this notebook to production

| Diagram box | In this notebook | In production |
|---|---|---|
| Ingestion and orchestration | Step functions called in order; `pipeline/dag.py` printed | Airflow (MWAA or containerised) runs `sydney_tree_count` monthly |
| Tree detection model | DeepForest on CPU, 36 chips | Same container on AWS Batch GPU (spot), chips listed from S3 |
| Lidar check | ELVIS tiles in `data/cache/lidar/`, rasterised to a COG in `data/s3/raw/lidar/` | Tiles fetched from the Requester Pays bucket once per survey; the height model COG on S3 is reused every month |
| Validation and QA | `validate()` + `qa_gate()` | Same functions; `ShortCircuitOperator` blocks `publish` |
| Spatial database | DuckDB spatial, `data/store.duckdb` | PostGIS on RDS, GIST index on `geom` |
| Object storage | `data/s3/` | S3 (`raw/`, `detections/`, `snapshots/`), versioned |
| API and tiles | `api/main.py` via `TestClient`; tiles encoded in the API | FastAPI on ECS Fargate; Martin for tiles; CloudFront CDN in front |
| Web map | `outputs/map.html` (MapLibre GL, embedded GeoJSON) | React + MapLibre GL reading vector tiles |
| Open data | `outputs/open-data/` | Public S3 bucket with permanent per-run URLs |
| Public API | GeoJSON + OGC API Features through `TestClient` | Same routes over HTTPS with OpenAPI docs |

**Scaling.** The demo covers 36 ha. The City of Sydney LGA is about 26 km², or roughly 2,600 chips of
100 m; at around 0.3 s per chip on a single GPU (an assumption, versus about 13 s per chip on this CPU)
that is under 15 minutes. Greater Sydney's urban area (roughly 1,700 km², about 170,000 chips) is about
15 GPU-hours, which AWS Batch splits across many spot instances.

**Infrastructure assumptions**

- AWS in `ap-southeast-2` (Sydney), so data stays in Australia and latency for local users is low.
- The GPU is needed only during the monthly run, so it is a short-lived AWS Batch job rather than an always-on server.
- Sentinel-2 is free and open. Higher-resolution NSW aerial imagery is used where its licence allows.
- Everything is containerised with Docker, defined in Terraform and deployed with GitHub Actions.
- The public API and map are read-only and cached by the CDN, so traffic spikes do not reach the database.
- CloudWatch and Grafana alert if a run fails, imagery is late, or the model's accuracy drops below the QA thresholds.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
out = Path(__file__).resolve().parents[1] / "notebooks" / "part2_walkthrough.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print("wrote", out)
