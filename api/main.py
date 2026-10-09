"""Public API and tile service over the tree store.

Production: this app runs in containers behind CloudFront, reading PostGIS, and
Martin/pg_tileserv serves the vector tiles. Locally it reads `data/store.duckdb`.

    uv run uvicorn api.main:app --reload
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import duckdb
import mapbox_vector_tile
import mercantile
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from pyproj import Transformer
from shapely import wkb
from shapely.geometry import box
from shapely.ops import transform

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.steps import DB_PATH, connect  # noqa: E402

DB = Path(os.environ.get("LYNXX_DB", DB_PATH))
GEOJSON = "application/geo+json"
MVT = "application/vnd.mapbox-vector-tile"

# Browser cache / CDN cache. Data only changes once a month, when a run is published.
CACHE_DATA = "public, max-age=300, s-maxage=86400, stale-while-revalidate=3600"
CACHE_TILES = "public, max-age=3600, s-maxage=604800"

app = FastAPI(title="Sydney Trees API", version="0.1.0",
              description="Tree counts by suburb, individual tree crowns, vector tiles and OGC API Features.")

TO_WEB = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform


def db():
    con = connect(read_only=True, path=DB)
    try:
        yield con
    finally:
        con.close()


@app.middleware("http")
async def cache_headers(request: Request, call_next):
    response = await call_next(request)
    if request.method == "GET" and response.status_code == 200 and "cache-control" not in response.headers:
        response.headers["Cache-Control"] = CACHE_DATA
    return response


def latest_run(con) -> str:
    row = con.execute("SELECT run_id FROM runs WHERE passed ORDER BY run_at DESC LIMIT 1").fetchone()
    if not row:
        raise HTTPException(503, "No published run yet")
    return row[0]


def parse_bbox(bbox: str | None) -> tuple[float, float, float, float] | None:
    if not bbox:
        return None
    try:
        minx, miny, maxx, maxy = (float(v) for v in bbox.split(","))
    except ValueError:
        raise HTTPException(400, "bbox must be minLon,minLat,maxLon,maxLat")
    return minx, miny, maxx, maxy


def records(cur) -> list[dict]:
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def tree_features(con, run_id: str, bbox, limit: int, offset: int) -> tuple[list[dict], int]:
    where, args = "run_id = ?", [run_id]
    if bbox:
        where += " AND ST_Intersects(geom, ST_MakeEnvelope(?, ?, ?, ?))"
        args += list(bbox)
    total = con.execute(f"SELECT COUNT(*) FROM trees WHERE {where}", args).fetchone()[0]
    rows = con.execute(f"""
        SELECT tree_id, score, crown_area_m2, crown_diameter_m, ST_AsGeoJSON(geom) AS geometry
        FROM trees WHERE {where} ORDER BY tree_id LIMIT ? OFFSET ?""", args + [limit, offset]).fetchall()
    feats = [{"type": "Feature", "id": r[0], "geometry": json.loads(r[4]),
              "properties": {"tree_id": r[0], "run_id": run_id, "score": r[1],
                             "crown_area_m2": r[2], "crown_diameter_m": r[3]}} for r in rows]
    return feats, total


# --------------------------------------------------------------------------- REST / JSON

@app.get("/runs")
def runs(con=Depends(db)):
    """Run history, including runs blocked by the QA gate."""
    return records(con.execute("SELECT * FROM runs ORDER BY run_at DESC"))


@app.get("/suburbs")
def suburbs(con=Depends(db)):
    """Tree count, density and crown cover per suburb for the latest published run."""
    return records(con.execute("SELECT * FROM suburb_stats"))


@app.get("/suburbs/{name}")
def suburb(name: str, con=Depends(db)):
    rows = records(con.execute("SELECT * FROM suburb_stats WHERE lower(suburb) = lower(?)", [name]))
    if not rows:
        raise HTTPException(404, f"Unknown suburb: {name}")
    return rows[0]


@app.get("/trees")
def trees(bbox: str = Query(..., description="minLon,minLat,maxLon,maxLat"),
          limit: int = Query(1000, le=10000), con=Depends(db)):
    """Tree crowns intersecting a bounding box, as a GeoJSON FeatureCollection."""
    feats, total = tree_features(con, latest_run(con), parse_bbox(bbox), limit, 0)
    return JSONResponse({"type": "FeatureCollection", "numberMatched": total, "features": feats},
                        media_type=GEOJSON)


# --------------------------------------------------------------------------- OGC API Features

COLLECTIONS = {
    "trees": "Detected tree crowns (latest published run)",
    "suburbs": "Suburbs with tree statistics (latest published run)",
}


def link(request: Request, path: str, rel: str, type_: str = "application/json", **params) -> dict:
    url = str(request.base_url).rstrip("/") + path
    if params:
        url += "?" + "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
    return {"href": url, "rel": rel, "type": type_}


@app.get("/")
def landing(request: Request):
    return {
        "title": "Sydney Trees",
        "description": "OGC API Features landing page",
        "links": [link(request, "/", "self"), link(request, "/conformance", "conformance"),
                  link(request, "/collections", "data"), link(request, "/openapi.json", "service-desc",
                                                              "application/vnd.oai.openapi+json;version=3.0")],
    }


@app.get("/conformance")
def conformance():
    return {"conformsTo": [
        "http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/core",
        "http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/geojson",
    ]}


def collection_doc(request: Request, cid: str) -> dict:
    return {"id": cid, "title": COLLECTIONS[cid], "itemType": "feature",
            "crs": ["http://www.opengis.net/def/crs/OGC/1.3/CRS84"],
            "links": [link(request, f"/collections/{cid}", "self"),
                      link(request, f"/collections/{cid}/items", "items", GEOJSON)]}


@app.get("/collections")
def collections(request: Request):
    return {"collections": [collection_doc(request, c) for c in COLLECTIONS],
            "links": [link(request, "/collections", "self")]}


@app.get("/collections/{cid}")
def collection(cid: str, request: Request):
    if cid not in COLLECTIONS:
        raise HTTPException(404, f"Unknown collection: {cid}")
    return collection_doc(request, cid)


@app.get("/collections/{cid}/items")
def items(cid: str, request: Request, bbox: str | None = None, limit: int = Query(100, ge=1, le=10000),
          offset: int = Query(0, ge=0), con=Depends(db)):
    if cid not in COLLECTIONS:
        raise HTTPException(404, f"Unknown collection: {cid}")
    box_ = parse_bbox(bbox)
    if cid == "trees":
        feats, total = tree_features(con, latest_run(con), box_, limit, offset)
    else:
        where, args = ("WHERE ST_Intersects(s.geom, ST_MakeEnvelope(?, ?, ?, ?))", list(box_)) if box_ else ("", [])
        rows = records(con.execute(f"""
            SELECT st.*, ST_AsGeoJSON(s.geom) AS geometry FROM suburb_stats st JOIN suburbs s USING (suburb)
            {where} ORDER BY st.suburb LIMIT ? OFFSET ?""", args + [limit, offset]))
        feats = [{"type": "Feature", "id": r["suburb"], "geometry": json.loads(r.pop("geometry")), "properties": r}
                 for r in rows]
        total = con.execute(f"SELECT COUNT(*) FROM suburbs s {where}", args).fetchone()[0]

    links = [link(request, f"/collections/{cid}/items", "self", GEOJSON, bbox=bbox, limit=limit, offset=offset)]
    if offset + len(feats) < total:
        links.append(link(request, f"/collections/{cid}/items", "next", GEOJSON, bbox=bbox, limit=limit,
                          offset=offset + limit))
    return JSONResponse({"type": "FeatureCollection", "numberMatched": total, "numberReturned": len(feats),
                         "features": feats, "links": links}, media_type=GEOJSON)


# --------------------------------------------------------------------------- Vector tiles

@app.get("/tiles/{z}/{x}/{y}.mvt")
def tile(z: int, x: int, y: int, con=Depends(db)):
    """Mapbox Vector Tile with `trees` and `suburbs` layers (what Martin serves in production)."""
    lonlat = mercantile.bounds(x, y, z)
    web = mercantile.xy_bounds(x, y, z)
    pad = (web.right - web.left) * 0.02
    clip = box(web.left - pad, web.bottom - pad, web.right + pad, web.top + pad)
    env = [lonlat.west, lonlat.south, lonlat.east, lonlat.north]

    def features(sql: str, args: list) -> list[dict]:
        out = []
        for row in records(con.execute(sql, args)):
            geom = transform(TO_WEB, wkb.loads(bytes(row.pop("wkb")))).intersection(clip)
            if not geom.is_empty:
                out.append({"geometry": geom, "properties": row})
        return out

    layers = []
    if z >= 14:
        layers.append({"name": "trees", "features": features(
            """SELECT tree_id, score, crown_area_m2, ST_AsWKB(geom) AS wkb FROM trees
               WHERE run_id = ? AND ST_Intersects(geom, ST_MakeEnvelope(?, ?, ?, ?))""",
            [latest_run(con)] + env)})
    layers.append({"name": "suburbs", "features": features(
        """SELECT st.suburb, st.trees, st.trees_per_ha, st.crown_cover_pct, ST_AsWKB(s.geom) AS wkb
           FROM suburb_stats st JOIN suburbs s USING (suburb)
           WHERE ST_Intersects(s.geom, ST_MakeEnvelope(?, ?, ?, ?))""", env)})

    data = mapbox_vector_tile.encode(layers, default_options={"quantize_bounds": tuple(web), "extents": 4096})
    return Response(data, media_type=MVT, headers={"Cache-Control": CACHE_TILES})
