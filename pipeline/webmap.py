"""Web map outputs: a standalone MapLibre GL page and a static PNG of the same view.

The production React app uses the same MapLibre style, but reads vector tiles from
`/tiles/{z}/{x}/{y}.mvt` instead of embedded GeoJSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd

from pipeline.steps import METRIC_CRS, SIX_TILE_URL, to_geojson

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Sydney Trees - __RUN_ID__</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet">
<script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
<style>
  html, body, #map { margin: 0; height: 100%; font-family: system-ui, sans-serif; }
  .panel { position: absolute; top: 12px; left: 12px; background: #fff; padding: 12px 14px; border-radius: 8px;
           box-shadow: 0 1px 4px rgba(0,0,0,.25); max-width: 260px; font-size: 13px; }
  .panel h1 { font-size: 16px; margin: 0 0 6px; }
  .swatch { display: inline-block; width: 14px; height: 10px; margin-right: 4px; vertical-align: middle; }
  label { display: block; margin-top: 6px; cursor: pointer; }
</style>
</head>
<body>
<div id="map"></div>
<div class="panel">
  <h1>Sydney Trees</h1>
  <div>Run <b>__RUN_ID__</b> &middot; <b>__N_TREES__</b> trees detected</div>
  <div style="margin-top:8px">Suburb trees per ha</div>
  <div id="legend"></div>
  <label><input type="checkbox" id="imagery" checked> NSW aerial imagery</label>
  <label><input type="checkbox" id="crowns" checked> Tree crowns</label>
</div>
<script>
const suburbs = __SUBURBS__;
const trees = __TREES__;
const breaks = __BREAKS__;
const colours = ["#edf8e9", "#bae4b3", "#74c476", "#31a354", "#006d2c"];

const map = new maplibregl.Map({
  container: "map",
  bounds: __BOUNDS__,
  fitBoundsOptions: { padding: 40 },
  style: {
    version: 8,
    glyphs: "https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf",
    sources: {
      carto: { type: "raster", tileSize: 256, attribution: "&copy; OpenStreetMap &copy; CARTO",
               tiles: ["https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png"] },
      nsw: { type: "raster", tileSize: 256, maxzoom: 21, attribution: "&copy; NSW Spatial Services",
             tiles: ["__SIX_TILES__"] }
    },
    layers: [
      { id: "carto", type: "raster", source: "carto" },
      { id: "nsw", type: "raster", source: "nsw", paint: { "raster-opacity": 0.85 } }
    ]
  }
});
map.addControl(new maplibregl.NavigationControl());

const fill = ["step", ["get", "trees_per_ha"], colours[0]];
breaks.slice(1).forEach((b, i) => fill.push(b, colours[i + 1]));

map.on("load", () => {
  map.addSource("suburbs", { type: "geojson", data: suburbs });
  map.addSource("trees", { type: "geojson", data: trees });
  map.addLayer({ id: "suburb-fill", type: "fill", source: "suburbs", paint: { "fill-color": fill, "fill-opacity": 0.3 } });
  map.addLayer({ id: "suburb-line", type: "line", source: "suburbs", paint: { "line-color": "#333", "line-width": 1.5 } });
  map.addLayer({ id: "crowns", type: "line", source: "trees", paint: { "line-color": "#ffeb3b", "line-width": 1.2 } });
  map.addLayer({ id: "suburb-label", type: "symbol", source: "suburbs",
                 layout: { "text-field": ["concat", ["get", "suburb"], "\\n", ["to-string", ["get", "trees"]], " trees"],
                           "text-size": 13, "text-font": ["Open Sans Semibold"] },
                 paint: { "text-color": "#111", "text-halo-color": "#fff", "text-halo-width": 2 } });

  map.on("click", "suburb-fill", e => {
    const p = e.features[0].properties;
    new maplibregl.Popup().setLngLat(e.lngLat).setHTML(
      `<b>${p.suburb}</b><br>${p.trees} trees in ${p.area_ha} ha<br>${p.trees_per_ha} trees/ha<br>` +
      `${p.crown_cover_pct}% crown cover`).addTo(map);
  });
  map.on("mouseenter", "suburb-fill", () => map.getCanvas().style.cursor = "pointer");
  map.on("mouseleave", "suburb-fill", () => map.getCanvas().style.cursor = "");
});

document.getElementById("imagery").onchange = e => map.setLayoutProperty("nsw", "visibility", e.target.checked ? "visible" : "none");
document.getElementById("crowns").onchange = e => map.setLayoutProperty("crowns", "visibility", e.target.checked ? "visible" : "none");
document.getElementById("legend").innerHTML = breaks.map((b, i) =>
  `<div><span class="swatch" style="background:${colours[i]}"></span>${b}${breaks[i + 1] ? "&ndash;" + breaks[i + 1] : "+"}</div>`).join("");
</script>
</body>
</html>
"""


def _breaks(values, n: int = 5) -> list[float]:
    lo, hi = float(min(values)), float(max(values))
    step = (hi - lo) / n or 1
    return [round(lo + i * step, 1) for i in range(n)]


def write_web_map(run_id: str, trees: gpd.GeoDataFrame, suburb_stats: gpd.GeoDataFrame, out: Path) -> Path:
    bounds = suburb_stats.total_bounds
    html = (TEMPLATE
            .replace("__RUN_ID__", run_id)
            .replace("__N_TREES__", f"{len(trees):,}")
            .replace("__SIX_TILES__", SIX_TILE_URL)
            .replace("__BOUNDS__", json.dumps([[bounds[0], bounds[1]], [bounds[2], bounds[3]]]))
            .replace("__BREAKS__", json.dumps(_breaks(suburb_stats["trees_per_ha"])))
            .replace("__SUBURBS__", json.dumps(to_geojson(suburb_stats)))
            .replace("__TREES__", json.dumps(to_geojson(trees[["tree_id", "score", "crown_area_m2", "geometry"]]))))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def write_map_png(run_id: str, trees: gpd.GeoDataFrame, suburb_stats: gpd.GeoDataFrame, out: Path) -> Path:
    import contextily as cx
    import matplotlib.pyplot as plt

    web_sub = suburb_stats.to_crs(3857)
    web_trees = trees.to_crs(3857)
    fig, ax = plt.subplots(figsize=(9, 9))
    web_sub.plot(ax=ax, column="trees_per_ha", cmap="Greens", alpha=0.3, edgecolor="#222", linewidth=1.5,
                 legend=True, legend_kwds={"label": "trees per ha", "shrink": 0.6})
    web_trees.boundary.plot(ax=ax, color="#ffeb3b", linewidth=0.7)
    for row in web_sub.itertuples():
        p = row.geometry.representative_point()
        ax.annotate(f"{row.suburb}\n{row.trees} trees", (p.x, p.y), ha="center", fontsize=11, weight="bold",
                    bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    cx.add_basemap(ax, source=SIX_TILE_URL, zoom=18, attribution="NSW Spatial Services")
    area_ha = suburb_stats.to_crs(METRIC_CRS).area.sum() / 10_000
    ax.set_title(f"Sydney Trees, run {run_id}: {len(trees):,} trees in {area_ha:.0f} ha of Redfern")
    ax.set_axis_off()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90, bbox_inches="tight")
    plt.close(fig)
    return out
