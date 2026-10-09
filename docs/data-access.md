# Data access

All sources are public and need no API key. Every call is HTTPS.

| Source | Endpoint | Protocol | Notes |
|---|---|---|---|
| Sentinel-2 L2A | `https://earth-search.aws.element84.com/v1`, collection `sentinel-2-l2a` | STAC API (JSON), then HTTP range requests on COGs in `s3://sentinel-cogs` (us-west-2) | 10 m bands `red`, `nir`; 20 m `scl` for cloud masking. Reflectance = DN x `scale` + `offset` from the asset's `raster:bands`. Tile 56HLH covers Sydney. |
| NSW aerial imagery | `https://maps.six.nsw.gov.au/arcgis/rest/services/public/NSW_Imagery/MapServer/export` | ArcGIS REST MapServer, `f=image` | Max 4096 px per request. `f=image` sometimes returns an HTML error, so the client checks `Content-Type` and retries. The `f=json` output link often returns 404. Cached tiles: `.../MapServer/tile/{z}/{y}/{x}` (level 21 is about 7.5 cm). |
| City of Sydney tree register | `https://services1.arcgis.com/cNVyNtjGVZybOQWZ/arcgis/rest/services/Trees/FeatureServer/0/query` | ArcGIS REST FeatureServer, `f=geojson` | `maxRecordCount` 2000, so the client pages with `resultOffset`. Street and park trees only (public land). Portal item `15c4713a688a48fcb604fc343118af05`. |
| NSW suburbs | `https://portal.spatial.nsw.gov.au/server/rest/services/NSW_Administrative_Boundaries_Theme/FeatureServer/2/query` | ArcGIS REST FeatureServer, `f=geojson` | Field `suburbname`. `maxRecordCount` 2000. |
| NSW lidar point clouds | Catalogue: `https://api.elevation.fsdf.org.au/elevation/downloadables?polygon=<WKT, lon lat>` (GET; needs `Origin`/`Referer: https://elevation.fsdf.org.au`, and spaces encoded as `%20`, not `+`). Files: `s3://nsw-elvis/elevation/lidar/z56/<survey>/...laz` | ELVIS API (JSON), then LAZ download | Classified (ground 2, vegetation 3-5, building 6), about 15 points/m². `Sydney202005` is GDA2020 / MGA 56 (EPSG:7856) with AHD heights; the code reads the CRS from each file's header. ELVIS downloads unpack into `NSW Government - Spatial Services/Point Clouds/AHD/`, and the code searches subfolders. The demo area needs two 2 km tiles from survey `Sydney202005` (about 221 MB each). **The bucket is Requester Pays**, so anonymous downloads get 403. Either order free through https://elevation.fsdf.org.au (emailed link) and save the files to `data/cache/lidar/`, or download with AWS credentials (`RequestPayer=requester`). |
| Greater Sydney tree canopy 2024/25 (production only) | `https://datasets.seed.nsw.gov.au/dataset/greater-sydney-region-tree-canopy-access-202425` | Download | Canopy polygons for cross-checking crown cover. Not used in the demo. |

Licences: Sentinel-2 data is free and open (Copernicus). NSW imagery and boundaries are
(c) NSW Spatial Services (CC BY 4.0). The tree register is (c) City of Sydney (CC BY 4.0).
