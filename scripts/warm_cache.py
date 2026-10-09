"""Download every input and run detection once so the notebook re-runs from cache.

    uv run python scripts/warm_cache.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import steps  # noqa: E402

MONTH = "2026-09"

bbox = steps.demo_area()
print("demo bbox", bbox)

t = time.time()
items = steps.search_sentinel(steps.demo_area(1000), MONTH)
print(f"{len(items)} Sentinel-2 scenes ({time.time() - t:.0f}s)")
ndvi = steps.make_ndvi_cog(items, steps.demo_area(1000), MONTH)
print("NDVI composite", ndvi.path, ndvi.composite.shape, f"({time.time() - t:.0f}s)")

print(len(steps.load_register(bbox)), "register trees")
print(steps.load_suburbs(bbox))

t = time.time()
chips = steps.fetch_aerial_chips(bbox)
print(f"{len(chips)} aerial chips ({time.time() - t:.0f}s)")

for n, chip in enumerate(chips.itertuples(), 1):
    t = time.time()
    df = steps.predict_chip(chip.path)
    print(f"[{n}/{len(chips)}] {chip.chip_id}: {len(df)} boxes ({time.time() - t:.0f}s)", flush=True)

trees = steps.detect_trees(chips)
print(len(trees), "trees after threshold and de-duplication")
