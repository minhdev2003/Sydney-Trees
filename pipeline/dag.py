"""Airflow DAG: monthly Sydney tree count.

Wraps the functions in `pipeline/steps.py` as tasks. Tasks pass S3 keys to each
other through XCom, never data. In production `detect` is dispatched to an AWS Batch
GPU queue and `publish` writes to PostGIS and S3 instead of the local stand-ins.

Shown in the notebook, not run locally (Airflow does not run natively on Windows).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import geopandas as gpd
from airflow.decorators import dag, task
from airflow.operators.python import ShortCircuitOperator

from pipeline import steps

BBOX = steps.demo_area()
CONTEXT_BBOX = steps.demo_area(1000)


def month_of(data_interval_start) -> str:
    return data_interval_start.strftime("%Y-%m")


@dag(
    dag_id="sydney_tree_count",
    schedule="0 3 2 * *",  # 03:00 on the 2nd: last month's scenes are in the catalogue by then
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=10)},
    render_template_as_native_obj=True,
    tags=["trees", "monthly"],
)
def sydney_tree_count():

    @task
    def search_sentinel(data_interval_start=None) -> int:
        return len(steps.search_sentinel(CONTEXT_BBOX, month_of(data_interval_start)))

    @task
    def make_ndvi_cog(n_scenes: int, data_interval_start=None) -> str:
        month = month_of(data_interval_start)
        items = steps.search_sentinel(CONTEXT_BBOX, month)
        return str(steps.make_ndvi_cog(items, CONTEXT_BBOX, month).path)

    @task
    def fetch_aerial() -> str:
        chips = steps.fetch_aerial_chips(BBOX)
        key = steps.S3 / "raw" / "aerial" / "chips.parquet"
        key.parent.mkdir(parents=True, exist_ok=True)
        chips.assign(export_bounds=chips["export_bounds"].map(list)).to_parquet(key)
        return str(key)

    @task(queue="gpu", execution_timeout=timedelta(hours=6))
    def detect(chips_key: str, data_interval_start=None) -> str:
        chips = gpd.read_parquet(chips_key)
        chips["export_bounds"] = chips["export_bounds"].map(tuple)
        key = steps.S3 / "detections" / month_of(data_interval_start) / "trees.parquet"
        key.parent.mkdir(parents=True, exist_ok=True)
        steps.detect_trees(chips).to_parquet(key)
        return str(key)

    @task
    def load_register() -> str:
        key = steps.S3 / "ground-truth" / "register.parquet"
        key.parent.mkdir(parents=True, exist_ok=True)
        steps.load_register(BBOX).to_parquet(key)
        return str(key)

    @task
    def validate(trees_key: str, register_key: str) -> dict:
        return steps.validate(gpd.read_parquet(trees_key), gpd.read_parquet(register_key)).as_dict()

    qa_gate = ShortCircuitOperator(
        task_id="qa_gate",
        python_callable=steps.qa_gate,
        op_kwargs={"metrics": "{{ ti.xcom_pull(task_ids='validate') }}", "min_precision": 0.8, "min_recall": 0.4},
    )

    @task
    def publish(trees_key: str, metrics: dict, data_interval_start=None) -> None:
        run_id = month_of(data_interval_start)
        trees = gpd.read_parquet(trees_key)
        steps.store_run(run_id, trees, steps.load_suburbs(BBOX),
                        steps.Validation(**metrics, matches=None, zone=None), passed=True)
        steps.export_open_data(run_id, trees, steps.suburb_stats_gdf())

    make_ndvi_cog(search_sentinel())

    trees_key = detect(fetch_aerial())
    metrics = validate(trees_key, load_register())
    metrics >> qa_gate >> publish(trees_key, metrics)


sydney_tree_count()
