from dagster import Definitions, define_asset_job, ScheduleDefinition

from gis_pipeline.assets import all_assets
from gis_pipeline.resources import PipelineConfig

# ── Jobs ──────────────────────────────────────────────────────────────────────

full_pipeline_job = define_asset_job(
    name="full_pipeline",
    selection=["gee_export", "train_unet", "evaluate_model"],
    description="Export GEE data → Train U-Net → Evaluate",
)

# Re-train only 
retrain_job = define_asset_job(
    name="retrain_and_evaluate",
    selection=["train_unet", "evaluate_model"],
    description="Train U-Net on existing TIF files → Evaluate",
)

# Evaluate only (if model already trained)
evaluate_job = define_asset_job(
    name="evaluate_only",
    selection=["evaluate_model"],
    description="Run evaluation on the saved model",
)

# ── Schedules ──────────────────────────────────────────────────────
# Monday at 02:00
# weekly_retrain = ScheduleDefinition(
#     job=retrain_job,
#     cron_schedule="0 2 * * 1",
#     execution_timezone="America/La_Paz",
# )

# ── Definitions ───────────────────────────────────────────────────────────────
defs = Definitions(
    assets=all_assets,
    resources={
        "config": PipelineConfig(),
    },
    jobs=[full_pipeline_job, retrain_job, evaluate_job],
)