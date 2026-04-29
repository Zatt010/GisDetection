from gis_pipeline.assets.gee_export import gee_export
from gis_pipeline.assets.training import train_unet
from gis_pipeline.assets.evaluation import evaluate_model

all_assets = [gee_export, train_unet, evaluate_model]