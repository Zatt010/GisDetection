from dagster import ConfigurableResource
from pydantic import Field
import os


class PipelineConfig(ConfigurableResource):
    # ── Paths ──────────────────────────────────────────────────────────────
    base_path: str = Field(
        default=r"C:\Users\afuhe\Desktop\materias\PG\Scripts\IA",
        description="Root folder for all pipeline artifacts"
    )
    gee_folder: str = Field(
        default="CNN_Training_Data",
        description="Google Drive folder where GEE exports go"
    )
    gee_project_id: str = Field(
        default="aifinal-480001",
        description="Google Earth Engine project ID"
    )

    # ── Dataset ────────────────────────────────────────────────────────────
    aoi_coords: list = Field(
        default=[-66.35, -17.50, -65.90, -17.20],
        description="[west, south, east, north] bounding box for GEE export"
    )
    date_start: str = Field(default="2023-05-01")
    date_end:   str = Field(default="2023-09-30")
    cloud_pct:  int = Field(default=15)

    # ── Model hyperparameters ──────────────────────────────────────────────
    patch_size:  int   = Field(default=64)
    patch_step:  int   = Field(default=32)
    channels:    int   = Field(default=13,  description="9 bands + 4 indices")
    num_classes: int   = Field(default=7)
    batch_size:  int   = Field(default=16)
    epochs:      int   = Field(default=100)
    learning_rate: float = Field(default=1e-3)
    test_size:   float = Field(default=0.2)

    # ── Class definitions ──────────────────────────────────────────────────
    class_names: list = Field(
        default=[
            "Bosque", "Matorrales", "Pastizales",
            "Tierras_Agricolas", "Infraestructura",
            "Suelo_Desnudo", "Agua"
        ]
    )
    # WorldCover pixel 
    class_mapping: dict = Field(
        default={"10": 0, "20": 1, "30": 2, "40": 3, "50": 4, "60": 5, "80": 6}
    )

    # ── Derived paths (properties) ─────────────────────────────────────────
    @property
    def img_path(self) -> str:
        return os.path.join(self.base_path, "Tif", "S2_Data_v4.tif")

    @property
    def label_path(self) -> str:
        return os.path.join(self.base_path, "Entrenamiento", "Labels_Data_v4.tif")

    @property
    def model_output_path(self) -> str:
        return os.path.join(self.base_path, "modelo_unet_pro_v4.keras")

    @property
    def metrics_output_path(self) -> str:
        return os.path.join(self.base_path, "metrics_v4.json")

    @property
    def plots_dir(self) -> str:
        path = os.path.join(self.base_path, "plots")
        os.makedirs(path, exist_ok=True)
        return path