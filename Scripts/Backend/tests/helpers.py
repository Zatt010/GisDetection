import io
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS


def make_tif_bytes(bands=7, width=128, height=128,
                   dtype="float32", pixel_value=5000.0, nodata=None) -> bytes:
    buf = io.BytesIO()
    transform = from_bounds(-66.35, -17.50, -65.90, -17.20, width, height)
    data = np.full((bands, height, width), pixel_value, dtype=dtype)
    profile = dict(driver="GTiff", dtype=dtype, width=width, height=height,
                   count=bands, crs=CRS.from_epsg(32719), transform=transform)
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(buf, "w", **profile) as dst:
        dst.write(data)
    buf.seek(0)
    return buf.read()


def make_label_tif_bytes(width=128, height=128) -> bytes:
    buf = io.BytesIO()
    transform = from_bounds(-66.35, -17.50, -65.90, -17.20, width, height)
    class_values = [10, 20, 30, 40, 50, 60, 80]
    data = np.zeros((1, height, width), dtype=np.uint8)
    for i, val in enumerate(class_values):
        col_start = i * (width // len(class_values))
        col_end   = col_start + (width // len(class_values))
        data[0, :, col_start:col_end] = val
    with rasterio.open(buf, "w", driver="GTiff", dtype="uint8",
        width=width, height=height, count=1,
        crs=CRS.from_epsg(32719), transform=transform) as dst:
        dst.write(data)
    buf.seek(0)
    return buf.read()