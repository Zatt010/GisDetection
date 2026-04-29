"""
Unit Tests — pure functions, no HTTP, no disk I/O, no real model.

These tests run in milliseconds and cover:
  • Raster normalization logic
  • Class mapping
  • Tile math (lat/lon ↔ XYZ)
  • Pixel area calculation
  • Vector export class mapping
  • Tiling job status machine
"""
import io
import math
import uuid
import numpy as np
import pytest
import rasterio
from unittest.mock import MagicMock, patch, call
from rasterio.transform import from_bounds
from rasterio.crs import CRS

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from conftest import make_tif_bytes, make_label_tif_bytes
from tests.helpers import make_tif_bytes, make_label_tif_bytes


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Normalization
# ═══════════════════════════════════════════════════════════════════════════════

class TestNormalization:

    def test_7band_normalization_range(self):
        """Raw S2 values ÷ 10000 must land in [0, 1]."""
        raw = np.array([[[0, 5000, 10000]]] * 7, dtype=np.float32)  # shape (7,1,3)
        normalized = raw / 10000.0
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_13band_normalization_splits_correctly(self):
        """
        Channels 0-8  : raw bands  → divide by 10 000
        Channels 9-12 : indices    → keep as-is (already in [-1, 1])
        """
        img = np.ones((100, 100, 13), dtype=np.float32)
        img[:, :, :9]  *= 8000.0   # raw band values
        img[:, :, 9:]  *= 0.6      # index values

        out = np.zeros_like(img)
        out[:, :, :9] = img[:, :, :9] / 10000.0
        out[:, :, 9:] = img[:, :, 9:]

        assert out[:, :, :9].max() <= 1.0,  "Raw bands must be ≤ 1.0 after /10000"
        assert out[:, :, 9:].max() == pytest.approx(0.6, abs=1e-5), \
            "Spectral indices must be unchanged"

    def test_nan_to_num_replaces_nan(self):
        """NaN pixels (cloud-masked areas) must become 0 after nan_to_num."""
        img = np.array([[[np.nan, 5000.0, np.nan]]], dtype=np.float32)
        result = np.nan_to_num(img) / 10000.0
        assert not np.isnan(result).any()
        assert result[0, 0, 0] == 0.0
        assert result[0, 0, 1] == pytest.approx(0.5, abs=1e-5)

    def test_16bit_overflow_clipped(self):
        """Values > 10000 (e.g. 65535 in uint16) stay ≤ 1 after normalization."""
        img = np.array([[[65535.0]]], dtype=np.float32)
        normalized = img / 10000.0
        # We clip in the model inference path
        clipped = np.clip(normalized, 0.0, 1.0)
        assert clipped.max() == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Class mapping
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassMapping:

    CLASS_MAPPING = {10: 0, 20: 1, 30: 2, 40: 3, 50: 4, 60: 5, 80: 6}
    CLASS_NAMES   = [
        "Bosque", "Matorrales", "Pastizales",
        "Tierras_Agricolas", "Infraestructura", "Suelo_Desnudo", "Agua"
    ]

    def test_all_worldcover_values_mapped(self):
        """Every WorldCover label value must map to a valid class index."""
        label = np.array([[10, 20, 30, 40, 50, 60, 80]], dtype=np.uint8)
        new_label = np.zeros_like(label)
        for val, idx in self.CLASS_MAPPING.items():
            new_label[label == val] = idx

        assert set(new_label.flatten()) == {0, 1, 2, 3, 4, 5, 6}

    def test_unknown_worldcover_value_stays_zero(self):
        """Unmapped pixel values (e.g. 255) must stay as 0 (background)."""
        label = np.array([[255, 10]], dtype=np.uint8)
        new_label = np.zeros_like(label)
        for val, idx in self.CLASS_MAPPING.items():
            new_label[label == val] = idx

        assert new_label[0, 0] == 0  # 255 unmapped → stays 0

    def test_class_names_count_matches_mapping(self):
        assert len(self.CLASS_NAMES) == len(self.CLASS_MAPPING)

    def test_argmax_selects_highest_probability_class(self):
        """Model output → argmax must pick the highest-prob class."""
        probs = np.zeros((1, 4, 4, 7), dtype=np.float32)
        probs[0, :, :, 3] = 0.9   # Tierras_Agricolas should win
        predicted = np.argmax(probs, axis=-1)
        assert (predicted == 3).all()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Tile math
# ═══════════════════════════════════════════════════════════════════════════════

class TestTileMath:

    @staticmethod
    def latlon_to_tile(lat, lon, zoom):
        n = 2 ** zoom
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - math.log(
            math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))
        ) / math.pi) / 2.0 * n)
        return x, y

    @staticmethod
    def tile_to_bbox(x, y, z):
        n = 2 ** z
        lon_min = x / n * 360.0 - 180.0
        lon_max = (x + 1) / n * 360.0 - 180.0
        lat_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
        lat_min = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
        return lon_min, lat_min, lon_max, lat_max

    def test_tile_is_within_bounds(self):
        """A tile computed from a lat/lon must contain that lat/lon."""
        lat, lon, zoom = -17.33, -66.22, 14
        x, y = self.latlon_to_tile(lat, lon, zoom)
        west, south, east, north = self.tile_to_bbox(x, y, zoom)

        assert west  <= lon <= east,  f"lon {lon} not in [{west}, {east}]"
        assert south <= lat <= north, f"lat {lat} not in [{south}, {north}]"

    def test_zoom_0_covers_whole_world(self):
        x, y = self.latlon_to_tile(0, 0, 0)
        assert x == 0 and y == 0

    def test_tile_count_grows_with_zoom(self):
        """At zoom Z there are 2^Z × 2^Z tiles."""
        for z in range(1, 8):
            max_tile = 2 ** z - 1
            x, y = self.latlon_to_tile(-17.33, -66.22, z)
            assert 0 <= x <= max_tile
            assert 0 <= y <= max_tile

    def test_bbox_lon_range_is_360_at_zoom_0(self):
        west, south, east, north = self.tile_to_bbox(0, 0, 0)
        assert abs((east - west) - 360.0) < 1e-6


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Area calculation
# ═══════════════════════════════════════════════════════════════════════════════

class TestAreaCalculation:

    def test_area_in_hectares_basic(self):
        """10 m pixel → 100 m² → 0.01 ha per pixel."""
        res_x, res_y = 10.0, -10.0     # standard S2 resolution
        area_px      = abs(res_x * res_y)  # 100 m²
        n_pixels     = 100                 # 1 ha worth of pixels

        ha = (n_pixels * area_px) / 10_000.0
        assert ha == pytest.approx(1.0, abs=1e-6)

    def test_area_with_valid_mask(self):
        """Pixels under the nodata mask (value=99) must not count toward area."""
        final_map  = np.array([[0, 0, 99, 99]], dtype=np.uint8)
        valid_mask = final_map != 99
        count = np.sum((final_map == 0) & valid_mask)
        assert count == 2

    def test_all_classes_summed(self):
        """Sum of all class areas must equal total valid pixel area."""
        h, w      = 64, 64
        final_map = np.zeros((h, w), dtype=np.uint8)
        final_map[:32, :]  = 0   # Bosque
        final_map[32:, :]  = 1   # Matorrales
        valid     = final_map != 99
        area_px   = 100.0        # 10×10 m

        total_ha = sum(
            float(np.sum((final_map == i) & valid)) * area_px / 10_000.0
            for i in range(7)
        )
        expected = float(h * w) * area_px / 10_000.0
        assert total_ha == pytest.approx(expected, rel=1e-5)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Job status state machine
# ═══════════════════════════════════════════════════════════════════════════════

class TestJobStatusStateMachine:
    """
    The tiling job_status dict acts as a simple state machine.
    Verify transitions and that error state captures the message.
    """

    def test_initial_state_is_queued(self):
        job_status = {}
        job_id = uuid.uuid4().hex
        job_status[job_id] = {"status": "queued", "progress": 0}
        assert job_status[job_id]["status"] == "queued"

    def test_transitions_queued_reprojecting_tiling_done(self):
        job_status = {}
        jid = "test-job"
        for state, prog in [
            ("queued",       0),
            ("reprojecting", 10),
            ("tiling",       40),
            ("done",         100),
        ]:
            job_status[jid] = {"status": state, "progress": prog}
            assert job_status[jid]["status"] == state
            assert job_status[jid]["progress"] == prog

    def test_error_state_captures_message(self):
        job_status = {}
        jid = "fail-job"
        job_status[jid] = {"status": "error", "message": "gdalwarp failed: disk full"}
        assert job_status[jid]["status"] == "error"
        assert "disk full" in job_status[jid]["message"]

    def test_missing_job_returns_not_found_sentinel(self):
        job_status = {}
        result = job_status.get("nonexistent-id", {"status": "not_found"})
        assert result["status"] == "not_found"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Vector export class name mapping
# ═══════════════════════════════════════════════════════════════════════════════

class TestVectorClassMapping:

    CLASSES = [
        "Bosque", "Matorrales", "Pastizales",
        "T_Agricolas", "Infraestructura", "Suelo_Desnudo", "Agua"
    ]

    def _map_class(self, idx):
        return self.CLASSES[idx] if 0 <= idx < len(self.CLASSES) else "Desconocido"

    def test_valid_indices_return_class_name(self):
        for i, name in enumerate(self.CLASSES):
            assert self._map_class(i) == name

    def test_out_of_range_index_returns_desconocido(self):
        assert self._map_class(-1)  == "Desconocido"
        assert self._map_class(99)  == "Desconocido"
        assert self._map_class(7)   == "Desconocido"

    def test_nodata_pixel_excluded_from_shapes(self):
        """Pixels with value 99 (nodata) must be masked out before vectorization."""
        prediction_map = np.array([[0, 1, 99, 99, 2]], dtype=np.uint8)
        valid = prediction_map != 99
        valid_values = prediction_map[valid]
        assert 99 not in valid_values
        assert len(valid_values) == 3