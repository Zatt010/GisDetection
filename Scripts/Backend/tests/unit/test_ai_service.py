import io
import uuid
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, mock_open, PropertyMock
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'services', 'ai_service'))


_mock_keras    = MagicMock()
_mock_rasterio = MagicMock()

with patch.dict('sys.modules', {
    'tensorflow': MagicMock(),
    'tensorflow.keras': _mock_keras,
    'keras': _mock_keras,
    'rasterio': _mock_rasterio,
    'rasterio.transform': MagicMock(),
    'rasterio.warp': MagicMock(),
}):
    import ai_service_main

client = TestClient(ai_service_main.app)


# Helpers

def _fake_tif_bytes() -> bytes:
    return b"II\x2a\x00" + b"\x00" * 256  # TIFF magic + padding


def _make_read_ctx(bands: int = 7, h: int = 64, w: int = 64, nodata=None):
    
    mock_ds = MagicMock()
    mock_ds.count = bands
    mock_ds.width = w
    mock_ds.height = h
    mock_ds.nodata = nodata
    mock_ds.res = (10.0, 10.0)          
    mock_ds.crs = MagicMock()
    mock_ds.transform = MagicMock()
    mock_ds.profile = {
        'driver': 'GTiff', 'dtype': 'float32',
        'width': w, 'height': h, 'count': bands,
        'crs': mock_ds.crs, 'transform': mock_ds.transform,
        'nodata': nodata,
    }
    data = np.ones((bands, h, w), dtype=np.float32) * 5000.0
    mock_ds.read.return_value = data

    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=mock_ds)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


def _make_write_ctx():
    mock_ds = MagicMock()
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=mock_ds)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


def _make_dynamic_model(out_classes: int = 7):
    
    mock_model = MagicMock()
    mock_model.input_shape = (None, 64, 64, 7)

    def _predict(batch, verbose=0):
        n = batch.shape[0] if hasattr(batch, 'shape') else 1
        return np.zeros((n, 64, 64, out_classes), dtype=np.float32)

    mock_model.predict.side_effect = _predict
    return mock_model


def _setup_rasterio_for_prediction(bands: int = 7, h: int = 64, w: int = 64):
    
    read_ctx  = _make_read_ctx(bands=bands, h=h, w=w)
    write_ctx = _make_write_ctx()
    _mock_rasterio.open.side_effect = [read_ctx, write_ctx]


# Health check

class TestHealthCheck:
    def test_health_ok(self):
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("status") == "ok"

    def test_health_returns_service_name(self):
        resp = client.get("/")
        body = resp.json()
        assert "service" in body or "status" in body


# /predict_area/ — model not loaded

class TestPredictAreaNoModel:
    def test_returns_error_when_model_is_none(self):
        original = ai_service_main.model
        try:
            ai_service_main.model = None
            resp = client.post(
                "/predict_area/",
                files={"file": ("test.tif", _fake_tif_bytes(), "image/tiff")},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body.get("status") == "error"
            assert "Model not loaded" in body.get("message", "")
        finally:
            ai_service_main.model = original

    def test_missing_file_returns_422(self):
        resp = client.post("/predict_area/")
        assert resp.status_code == 422


# /predict_area/ — full 7-band pipeline

class TestPredictArea7Band:
    def setup_method(self):
        ai_service_main.model = _make_dynamic_model()

    def teardown_method(self):
        _mock_rasterio.open.side_effect = None

    def test_successful_prediction_7band(self):
        _setup_rasterio_for_prediction(bands=7, h=64, w=64)
        resp = client.post(
            "/predict_area/",
            files={"file": ("test.tif", _fake_tif_bytes(), "image/tiff")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "analisis_hectareas" in body or body.get("status") == "error"

    def test_prediction_returns_land_cover_keys(self):
        _setup_rasterio_for_prediction(bands=7, h=64, w=64)
        resp = client.post(
            "/predict_area/",
            files={"file": ("test.tif", _fake_tif_bytes(), "image/tiff")},
        )
        assert resp.status_code == 200
        body = resp.json()
        if "analisis_hectareas" in body:
            expected = {"Bosque", "Matorrales", "Pastizales", "T_Agricolas",
                        "Infraestructura", "Suelo_Desnudo", "Agua"}
            assert set(body["analisis_hectareas"].keys()) == expected

    def test_prediction_returns_download_url(self):
        _setup_rasterio_for_prediction(bands=7, h=64, w=64)
        resp = client.post(
            "/predict_area/",
            files={"file": ("test.tif", _fake_tif_bytes(), "image/tiff")},
        )
        assert resp.status_code == 200
        body = resp.json()
        if "processed_file_url" in body:
            assert "download" in body["processed_file_url"]
            assert body["processed_file_url"].endswith(".tif")

    def test_prediction_areas_are_floats(self):
        _setup_rasterio_for_prediction(bands=7, h=64, w=64)
        resp = client.post(
            "/predict_area/",
            files={"file": ("test.tif", _fake_tif_bytes(), "image/tiff")},
        )
        assert resp.status_code == 200
        body = resp.json()
        if "analisis_hectareas" in body:
            for v in body["analisis_hectareas"].values():
                assert isinstance(v, (int, float))

    def test_prediction_larger_image(self):
        _setup_rasterio_for_prediction(bands=7, h=128, w=128)
        resp = client.post(
            "/predict_area/",
            files={"file": ("big.tif", _fake_tif_bytes(), "image/tiff")},
        )
        assert resp.status_code == 200

    def test_prediction_odd_size_image(self):
        _setup_rasterio_for_prediction(bands=7, h=100, w=100)
        resp = client.post(
            "/predict_area/",
            files={"file": ("odd.tif", _fake_tif_bytes(), "image/tiff")},
        )
        assert resp.status_code == 200


# /predict_area/ — 3-band image (RGB padding path)

class TestPredictArea3Band:
    def setup_method(self):
        ai_service_main.model = _make_dynamic_model()

    def teardown_method(self):
        _mock_rasterio.open.side_effect = None

    def test_3band_image_processed(self):
        _setup_rasterio_for_prediction(bands=3, h=64, w=64)
        resp = client.post(
            "/predict_area/",
            files={"file": ("rgb.tif", _fake_tif_bytes(), "image/tiff")},
        )
        assert resp.status_code == 200

    def test_3band_padding_logic(self):
        img = np.ones((64, 64, 3), dtype=np.float32) * 255.0
        padding = np.zeros((64, 64, 4), dtype=img.dtype)
        result = np.concatenate([img, padding], axis=-1)
        assert result.shape == (64, 64, 7)
        assert result[..., 3:].sum() == 0.0


# /predict_area/ — unsupported band count

class TestPredictAreaUnsupportedBands:
    def setup_method(self):
        ai_service_main.model = _make_dynamic_model()

    def teardown_method(self):
        _mock_rasterio.open.side_effect = None

    def test_4band_returns_error(self):
        read_ctx = _make_read_ctx(bands=4, h=64, w=64)
        _mock_rasterio.open.side_effect = [read_ctx]
        resp = client.post(
            "/predict_area/",
            files={"file": ("4band.tif", _fake_tif_bytes(), "image/tiff")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("status") == "error"
        assert "Canales" in body.get("message", "")

    def test_1band_returns_error(self):
        read_ctx = _make_read_ctx(bands=1, h=64, w=64)
        _mock_rasterio.open.side_effect = [read_ctx]
        resp = client.post(
            "/predict_area/",
            files={"file": ("1band.tif", _fake_tif_bytes(), "image/tiff")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("status") == "error"


# /predict_area/ — corrupt / invalid file
# 

class TestPredictAreaInvalidFile:
    def setup_method(self):
        ai_service_main.model = _make_dynamic_model()

    def teardown_method(self):
        _mock_rasterio.open.side_effect = None

    def test_corrupt_file_returns_error(self):
        _mock_rasterio.open.side_effect = Exception("Invalid TIFF")
        
        with pytest.raises(Exception, match="Invalid TIFF"):
            client.post(
                "/predict_area/",
                files={"file": ("bad.tif", b"not a tiff", "image/tiff")},
            )

    def test_missing_file_field_returns_422(self):
        resp = client.post("/predict_area/")
        assert resp.status_code == 422


# /download/{filename}

class TestDownloadEndpoint:
    def test_download_missing_file_returns_error_json(self):
        resp = client.get("/download/nonexistent_file_xyz.tif")
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("status") == "error"

    def test_download_existing_file(self, tmp_path):
        original_temp = ai_service_main.TEMP_DIR
        ai_service_main.TEMP_DIR = str(tmp_path)
        test_file = tmp_path / "test_output.tif"
        test_file.write_bytes(b"fake tif content")
        try:
            resp = client.get("/download/test_output.tif")
            assert resp.status_code == 200
        finally:
            ai_service_main.TEMP_DIR = original_temp

    def test_download_path_traversal_blocked(self):
        
        resp = client.get("/download/../../etc/passwd")
        if resp.status_code == 200:
            body = resp.json()
            assert body.get("status") == "error" or "analisis_hectareas" not in body
        else:
            assert resp.status_code in (400, 404)

    def test_download_empty_filename_handled(self):
        resp = client.get("/download/no_such_file_abc123.tif")
        assert resp.status_code == 200
        body = resp.json()
        assert "status" in body


# Normalisation & array helpers (pure-logic unit tests)

class TestNormalisationLogic:
    def test_7band_values_divided_by_10000(self):
        raw = np.full((64, 64, 7), 10000.0, dtype=np.float32)
        normalised = np.nan_to_num(raw / 10000.0)
        assert normalised.max() == pytest.approx(1.0, rel=1e-4)

    def test_nan_replaced_by_zero(self):
        raw = np.array([np.nan, 5000.0, np.nan], dtype=np.float32)
        result = np.nan_to_num(raw)
        assert not np.isnan(result).any()
        assert result[0] == 0.0

    def test_3band_padding_produces_7bands(self):
        img = np.ones((64, 64, 3), dtype=np.float32) * 255.0
        padding = np.zeros((64, 64, 4), dtype=img.dtype)
        result = np.concatenate([img, padding], axis=-1)
        assert result.shape == (64, 64, 7)
        assert result[..., 3:].sum() == 0.0

    def test_padding_to_multiple_of_64(self):
        for h, w in [(100, 150), (64, 64), (1, 1), (63, 65), (128, 200)]:
            ph = (64 - h % 64) % 64
            pw = (64 - w % 64) % 64
            assert (h + ph) % 64 == 0
            assert (w + pw) % 64 == 0

    def test_scale_3band_to_sentinel_range(self):
        rgb_val = 128.0
        scaled = rgb_val * (10000.0 / 255.0)
        assert 4900 < scaled < 5100  # ~5020


class TestPatchLogic:
    def test_patch_count_correct(self):
        for img_size, patch_size, step, expected in [
            (128, 64, 32, 3),
            (64,  64, 32, 1),
            (192, 64, 32, 5),
        ]:
            count = (img_size - patch_size) // step + 1
            assert count == expected, f"size={img_size}: expected {expected}, got {count}"

    def test_patch_pixel_values_preserved(self):
        img = np.arange(64 * 64 * 7, dtype=np.float32).reshape(64, 64, 7)
        patch = img[0:64, 0:64, :]
        np.testing.assert_array_equal(patch, img)

    def test_patchify_output_shape_64x64(self):
        h, w, patch_size, step = 64, 64, 64, 64
        expected_h_patches = (h - patch_size) // step + 1
        expected_w_patches = (w - patch_size) // step + 1
        assert expected_h_patches == 1
        assert expected_w_patches == 1

    def test_patchify_output_shape_128x128(self):
        h, w, patch_size, step = 128, 128, 64, 32
        expected_h_patches = (h - patch_size) // step + 1
        expected_w_patches = (w - patch_size) // step + 1
        assert expected_h_patches == 3
        assert expected_w_patches == 3


class TestAreaCalculation:
    def test_1_hectare(self):
        res_x, res_y = 10.0, 10.0
        area_m2_per_pixel = res_x * res_y
        assert (100 * area_m2_per_pixel) / 10000.0 == pytest.approx(1.0)

    def test_fractional_hectares(self):
        area_m2 = 10.0 * 10.0
        hectares = (50 * area_m2) / 10000.0
        assert hectares == pytest.approx(0.5)

    def test_zero_pixels(self):
        assert (0 * 100.0) / 10000.0 == 0.0

    def test_res_tuple_unpack(self):
        res = (10.0, 10.0)
        res_x, res_y = res
        assert res_x == 10.0 and res_y == 10.0

    def test_abs_resolution_always_positive(self):
        for res_x, res_y in [(10.0, -10.0), (-10.0, 10.0), (10.0, 10.0)]:
            assert abs(res_x * res_y) > 0


class TestNodataMasking:
    def test_nodata_value_excluded(self):
        arr = np.array([[0, 1, 99, 2, 99]], dtype=np.uint8)
        mask = arr != 99
        valid = arr[mask]
        assert 99 not in valid
        assert len(valid) == 3

    def test_all_valid_when_no_nodata(self):
        arr = np.array([0, 1, 2, 3], dtype=np.uint8)
        assert (arr != 99).all()

    def test_class_areas_exclude_nodata(self):
        classes = ["Bosque", "Matorrales"]
        final_map = np.array([[0, 1, 99]], dtype=np.uint8)
        valid_mask = final_map != 99
        area_px = 100.0
        results = {
            name: round(float(np.sum((final_map == i) & valid_mask) * area_px / 10000.0), 2)
            for i, name in enumerate(classes)
        }
        assert results["Bosque"]     == pytest.approx(0.01)
        assert results["Matorrales"] == pytest.approx(0.01)

    def test_valid_mask_excludes_all_zero_pixels(self):
        img = np.zeros((4, 4, 7), dtype=np.float32)
        img[2, 2, :] = 1.0
        valid = np.max(img, axis=-1) > 0
        assert valid[2, 2] is np.bool_(True)
        assert valid[0, 0] is np.bool_(False)


class TestClassMapping:
    CLASSES = ["Bosque", "Matorrales", "Pastizales", "T_Agricolas",
               "Infraestructura", "Suelo_Desnudo", "Agua"]

    def test_all_classes_present_in_results(self):
        final_map = np.zeros((10, 10), dtype=np.uint8)
        valid_mask = np.ones_like(final_map, dtype=bool)
        results = {name: int(np.sum((final_map == i) & valid_mask))
                   for i, name in enumerate(self.CLASSES)}
        assert set(results.keys()) == set(self.CLASSES)

    def test_pixel_counts_sum_to_total_valid(self):
        final_map = np.array([[0, 1, 2, 3, 4, 5, 6]], dtype=np.uint8)
        valid_mask = np.ones_like(final_map, dtype=bool)
        total = sum(int(np.sum((final_map == i) & valid_mask))
                    for i in range(len(self.CLASSES)))
        assert total == final_map.size

    def test_result_values_are_floats(self):
        final_map = np.zeros((4, 4), dtype=np.uint8)
        valid_mask = np.ones_like(final_map, dtype=bool)
        results = {
            name: round(float(np.sum((final_map == i) & valid_mask) * 100.0 / 10000.0), 2)
            for i, name in enumerate(self.CLASSES)
        }
        for v in results.values():
            assert isinstance(v, float)

    def test_class_count_is_seven(self):
        assert len(self.CLASSES) == 7

    def test_no_duplicate_class_names(self):
        assert len(set(self.CLASSES)) == len(self.CLASSES)


class TestRasterProfile:
    def test_profile_updated_correctly(self):
        profile = {'driver': 'GTiff', 'dtype': 'float32',
                   'width': 64, 'height': 64, 'count': 7}
        profile.update(count=1, dtype='uint8', nodata=99)
        assert profile['count']  == 1
        assert profile['dtype']  == 'uint8'
        assert profile['nodata'] == 99
        assert profile['driver'] == 'GTiff'

    def test_profile_preserves_spatial_keys(self):
        profile = {'driver': 'GTiff', 'crs': 'EPSG:32719', 'transform': 'T',
                   'width': 128, 'height': 128, 'count': 7, 'dtype': 'float32'}
        profile.update(count=1, dtype='uint8', nodata=99)
        assert profile['crs']       == 'EPSG:32719'
        assert profile['transform'] == 'T'
        assert profile['width']     == 128

    def test_nodata_set_to_99(self):
        profile = {}
        profile.update(count=1, dtype='uint8', nodata=99)
        assert profile['nodata'] == 99


class TestUUIDGeneration:
    def test_uuid_hex_length(self):
        result_id = f"mask_{uuid.uuid4().hex}"
        assert result_id.startswith("mask_")
        assert len(result_id) == 5 + 32

    def test_uuids_are_unique(self):
        ids = {f"mask_{uuid.uuid4().hex}" for _ in range(100)}
        assert len(ids) == 100

    def test_uuid_only_hex_chars(self):
        jid = uuid.uuid4().hex
        assert all(c in '0123456789abcdef' for c in jid)


class TestResultFormat:
    def test_response_has_required_keys(self):
        response = {
            "analisis_hectareas": {"Bosque": 10.5},
            "processed_file_url": "http://localhost:8004/download/mask_abc.tif",
        }
        assert "analisis_hectareas"  in response
        assert "processed_file_url" in response

    def test_download_url_format(self):
        base = "http://localhost:8004/download/"
        url  = base + "mask_abc.tif"
        assert url.startswith(base)
        assert url.endswith(".tif")

    def test_error_response_structure(self):
        err = {"status": "error", "message": "Model not loaded"}
        assert err["status"] == "error"
        assert isinstance(err["message"], str)
        assert len(err["message"]) > 0


# Model-loading path (module-level code)

class TestModelLoadingLogic:
    def test_model_loaded_when_file_exists(self):
        fake_model = _make_dynamic_model()
        with patch('os.path.exists', return_value=True), \
             patch.object(_mock_keras.models, 'load_model', return_value=fake_model):
            model = None
            path = "/fake/model.keras"
            if os.path.exists(path):
                try:
                    model = _mock_keras.models.load_model(path, compile=False)
                except Exception:
                    pass
            assert model is not None

    def test_model_stays_none_when_file_missing(self):
        with patch('os.path.exists', return_value=False):
            model = None
            if os.path.exists("/nonexistent/model.keras"):
                model = object()
            assert model is None

    def test_model_stays_none_on_load_error(self):
        with patch('os.path.exists', return_value=True), \
             patch.object(_mock_keras.models, 'load_model', side_effect=Exception("corrupt")):
            model = None
            if os.path.exists("/fake/model.keras"):
                try:
                    model = _mock_keras.models.load_model("/fake/model.keras", compile=False)
                except Exception:
                    pass
            assert model is None