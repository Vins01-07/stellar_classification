"""
Integration and Unit Tests for Astronomical Catalog Resolution and Inference Flow.

Tests:
1. Catalog resolution for bright stars (Sirius, Betelgeuse) with missing SDSS photometry.
2. Catalog resolution for standard benchmark extragalactic objects (M87, 3C 273).
3. Demonstration benchmark application preserving true physical redshift.
4. Mode 3 manual input workflow reaching the ML inference engine.
5. SDSS J-coordinate designation parsing and local sample index lookups.
6. Error handling for invalid inputs and missing inference parameters.
"""

import sys
import os
import pytest

# Import stellar_classification and app modules
import stellar_classification as sc
import app


def _get_resolved_results():
    """Helper resolving canonical catalog objects for both pytest and direct execution."""
    test_queries = ["Sirius", "Betelgeuse", "M87", "3C 273"]
    results = {}
    for q in test_queries:
        results[q] = app.resolve_astronomical_object(q)
    return results


@pytest.fixture
def resolved_results():
    """Pytest fixture resolving canonical catalog objects."""
    return _get_resolved_results()


def test_catalog_resolution(resolved_results):
    """
    Step 1: Test catalog resolution and data integrity.
    Uses assertions rather than returning a dictionary.
    """
    print("=" * 80)
    print("STEP 1: TESTING CATALOG RESOLUTION & DATA INTEGRITY")
    print("=" * 80)

    # 1. Sirius assertion
    sirius = resolved_results["Sirius"]
    assert sirius["status"] == "success", f"Sirius resolution failed: {sirius.get('message')}"
    assert sirius["missing_photometry"] is True, "Sirius must flag missing photometry due to SDSS saturation!"
    assert sirius["u"] is None and sirius["r"] is None, "Sirius must not have fabricated SDSS optical magnitudes!"
    assert sirius["redshift"] == -0.000018, f"Sirius radial velocity redshift altered: {sirius.get('redshift')}"
    assert "STAR" in sirius["known_type"], f"Sirius expected to be a STAR, got {sirius.get('known_type')}"
    assert "benchmark_photometry" in sirius and sirius["benchmark_photometry"]["spectral_type"] == "A1V"

    # 2. Betelgeuse assertion
    betel = resolved_results["Betelgeuse"]
    assert betel["status"] == "success", f"Betelgeuse resolution failed: {betel.get('message')}"
    assert betel["missing_photometry"] is True, "Betelgeuse must flag missing photometry due to SDSS saturation!"
    assert betel["u"] is None and betel["r"] is None, "Betelgeuse must not have fabricated SDSS optical magnitudes!"
    assert betel["redshift"] == 0.000073, f"Betelgeuse radial velocity redshift altered: {betel.get('redshift')}"
    assert "STAR" in betel["known_type"], f"Betelgeuse expected to be a STAR, got {betel.get('known_type')}"
    assert "benchmark_photometry" in betel and betel["benchmark_photometry"]["spectral_type"] == "M2Ib"

    # 3. M87 assertion
    m87 = resolved_results["M87"]
    assert m87["status"] == "success", f"M87 resolution failed: {m87.get('message')}"
    assert m87["missing_photometry"] is False, "M87 should have valid catalog photometry!"
    assert m87["u"] == 12.80 and m87["g"] == 10.95 and m87["redshift"] == 0.00428
    assert "GALAXY" in m87["known_type"]

    # 4. 3C 273 assertion
    c273 = resolved_results["3C 273"]
    assert c273["status"] == "success", f"3C 273 resolution failed: {c273.get('message')}"
    assert c273["missing_photometry"] is False, "3C 273 should have valid catalog photometry!"
    assert c273["u"] == 13.12 and c273["redshift"] == 0.1583
    assert "QSO" in c273["known_type"]

    print("-> SUCCESS: All catalog records resolved with strict astrophysical data integrity.")


def test_inference_data_flow(resolved_results):
    """
    Step 2: Test inference data flow, missing photometry guards, and valid predictions.
    """
    print("\n" + "=" * 80)
    print("STEP 2: TESTING INFERENCE DATA FLOW & PREDICTION GUARDS")
    print("=" * 80)

    # 1. Sirius Initial State (Missing SDSS photometry due to detector saturation)
    sirius_data = resolved_results["Sirius"]
    is_missing_sirius = (
        sirius_data.get("missing_photometry", False) or
        sirius_data.get("u") is None or
        sirius_data.get("redshift") is None
    )
    assert is_missing_sirius is True, "Sirius must trigger missing photometry guard to prevent fake prediction!"

    # 2. Sirius with Standard Calibrated A1V Photometry Benchmark
    sirius_calibrated_input = {
        "u": 14.30,
        "g": 14.10,
        "r": 14.12,
        "i": 14.15,
        "z": 14.20,
        "redshift": sirius_data.get("redshift")  # Real resolved redshift preserved (-0.000018)
    }
    assert sirius_calibrated_input["redshift"] == -0.000018, "Resolved redshift must NOT be replaced with generic default!"
    pred_sirius = sc.predict_stellar_object(sirius_calibrated_input)
    assert pred_sirius["predicted_class"] == "STAR", f"Expected STAR for Sirius with A1V photometry, got {pred_sirius['predicted_class']}"
    assert pred_sirius["confidence_pct"] > 70.0, "Expected high confidence for Sirius A1V standard profile"

    # 3. Betelgeuse with Calibrated M2Ib Red Supergiant Benchmark
    betel_data = resolved_results["Betelgeuse"]
    betel_calibrated_input = {
        "u": 17.80,
        "g": 15.10,
        "r": 13.40,
        "i": 12.60,
        "z": 12.10,
        "redshift": betel_data.get("redshift")  # Real resolved redshift preserved (0.000073)
    }
    assert betel_calibrated_input["redshift"] == 0.000073, "Betelgeuse resolved redshift must NOT be altered!"
    pred_betel = sc.predict_stellar_object(betel_calibrated_input)
    assert pred_betel["predicted_class"] == "STAR", f"Expected STAR for Betelgeuse with M2Ib photometry, got {pred_betel['predicted_class']}"
    assert pred_betel["confidence_pct"] > 70.0, "Expected high confidence for Betelgeuse M2Ib profile"

    # 4. M87 Verification
    m87_data = resolved_results["M87"]
    m87_input = {
        "u": m87_data["u"],
        "g": m87_data["g"],
        "r": m87_data["r"],
        "i": m87_data["i"],
        "z": m87_data["z"],
        "redshift": m87_data["redshift"]
    }
    assert m87_input["u"] == 12.80 and m87_input["redshift"] == 0.00428, "M87 values altered unexpectedly!"
    pred_m87 = sc.predict_stellar_object(m87_input)
    assert pred_m87["predicted_class"] == "GALAXY", f"Expected GALAXY for M87, got {pred_m87['predicted_class']}"

    # 5. 3C 273 Verification
    c273_data = resolved_results["3C 273"]
    c273_input = {
        "u": c273_data["u"],
        "g": c273_data["g"],
        "r": c273_data["r"],
        "i": c273_data["i"],
        "z": c273_data["z"],
        "redshift": c273_data["redshift"]
    }
    assert c273_input["u"] == 13.12 and c273_input["redshift"] == 0.1583, "3C 273 values altered unexpectedly!"
    pred_c273 = sc.predict_stellar_object(c273_input)
    assert "predicted_class" in pred_c273 and "confidence_pct" in pred_c273
    assert "probabilities" in pred_c273

    print("-> SUCCESS: Inference data flow correctly handles benchmarks, real observations, and missing-data guards.")


def test_manual_mode_inference():
    """
    Step 3: Test that manual inputs (Mode 3) reach the inference engine and execute reliably.
    """
    print("\n" + "=" * 80)
    print("STEP 3: TESTING MANUAL MODE 3 INFERENCE PIPELINE")
    print("=" * 80)

    # Synthetic candidate input entered manually by user
    manual_input = {
        "u": 18.20,
        "g": 17.10,
        "r": 16.60,
        "i": 16.40,
        "z": 16.30,
        "redshift": 0.0001
    }
    pred = sc.predict_stellar_object(manual_input)

    assert "predicted_class" in pred
    assert pred["predicted_class"] in ["STAR", "GALAXY", "QSO"]
    assert 0.0 <= pred["confidence_pct"] <= 100.0
    assert "probabilities" in pred
    assert abs(sum(pred["probabilities"].values()) - 1.0) < 0.01
    assert "color_indices" in pred
    assert "anomaly_score" in pred
    assert "fuzzy_quasar_likelihood" in pred
    assert "diagnostic_report" in pred

    # Quasar candidate input
    qso_manual = {
        "u": 16.32,
        "g": 15.94,
        "r": 15.81,
        "i": 15.75,
        "z": 15.65,
        "redshift": 2.2190
    }
    pred_qso = sc.predict_stellar_object(qso_manual)
    assert pred_qso["predicted_class"] == "QSO", f"Expected QSO for z=2.219 with UV excess, got {pred_qso['predicted_class']}"

    print("-> SUCCESS: Manual mode values reach inference cleanly and produce structured predictions.")


def test_coordinate_and_index_searches():
    """
    Step 4: Test SDSS J-coordinate designation parsing and local index lookups.
    """
    print("\n" + "=" * 80)
    print("STEP 4: TESTING SDSS J-COORDINATE AND INDEX LOOKUPS")
    print("=" * 80)

    # 1. SDSS J-format designation parsing
    j_query = "SDSS J000000.45+004040.3"
    res_j = app.resolve_astronomical_object(j_query)
    assert res_j["status"] == "success", f"Failed to parse or resolve J-coordinate query: {res_j.get('message')}"
    assert abs(res_j["ra"] - 0.001875) < 0.005, f"RA miscalculated: {res_j['ra']}"
    assert abs(res_j["dec"] - 0.677861) < 0.005, f"Dec miscalculated: {res_j['dec']}"

    # 2. Local DR14 survey index lookups
    idx_0 = app.resolve_astronomical_object("0")
    assert idx_0["status"] == "success"
    assert "SDSS DR14 Target #0" in idx_0["object_name"]
    assert idx_0["missing_photometry"] is False
    assert idx_0["u"] is not None and idx_0["redshift"] is not None

    idx_9999 = app.resolve_astronomical_object("9999")
    assert idx_9999["status"] == "success"
    assert "SDSS DR14 Target #9999" in idx_9999["object_name"]

    print("-> SUCCESS: SDSS coordinate parser and index lookup verified.")


def test_invalid_inputs_handling():
    """
    Step 5: Test error handling for invalid queries, out-of-bounds indices, and missing parameters.
    """
    print("\n" + "=" * 80)
    print("STEP 5: TESTING ERROR HANDLING AND INPUT VALIDATION")
    print("=" * 80)

    # Empty query
    res_empty = app.resolve_astronomical_object("")
    assert res_empty["status"] == "error"

    # Out of range index
    res_oob = app.resolve_astronomical_object("10000")
    assert res_oob["status"] == "error"
    assert "out of range" in res_oob["message"].lower()

    # Completely unrecognized object
    res_unknown = app.resolve_astronomical_object("NonExistentCelestialTarget123XYZ")
    assert res_unknown["status"] == "error"
    assert "could not resolve" in res_unknown["message"].lower()

    # Missing parameter in predict_stellar_object must raise ValueError
    with pytest.raises(ValueError, match="Missing required parameter"):
        sc.predict_stellar_object({"u": 18.0, "g": 17.0})

    with pytest.raises(ValueError, match="Missing required parameter"):
        sc.predict_stellar_object({"u": 18.0, "g": 17.0, "r": 16.0, "i": 15.0, "z": 14.0, "redshift": None})

    print("-> SUCCESS: Error handling properly validates inputs and blocks incomplete parameters.")


if __name__ == "__main__":
    fixtures = _get_resolved_results()
    test_catalog_resolution(fixtures)
    test_inference_data_flow(fixtures)
    test_manual_mode_inference()
    test_coordinate_and_index_searches()
    test_invalid_inputs_handling()
    print("\n" + "=" * 80)
    print("ALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
