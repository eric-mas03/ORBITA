"""Metadata-only confirmation acquisition tests; no real outcomes are inspected."""
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "confirmation_acquisition", Path(__file__).resolve().parents[1] / "scripts/acquire_confirmation_holdout.py")
acquisition = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(acquisition)


def test_selection_is_reproducible_and_excludes_names_across_sites():
    candidates = {"A": ["p1.psv", "p2.psv", "p3.psv", "p4.psv"],
                  "B": ["p3.psv", "p4.psv", "p5.psv", "p6.psv", "p7.psv"]}
    first = acquisition.select_unused(candidates, {"p1.psv"}, seed=99, per_site=2)
    second = acquisition.select_unused({k: list(reversed(v)) for k, v in candidates.items()},
                                       {"p1.psv"}, seed=99, per_site=2)
    assert first == second
    names = [name for selected in first.values() for name in selected]
    assert len(names) == len(set(names)) == 4
    assert "p1.psv" not in names
    with pytest.raises(ValueError, match="Insufficient"):
        acquisition.select_unused({"A": ["p1.psv"]}, {"p1.psv"}, per_site=1)


def test_structure_only_validation_treats_fields_as_opaque():
    header = "|".join(acquisition.EXPECTED_COLUMNS)
    opaque = "|".join(["not-inspected"] * len(acquisition.EXPECTED_COLUMNS))
    result = acquisition.validate_psv_metadata((header + "\n" + opaque + "\n").encode())
    assert result == {"columns": 41, "rows": 1, "value_inspection": False}
    with pytest.raises(ValueError, match="header"):
        acquisition.validate_psv_metadata(b"unexpected|header\n")
    with pytest.raises(ValueError, match="row structure"):
        acquisition.validate_psv_metadata((header + "\nwrong|shape\n").encode())


def test_existing_inventory_uses_raw_and_manifest_without_opening_values(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "p000001.psv").write_bytes(b"opaque unread source")
    manifest = tmp_path / "external.json"
    manifest.write_text(json.dumps({"files": [{"path": "another/place/p000002.psv"}]}))
    assert acquisition.existing_names(raw, manifest) == {"p000001.psv", "p000002.psv"}


def test_selection_is_frozen_before_download_and_tampering_is_rejected(tmp_path):
    raw = tmp_path / "source"
    raw.mkdir()
    for site, base in zip(acquisition.SITES, [0, 100000]):
        names = [f"p{base+i:06d}.psv" for i in range(180)]
        index = "".join(f'<a href="{name}">{name}</a>' for name in names)
        (raw / f"{site}_index.html").write_text(index)
    manifest = tmp_path / "manifests/confirmation.json"
    external = tmp_path / "external.json"
    destination = tmp_path / "confirmation"
    first = acquisition.freeze_inventory(tmp_path, raw, destination, manifest, external)
    assert first["status"] == "identities_frozen_before_download"
    assert first["selected_episodes"] == 300
    assert not destination.exists()
    assert not first["inspection"]["outcomes_or_values_inspected"]
    assert acquisition.freeze_inventory(tmp_path, raw, destination, manifest, external) == first
    first["selection_sha256"] = "tampered"
    manifest.write_text(json.dumps(first))
    with pytest.raises(ValueError, match="integrity"):
        acquisition.freeze_inventory(tmp_path, raw, destination, manifest, external)


def test_cached_hash_mismatch_is_reported_without_overwrite(tmp_path):
    destination = tmp_path / "confirmation"
    destination.mkdir()
    target = destination / "LICENSE.txt"
    target.write_bytes(b"opaque license")
    manifest = {"files": [{"kind": "license", "path": "confirmation/LICENSE.txt",
                            "url": "https://physionet.org/unused", "sha256": "incorrect"}]}
    result = acquisition.acquire(manifest, tmp_path, destination)
    assert result["status"] == "partial_acquisition_retry_frozen_selection"
    assert result["failures_last_run"][0]["error_type"] == "ValueError"
    assert target.read_bytes() == b"opaque license"


def test_unsafe_target_never_written(tmp_path):
    manifest = {"files": [{"kind": "license", "path": "outside/LICENSE.txt", "url": "unused"}]}
    with pytest.raises(ValueError, match="Unsafe"):
        acquisition.acquire(manifest, tmp_path, tmp_path / "confirmation")
    assert not (tmp_path / "outside").exists()
