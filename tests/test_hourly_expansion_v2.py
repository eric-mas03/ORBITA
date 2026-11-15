import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("hourly_expansion", Path(__file__).resolve().parents[1] / "scripts/acquire_hourly_expansion_v2.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


@pytest.fixture
def inventory(tmp_path, monkeypatch):
    root = tmp_path
    old = root / "data/raw/external/PHYSIONET_SEPSIS_2019_1.0.0"
    new = root / "data/raw/external/PHYSIONET_EXPANSION_V2"
    manifest = root / "data/manifests/hourly_expansion_v2.json"
    old.mkdir(parents=True)
    manifest.parent.mkdir(parents=True)
    for site, start in [("training_setA", 1), ("training_setB", 100001)]:
        text = "\n".join(f'<a href="p{i:06d}.psv">episode</a>' for i in range(start, start+2000))
        (old / f"{site}_index.html").write_text(text)
    previous = root / "data/raw/external/PHYSIONET_CONFIRMATION/training_setA"
    previous.mkdir(parents=True)
    (previous / "p000001.psv").write_bytes(b"opaque clinical values must not be parsed")
    for name, value in [("ROOT", root), ("OLD", old), ("RAW", new), ("MANIFEST", manifest)]:
        monkeypatch.setattr(module, name, value)
    return module.freeze_inventory(), manifest


def test_roles_frozen_before_download_and_consume_no_previous_values(inventory):
    result, manifest = inventory
    assert len(result["files"]) == 1500
    assert all(Path(row["path"]).name != "p000001.psv" for row in result["files"])
    assert len({row["episode_id"] for row in result["files"]}) == 1500
    assert {role:sum(row["role"] == role for row in result["files"]) for role in ("train", "meta", "final")} == {"train":1000,"meta":200,"final":300}
    assert result["final_values_inspected"] is False
    assert module.freeze_inventory() == result


def test_tampered_roles_rejected_before_network(inventory):
    result, manifest = inventory
    result["files"][0]["role"] = "final"
    manifest.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="inventory changed"):
        module.freeze_inventory()
