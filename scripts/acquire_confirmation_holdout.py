"""Freeze and acquire unused public ICU episodes, without outcome inspection.

Only names, source-index hashes, byte hashes, header/field counts and row counts
are inspected. No numeric values, labels, quality summaries or models are read.
The acquisition directory is separate from every current train/dev/cal/test set.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import threading

import requests

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/external/PHYSIONET_SEPSIS_2019_1.0.0"
DESTINATION = ROOT / "data/raw/external/PHYSIONET_CONFIRMATION"
MANIFEST = ROOT / "data/manifests/confirmation_holdout.json"
EXTERNAL_MANIFEST = ROOT / "data/manifests/external_downloads.json"
SEED = 20261114
PER_SITE = 150
MAX_TOTAL_BYTES = 5_000_000
SITES = ("training_setA", "training_setB")
BASE = "https://physionet.org/files/challenge-2019/1.0.0/"
EXPECTED_COLUMNS = (
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "BaseExcess",
    "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos", "Calcium",
    "Chloride", "Creatinine", "Bilirubin_direct", "Glucose", "Lactate", "Magnesium",
    "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT",
    "WBC", "Fibrinogen", "Platelets", "Age", "Gender", "Unit1", "Unit2",
    "HospAdmTime", "ICULOS", "SepsisLabel",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def public_names(index: bytes) -> list[str]:
    """Extract published file identities from an index, never clinical fields."""
    return sorted(set(re.findall(r'href="(p\d+\.psv)"', index.decode("utf-8"))))


def existing_names(raw: Path = SOURCE, manifest_path: Path = EXTERNAL_MANIFEST) -> set[str]:
    names = {path.name for path in raw.rglob("*.psv")}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for record in manifest.get("files", []):
            name = Path(record["path"]).name
            if re.fullmatch(r"p\d+\.psv", name):
                names.add(name)
    return names


def select_unused(
    by_site: dict[str, list[str]], excluded_names: set[str], seed: int = SEED,
    per_site: int = PER_SITE,
) -> dict[str, list[str]]:
    """Choose uniformly from sorted unused file IDs; keep IDs unique across sites."""
    if per_site < 1:
        raise ValueError("Positive per-site size required")
    selected = {}
    forbidden = set(excluded_names)
    for site in sorted(by_site):
        candidates = sorted(set(by_site[site]) - forbidden)
        if len(candidates) < per_site:
            raise ValueError("Insufficient unused episode identities")
        rng = random.Random(f"{seed}|{site}")
        chosen = sorted(rng.sample(candidates, per_site))
        selected[site] = chosen
        forbidden.update(chosen)
    return selected


def validate_psv_metadata(payload: bytes) -> dict:
    """Validate structural metadata without parsing or summarizing any value."""
    lines = payload.splitlines()
    if not lines or lines[0].decode("utf-8-sig").split("|") != list(EXPECTED_COLUMNS):
        raise ValueError("Unexpected confirmation header")
    # Counting delimiters validates table shape. Field contents remain opaque.
    rows = [line for line in lines[1:] if line.strip()]
    if not rows or any(line.count(b"|") != len(EXPECTED_COLUMNS) - 1 for line in rows):
        raise ValueError("Unexpected confirmation row structure")
    return {"columns": len(EXPECTED_COLUMNS), "rows": len(rows), "value_inspection": False}


def freeze_inventory(
    root: Path = ROOT, raw: Path = SOURCE, destination: Path = DESTINATION,
    manifest_path: Path = MANIFEST, external_manifest: Path = EXTERNAL_MANIFEST,
) -> dict:
    """Persist identities before acquisition and never reselect on resume."""
    excluded = existing_names(raw, external_manifest)
    indexes, by_site = {}, {}
    for site in SITES:
        index_path = raw / f"{site}_index.html"
        index = index_path.read_bytes()
        by_site[site] = public_names(index)
        indexes[site] = {"path": index_path.relative_to(root).as_posix(), "sha256": sha256(index),
                         "listed_files": len(by_site[site]), "url": BASE + "training/" + site + "/"}
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved["seed"] != SEED or saved["per_site"] != PER_SITE:
            raise ValueError("Frozen selection settings differ")
        if saved["source_indexes"] != indexes:
            raise ValueError("Frozen source indexes differ")
        chosen = [Path(record["path"]).name for record in saved["files"] if record["kind"] == "episode"]
        if set(chosen) & excluded or len(chosen) != len(set(chosen)):
            raise ValueError("Frozen confirmation identities overlap existing development data")
        frozen = {site: sorted(Path(record["path"]).name for record in saved["files"]
                               if record["kind"] == "episode" and record["site"] == site) for site in SITES}
        if (any(len(names) != PER_SITE or not set(names).issubset(by_site[site])
                for site, names in frozen.items()) or
                sha256(json.dumps(frozen, sort_keys=True).encode()) != saved["selection_sha256"]):
            raise ValueError("Frozen confirmation selection integrity failed")
        if saved.get("inspection", {}).get("outcomes_or_values_inspected") is not False:
            raise ValueError("Confirmation acquisition contract changed")
        return saved
    selection = select_unused(by_site, excluded)
    records = []
    for site, names in selection.items():
        for name in names:
            path = destination / site / name
            records.append({"kind": "episode", "site": site, "path": path.relative_to(root).as_posix(),
                            "url": BASE + "training/" + site + "/" + name,
                            "status": "planned", "bytes": None, "sha256": None})
    records.append({"kind": "license", "path": (destination / "LICENSE.txt").relative_to(root).as_posix(),
                    "url": BASE + "LICENSE.txt", "status": "planned", "bytes": None, "sha256": None})
    manifest = {"schema_version": "1.0.0", "status": "identities_frozen_before_download",
                "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED, "per_site": PER_SITE,
                "selection": "uniform without replacement per site from cached indexes; random.Random(str(seed)+'|'+site)",
                "selected_episodes": sum(map(len, selection.values())), "excluded_existing_episode_names": len(excluded),
                "excluded_inventory_sha256": sha256(json.dumps(sorted(excluded)).encode()),
                "selection_sha256": sha256(json.dumps(selection, sort_keys=True).encode()),
                "source_indexes": indexes, "maximum_new_payload_bytes": MAX_TOTAL_BYTES,
                "files": records, "inspection": {"header_and_field_count": True, "row_counts": True,
                "outcomes_or_values_inspected": False, "models_evaluated": False,
                "numeric_quality_or_target_summary": False},
                "access": "public without authentication; no agreements accepted",
                "identity_limit": "No overlapping published p-file IDs. Unique people across ICU admissions cannot be established from available metadata; no identity inference from clinical values.",
                "license_note": "Landing page declares CC-BY-4.0; bundled LICENSE.txt declares ODbL. Preserve both provenance statements; no redistribution license inferred.",
                "domain": "terrestrial ICU, not astronaut or external-hospital validation",
                "isolation": "Outside current packed train/dev/calibration/test data; final confirmation after development-only champion freeze, not adaptive tuning."}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def acquire(manifest: dict, root: Path = ROOT, destination: Path = DESTINATION) -> dict:
    """Acquire immutable bytes and structural metadata; no clinical interpretation."""
    destination.mkdir(parents=True, exist_ok=True)
    destination = destination.resolve()
    for record in manifest["files"]:
        target = (root / record["path"]).resolve()
        if not target.is_relative_to(destination):
            raise ValueError("Unsafe confirmation target")
        target.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    total = 0

    def one(record: dict) -> dict:
        nonlocal total
        target = (root / record["path"]).resolve()
        if target.exists():
            payload = target.read_bytes()
        else:
            response = requests.get(record["url"], timeout=(15, 60))
            response.raise_for_status()
            if "login" in response.url or "credential" in response.url:
                raise ValueError("Authenticated content not permitted")
            payload = response.content
        digest = sha256(payload)
        if record.get("sha256") and record["sha256"] != digest:
            raise ValueError("Immutable confirmation file differs from manifest")
        structural = validate_psv_metadata(payload) if record["kind"] == "episode" else {}
        with lock:
            if total + len(payload) > MAX_TOTAL_BYTES:
                raise ValueError("Confirmation payload exceeds acquisition budget")
            total += len(payload)
            if not target.exists():
                target.write_bytes(payload)
        return {**record, "status": "downloaded_structure_verified", "bytes": len(payload),
                "sha256": digest, **structural}

    completed, failures = {}, []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(one, record): record for record in manifest["files"]}
        for future in as_completed(futures):
            record = futures[future]
            try:
                completed[record["path"]] = future.result()
            except Exception as exc:
                failures.append({"path": record["path"], "error_type": type(exc).__name__})
    manifest["files"] = [completed.get(record["path"], record) for record in manifest["files"]]
    manifest["failures_last_run"] = failures
    manifest["payload_bytes"] = total
    manifest["status"] = "complete_structural_validation_only" if not failures else "partial_acquisition_retry_frozen_selection"
    manifest["last_acquisition_at_utc"] = datetime.now(timezone.utc).isoformat()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    manifest = freeze_inventory()
    if not args.freeze_only:
        manifest = acquire(manifest)
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "selected_episodes": manifest["selected_episodes"],
                      "selection_sha256": manifest["selection_sha256"],
                      "payload_bytes": manifest.get("payload_bytes"),
                      "failures": len(manifest.get("failures_last_run", [])),
                      "outcomes_inspected": False, "models_evaluated": False}))
    if manifest.get("failures_last_run"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
