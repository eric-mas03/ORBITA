"""Expand terrestrial training; freeze disjoint meta/final IDs before download.

Official challenge resources are documentary, not these ICU observations.
Final files receive structural checks only and are not packed here.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import shutil
import sys
import threading
import time

import requests
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location("confirmation_acquisition", ROOT / "scripts/acquire_confirmation_holdout.py")
previous = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(previous)
RAW = ROOT / "data/raw/external/PHYSIONET_EXPANSION_V2"
MANIFEST = ROOT / "data/manifests/hourly_expansion_v2.json"
OLD = ROOT / "data/raw/external/PHYSIONET_SEPSIS_2019_1.0.0"
SEED = 2026111402


def sha(body):
    return hashlib.sha256(body).hexdigest()


def freeze_inventory():
    excluded = {p.name for directory in (OLD, ROOT / "data/raw/external/PHYSIONET_CONFIRMATION")
                for p in directory.rglob("*.psv")}
    records = []
    indexes = {}
    for site in previous.SITES:
        body = (OLD / f"{site}_index.html").read_bytes()
        indexes[site] = sha(body)
        names = sorted(set(previous.public_names(body)) - excluded)
        random.Random(f"{SEED}|{site}").shuffle(names)
        names = names[:750]
        if len(names) != 750:
            raise ValueError("Insufficient unused published episodes")
        for index, name in enumerate(names):
            role = "train" if index < 500 else "meta" if index < 600 else "final"
            records.append(dict(episode_id=f"{site}/{Path(name).stem}", role=role, site=site,
                path=(RAW / site / name).relative_to(ROOT).as_posix(),
                url=previous.BASE + "training/" + site + "/" + name, status="planned"))
        excluded.update(names)
    if MANIFEST.exists():
        saved = json.loads(MANIFEST.read_text())
        identity = lambda record: tuple(record[key] for key in ("episode_id", "role", "site", "path", "url"))
        if (saved["seed"] != SEED or saved["index_sha256"] != indexes or
                sorted(map(identity, saved["files"])) != sorted(map(identity, records))):
            raise ValueError("Frozen expansion inventory changed or overlaps used episodes")
        return saved
    manifest = dict(version=2, seed=SEED, status="roles_frozen_before_acquisition", files=records,
        frozen_at_utc=datetime.now(timezone.utc).isoformat(), source="PhysioNet Challenge 2019 1.0.0",
        domain="terrestrial ICU; additional source, not a dataset supplied by NASA challenge resources",
        index_sha256=indexes, new_role_counts={"train":1000,"meta":200,"final":300},
        old_test_and_confirmation_excluded=True, maximum_payload_bytes=40_000_000,
        final_values_inspected=False, meta_disjoint_from_component_fit=True,
        license_note="Preserve source CC-BY-4.0 landing declaration and bundled ODbL license without resolving their discrepancy here.")
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def acquire(manifest, verify=True):
    lock = threading.Lock()
    downloaded = 0
    total = 0

    def one(record):
        nonlocal total, downloaded
        path = (ROOT / record["path"]).resolve()
        if not path.is_relative_to(RAW.resolve()):
            raise ValueError("Unsafe expansion path")
        body = None
        if path.exists():
            body = path.read_bytes()
            if record.get("sha256") and sha(body) != record["sha256"]:
                raise ValueError("Cached expansion changed")
        for attempt in range(3):
            if body is not None:
                break
            try:
                response = requests.get(record["url"], timeout=(15, 60), verify=verify)
                response.raise_for_status()
                if "login" in response.url or len(response.content) > 2_000_000:
                    raise ValueError("Unexpected public episode response")
                body = response.content
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        structure = previous.validate_psv_metadata(body)
        with lock:
            total += len(body)
            if total > manifest["maximum_payload_bytes"]:
                raise ValueError("Acquisition payload budget exceeded")
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(body)
            record.update(status="acquired_structurally_verified", bytes=len(body), sha256=sha(body), structure=structure)
            downloaded += 1
            if downloaded % 50 == 0:
                MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                print(f"Acquired {downloaded}/{len(manifest['files'])}; bytes={total}", flush=True)

    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(one, record) for record in manifest["files"]]):
            future.result()
    # Reuse the exact original license artifact; never submit an agreement.
    license_paths = list(OLD.glob("*LICENSE*"))
    if license_paths:
        (RAW / license_paths[0].name).write_bytes(license_paths[0].read_bytes())
    manifest.update(status="acquired_structural_validation_complete", bytes=total,
                    tls_certificate_verification=verify, acquired_at_utc=datetime.now(timezone.utc).isoformat())
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def pack_nonfinal(manifest):
    from astronaut_twin.hourly_data import build_packed
    destination = ROOT / "data/processed/hourly_v2"
    destination.mkdir(parents=True, exist_ok=True)
    staging = destination / "nonfinal_sources"
    old_splits = json.loads((ROOT / "data/processed/hourly/splits.json").read_text())["episode_split"]
    roles = {}
    for path in sorted(OLD.rglob("*.psv")):
        identity = f"{path.parent.name}/{path.stem}"
        role = old_splits[identity]
        if role not in {"train", "dev", "calibration"}:
            continue
        target = staging / path.parent.name / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != path.read_bytes():
            raise ValueError("Staged original data changed")
        if not target.exists():
            shutil.copyfile(path, target)
        roles[identity] = role
    for record in manifest["files"]:
        if record["role"] == "final":
            continue
        path = ROOT / record["path"]
        target = staging / record["site"] / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(path, target)
        if sha(target.read_bytes()) != record["sha256"]:
            raise ValueError("Staged new data changed")
        roles[record["episode_id"]] = record["role"]
    packed_path = destination / "hourly_nonfinal_dataset.npz"
    summary = build_packed(staging, destination / "temporary_builder.npz", destination / "temporary_builder_splits.json")
    with np.load(destination / "temporary_builder.npz", allow_pickle=False) as source:
        packed = {key: source[key] for key in source.files}
    # Override builder balancing labels before any features, statistics or fit.
    packed["episode_split"] = np.asarray([roles[str(name)] for name in packed["episode_ids"]])
    if set(packed["episode_ids"]) != set(roles) or "final" in set(packed["episode_split"]):
        raise ValueError("Nonfinal inventory mismatch")
    np.savez_compressed(packed_path, **packed)
    counts = {role: int((packed["episode_split"] == role).sum()) for role in sorted(set(roles.values()))}
    samples = packed["episode_split"][packed["sample_episode"]]
    audit = dict(status="audited_nonfinal_v2", dataset=str(packed_path.relative_to(ROOT)),
        dataset_sha256=sha(packed_path.read_bytes()), episode_role_counts=counts,
        sample_role_counts={role: int((samples == role).sum()) for role in counts},
        source_manifest_sha256=sha(MANIFEST.read_bytes()), final_not_packed=True,
        excluded_old_test_episodes=150, excluded_previous_confirmation_episodes=300,
        final_episode_ids=[r["episode_id"] for r in manifest["files"] if r["role"] == "final"],
        final_directory=str(RAW.relative_to(ROOT)), quality_flags=summary["quality_flags"],
        variables=list(map(str, packed["variables"])), horizons=list(map(int, packed["horizons"])),
        context_hours=24, source_resolution_hours=1, domain="terrestrial ICU")
    (destination / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (destination / "roles.json").write_text(json.dumps(roles, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in audit.items() if k not in {"final_episode_ids", "quality_flags", "variables"}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tls-verify", action="store_true", help="Public files only; record environment TLS workaround")
    args = parser.parse_args()
    inventory = freeze_inventory()
    acquire(inventory, verify=not args.skip_tls_verify)
    pack_nonfinal(inventory)
