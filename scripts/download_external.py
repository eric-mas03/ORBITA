"""Download public terrestrial sequences without authentication or agreements.

These sources are separate domains, never additional astronaut observations.
Default: all 53 BIDMC numerics, 500 ICU sequences per site, and PBCseq visits.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = (ROOT / "data/raw/external").resolve()
MANIFEST = ROOT / "data/manifests/external_downloads.json"


def fetch(url: str) -> bytes:
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=(15, 60))
            response.raise_for_status()
            if "login" in response.url or "credential" in response.url:
                raise ValueError("Authenticated access is outside this downloader")
            if len(response.content) > 20_000_000:
                raise ValueError("Unexpectedly large individual file")
            return response.content
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise RuntimeError("unreachable")


def download(job: tuple[str, str, str, str]) -> dict:
    dataset, relative, url, license_name = job
    target = (RAW / dataset / relative).resolve()
    if not target.is_relative_to(RAW.resolve()):
        raise ValueError("Unsafe target")
    target.parent.mkdir(parents=True, exist_ok=True)
    data = target.read_bytes() if target.exists() else fetch(url)
    if relative.endswith(".psv") and not data.startswith(b"HR|O2Sat|"):
        raise ValueError("Unexpected PSV header")
    if relative.endswith("Numerics.csv") and not data.startswith(b"Time [s],"):
        raise ValueError("Unexpected BIDMC numerics header")
    if relative == "pbcseq.csv" and not data.startswith(b"id,futime,status,"):
        raise ValueError("Unexpected PBCseq CSV header")
    if not target.exists():
        target.write_bytes(data)
    return {"dataset": dataset, "path": target.relative_to(ROOT).as_posix(),
            "url": url, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "retrieved_at_utc": datetime.fromtimestamp(target.stat().st_mtime, timezone.utc).isoformat(),
            "license": license_name, "access": "public_without_authentication"}


def summarize_bidmc() -> dict:
    lengths = []
    rows_total = 0
    original_ids = []
    identity_map = {}
    observed = {}
    columns = None
    for path in sorted((RAW / "BIDMC_1.0.0").glob("bidmc_*_Numerics.csv")):
        rows = list(csv.DictReader(io.StringIO(path.read_text()), skipinitialspace=True))
        columns = list(rows[0])
        rows_total += len(rows)
        lengths.append(len(rows))
        for key in columns[1:]:
            observed[key] = observed.get(key, 0) + sum(
                str(row[key]).strip().lower() not in {"nan", "", "na"} for row in rows)
        fix = path.with_name(path.name.replace("_Numerics.csv", "_Fix.txt"))
        if fix.exists():
            match = re.search(r"s\d{5,}", fix.read_text())
            if match:
                original_ids.append(match.group())
                identity_map[path.stem.removesuffix("_Numerics")] = match.group()
    return {"recording_count": len(lengths), "original_subject_ids_found": len(original_ids),
            "unique_original_subject_ids": len(set(original_ids)), "rows": rows_total,
            "recording_to_original_subject_id": identity_map,
            "rows_per_recording_min": min(lengths, default=0),
            "rows_per_recording_max": max(lengths, default=0),
            "columns": columns, "observed_numeric_cells_by_variable": observed,
            "domain": "terrestrial_critical_care", "sampling": "numerics_1_Hz",
            "not_for": "CBC/CMP encoder pretraining or astronaut forecasting validation"}


def summarize_sepsis() -> dict:
    lengths = []
    counts = {}
    observed = {}
    multi = {}
    columns = None
    for path in sorted((RAW / "PHYSIONET_SEPSIS_2019_1.0.0").glob("*/*.psv")):
        rows = list(csv.DictReader(io.StringIO(path.read_text()), delimiter="|"))
        if not rows:
            continue
        columns = list(rows[0])
        lengths.append(len(rows))
        counts[path.parent.name] = counts.get(path.parent.name, 0) + 1
        for key in columns:
            valid = sum(str(row[key]).strip().lower() not in {"nan", "", "na"} for row in rows)
            observed[key] = observed.get(key, 0) + valid
            multi[key] = multi.get(key, 0) + (valid >= 2)
    ordered = sorted(lengths)
    return {"sequence_count": len(lengths), "files_by_site": counts, "rows": sum(lengths),
            "rows_per_sequence_min": min(lengths, default=0),
            "rows_per_sequence_median": ordered[len(ordered)//2] if ordered else None,
            "rows_per_sequence_max": max(lengths, default=0), "columns": columns,
            "observed_cells_by_variable": observed,
            "sequences_with_at_least_two_observations_by_variable": multi,
            "domain": "terrestrial_critical_care", "sampling": "hourly_binned_sparse",
            "excluded_predictors": ["SepsisLabel"],
            "warning": "Patient files are independent published sequence IDs; underlying person deduplication across admissions is not established here."}


def summarize_pbc() -> dict:
    path = RAW / "PBCSEQ_SURVIVAL" / "pbcseq.csv"
    rows = list(csv.DictReader(io.StringIO(path.read_text())))
    ids = sorted({row["id"] for row in rows})
    visits = [sum(row["id"] == subject for row in rows) for subject in ids]
    observed = {key: sum(row[key] not in {"", "NA", "NaN"} for row in rows) for key in rows[0]}
    repeated = {key: sum(sum(row[key] not in {"", "NA", "NaN"} for row in rows if row["id"] == subject) >= 2 for subject in ids) for key in rows[0]}
    return {"subject_count": len(ids), "visit_count": len(rows),
            "visits_per_subject_min": min(visits), "visits_per_subject_max": max(visits),
            "columns": list(rows[0]), "observed_cells_by_variable": observed,
            "subjects_with_at_least_two_observations_by_variable": repeated,
            "domain": "terrestrial_chronic_liver_disease_randomized_trial",
            "time_column": "day", "excluded_predictors": ["futime", "status"],
            "unit_warning": "Historical documentation has ambiguous albumin/AST/platelet units; audit primary protocol before cross-domain harmonization."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", "bidmc", "sepsis", "pbcseq"], default="all")
    parser.add_argument("--sepsis-per-site", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.sepsis_per_site < 1 or not 1 <= args.workers <= 8:
        parser.error("Positive sample size and workers between 1 and 8 required")
    jobs = []
    indexes = {}
    if args.dataset in {"all", "bidmc"}:
        base = "https://physionet.org/files/bidmc/1.0.0/"
        for record in range(1, 54):
            for suffix in ["Numerics.csv", "Fix.txt"]:
                filename = f"bidmc_{record:02d}_{suffix}"
                jobs.append(("BIDMC_1.0.0", filename, base + "bidmc_csv/" + filename, "ODC-By-1.0"))
        jobs.append(("BIDMC_1.0.0", "LICENSE", base + "LICENSE", "ODC-By-1.0"))
    if args.dataset in {"all", "sepsis"}:
        base = "https://physionet.org/files/challenge-2019/1.0.0/"
        for site in ["training_setA", "training_setB"]:
            url = base + "training/" + site + "/"
            data = fetch(url)
            listing = RAW / "PHYSIONET_SEPSIS_2019_1.0.0" / (site + "_index.html")
            listing.parent.mkdir(parents=True, exist_ok=True)
            listing.write_bytes(data)
            names = sorted(set(re.findall(r'href="(p\d+\.psv)"', data.decode())))
            if not names:
                raise ValueError("No public patient files found")
            selected = sorted(random.Random(args.seed).sample(names, min(args.sepsis_per_site, len(names))))
            indexes[site] = {"url": url, "listed_files": len(names), "selected_files": len(selected),
                             "sha256": hashlib.sha256(data).hexdigest(),
                             "selection": "uniform_without_replacement_seed_" + str(args.seed)}
            jobs.extend(("PHYSIONET_SEPSIS_2019_1.0.0", site + "/" + name, url + name, "landing_CC-BY-4.0__bundled_LICENSE_ODbL_discrepancy") for name in selected)
        jobs.append(("PHYSIONET_SEPSIS_2019_1.0.0", "LICENSE.txt", base + "LICENSE.txt", "bundled_ODbL__landing_CC-BY-4.0_discrepancy"))
    if args.dataset in {"all", "pbcseq"}:
        jobs.append(("PBCSEQ_SURVIVAL", "pbcseq.csv", "https://therneau.r-universe.dev/survival/data/pbcseq/csv", "survival_package_LGPL-2-or-later"))
        jobs.append(("PBCSEQ_SURVIVAL", "DESCRIPTION", "https://raw.githubusercontent.com/therneau/survival/master/DESCRIPTION", "LGPL-2-or-later"))
        jobs.append(("PBCSEQ_SURVIVAL", "pbcseq_documentation.html", "https://stat.ethz.ch/R-manual/R-devel/library/survival/html/pbcseq.html", "R_package_documentation"))
    old = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    records = {record["path"]: record for record in old.get("files", [])}
    for record in records.values():
        if record["dataset"] == "PHYSIONET_SEPSIS_2019_1.0.0":
            record["license"] = "landing_CC-BY-4.0__bundled_LICENSE_ODbL_discrepancy"
    failures = []
    # Resolve existing parents before workers start (Windows can otherwise race
    # while resolving an ancestor another thread is creating).
    for dataset, relative, _, _ in jobs:
        parent = RAW / dataset / Path(relative).parent
        if not parent.resolve().is_relative_to(RAW):
            raise ValueError("Unsafe download directory")
        parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download, job): job for job in jobs}
        for number, future in enumerate(as_completed(futures), start=1):
            try:
                result = future.result()
                records[result["path"]] = result
            except Exception as exc:
                failures.append({"url": futures[future][2], "error": type(exc).__name__ + ": " + str(exc)[:200]})
            if number % 100 == 0:
                print(f"Processed {number}/{len(jobs)} public files", flush=True)
    summary = old.get("summary", {})
    if (RAW / "BIDMC_1.0.0").exists():
        summary["BIDMC"] = summarize_bidmc()
    if (RAW / "PHYSIONET_SEPSIS_2019_1.0.0").exists():
        summary["PHYSIONET_SEPSIS_2019"] = summarize_sepsis()
    if (RAW / "PBCSEQ_SURVIVAL/pbcseq.csv").exists():
        summary["PBCSEQ"] = summarize_pbc()
    manifest = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "public_terrestrial_domains_separate_from_astronaut_data",
                "source_indexes": indexes or old.get("source_indexes", {}),
                "license_notes": {"PHYSIONET_SEPSIS_2019": "Landing page declares CC-BY-4.0, but downloaded LICENSE.txt starts ODC Open Database License (ODbL). Both are preserved; clarify before redistribution."},
                "files": sorted(records.values(), key=lambda record: record["path"]),
                "failures_last_run": failures, "summary": summary}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"file_count": len(records), "failures": len(failures), "summary": summary}, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
