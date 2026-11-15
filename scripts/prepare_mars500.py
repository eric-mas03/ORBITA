"""Audit longitudinal Mars500 microbiome counts; no clinical-data merging."""
import json
import re
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def time_fields(label):
    if re.fullmatch(r"\d+ days?", label):
        day = int(label.split()[0])
        return day, float(day), "isolation"
    if label == "minus10 days":
        return -10, -10., "before"
    if label == "520 days plus 10 days":
        return 530, 530., "after"
    if label in {"520 days plus 3 month", "520 days plus 6 months"}:
        # Ordering is known; elapsed days are not. The ordinal here is used
        # only to sort these two visits after the day-530 observation.
        return (1_000 if "3 month" in label else 2_000), np.nan, "after"
    raise ValueError(f"Unknown time label: {label!r}")


def main():
    raw = ROOT / "data/raw/OSD-191"
    output = ROOT / "data/processed/mars500"
    report = ROOT / "outputs/data_expansion"
    output.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(next(raw.glob("*ISA.zip"))) as archive:
        metadata = pd.read_csv(archive.open("s_OSD-191.txt"), sep="\t")
    taxonomy = pd.read_csv(raw / "GLDS-191_GAmplicon_taxonomy-and-counts.tsv", sep="\t")
    tax_columns = ["ASV_ID", "domain", "phylum", "class", "order", "family", "genus", "species"]
    samples = [c for c in taxonomy if c not in tax_columns]
    if metadata["Sample Name"].duplicated().any() or taxonomy.ASV_ID.duplicated().any():
        raise ValueError("Duplicated sample or ASV identity")
    if set(samples) != set(metadata["Sample Name"]):
        raise ValueError("Count matrix and ISA sample identities do not agree")
    values = taxonomy[samples].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, values.round()).all():
        raise ValueError("Unexpected missing, negative or noninteger sequencing counts")
    meta = pd.DataFrame({"sample_id": metadata["Sample Name"],
                         "subject_id": "MARS500_" + metadata["Characteristics[Subject]"],
                         "cohort_id": "MARS500_520day", "domain": "space_analog_isolation",
                         "source_study": "OSD-191", "time_label": metadata["Factor Value[Time]"]})
    parsed = meta.time_label.map(time_fields)
    meta["sort_key"] = parsed.map(lambda x: x[0])
    meta["day_from_isolation_start"] = parsed.map(lambda x: x[1])
    meta["phase"] = parsed.map(lambda x: x[2])
    meta = meta.sort_values(["subject_id", "sort_key"])
    meta["visit_order"] = meta.groupby("subject_id").cumcount()
    meta["available_time_known"] = False
    meta = meta.drop(columns="sort_key").set_index("sample_id")
    counts = taxonomy.set_index("ASV_ID")[samples].T.reindex(meta.index)
    counts.columns.name = None
    total = counts.sum(axis=1)
    if total.le(0).any():
        raise ValueError("A sample has zero sequencing depth")
    meta["sequencing_reads"] = total
    meta.to_csv(output / "sample_metadata.csv")
    counts.to_csv(output / "asv_counts.csv")
    taxonomy[tax_columns].to_csv(output / "taxonomy.csv", index=False)
    phylum = taxonomy.phylum.fillna("Unclassified").astype(str)
    aggregated = taxonomy[samples].groupby(phylum).sum().T.reindex(meta.index)
    np.testing.assert_array_equal(aggregated.sum(axis=1).to_numpy(), total.to_numpy())
    relative = aggregated.div(total, axis=0)
    np.testing.assert_allclose(relative.sum(axis=1), 1.)
    aggregated.to_csv(output / "phylum_counts.csv")
    relative.to_csv(output / "phylum_relative_abundance.csv")
    summary = {"dataset": "OSD-191 Mars500", "domain": "space_analog_isolation",
               "subjects": meta.subject_id.nunique(), "samples": len(meta), "asvs": len(taxonomy),
               "phyla_including_unclassified": aggregated.shape[1],
               "visits_per_subject": meta.groupby("subject_id").size().to_dict(),
               "samples_with_exact_relative_day": int(meta.day_from_isolation_start.notna().sum()),
               "unknown_exact_day_labels": meta.loc[meta.day_from_isolation_start.isna(), "time_label"].unique().tolist(),
               "read_depth_min": int(total.min()), "read_depth_max": int(total.max()),
               "asv_zero_fraction": float(counts.eq(0).to_numpy().mean()),
               "real_astronauts_added": 0, "analog_participants": 6,
               "cohort_overlap_warning": "OSD-140 concerns the same Mars500 cohort: do not count six new people again.",
               "training_status": "not_trained_not_merged_with_blood_panels",
               "constraints": ["Relative abundances are compositional; zeros are sampling counts, not missing assays.",
                               "No arbitrary 30-day conversion of month labels.",
                               "No batch/subject splits or filtering learned yet.",
                               "No health outcome is inferred from sequencing abundance."]}
    (report / "mars500_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(10, 4))
    subjects = meta.subject_id.unique().tolist()
    for i, subject in enumerate(subjects):
        sub = meta[meta.subject_id.eq(subject)]
        ax.scatter(sub.day_from_isolation_start, np.full(len(sub), i), s=14)
    ax.set_yticks(range(len(subjects)), subjects)
    ax.set(xlabel="Días desde inicio del aislamiento", title="Mars500 · muestras con día relativo exacto\nLas visitas +3/+6 meses se conservan aparte, sin inventar días")
    ax.axvspan(0, 520, color="gray", alpha=.08)
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(report / "mars500_timeline.png", dpi=150)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
