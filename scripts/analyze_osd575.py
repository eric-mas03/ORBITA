"""Reproducible descriptive EDA on the downloaded OSD-575 tables.

No pooled p-values: 28 visits are repeated observations of only four people.
No interpolation, percentile-derived columns, or duplicated transformed assays.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["subject_id", "timepoint", "assay", "analyte"]


def parse_time(label: str) -> tuple[str, int]:
    match = re.fullmatch(r"([LR])([+-]\d+)", label.strip())
    if not match:
        raise ValueError(f"Unsupported timepoint: {label!r}")
    return ("launch" if match[1] == "L" else "return", int(match[2]))


def load_observations(raw: Path, cbc_raw: Path | None = None) -> pd.DataFrame:
    tables = []
    definitions = [
        ("CMP", "LSDS-8_Comprehensive_Metabolic_Panel_CMP.upload_SUBMITTED.csv",
         {"ANALYTE": "analyte", "VALUE": "value", "UNITS": "unit", "SUBJECT_ID": "subject_id", "timepoint": "timepoint"}),
        ("EVE_CARDIO", "LSDS-8_Multiplex_serum.cardiovascular.EvePanel_SUBMITTED.csv",
         {"Analyte": "analyte", "Concentration": "value", "Unit": "unit", "ID": "subject_id", "Timepoint": "timepoint"}),
        ("EVE_IMMUNE", "LSDS-8_Multiplex_serum.immune.EvePanel_SUBMITTED.csv",
         {"Analyte": "analyte", "Concentration": "value", "Unit": "unit", "ID": "subject_id", "Timepoint": "timepoint"}),
    ]
    for assay, filename, mapping in definitions:
        original = pd.read_csv(raw / filename)
        frame = original[list(mapping)].rename(columns=mapping)
        frame["source_row"] = np.arange(len(frame)) + 2
        frame["assay"] = assay
        frame["source_file"] = filename
        tables.append(frame)
    filename = "LSDS-8_Multiplex_serum.immune.AlamarPanel_TRANSFORMED.csv"
    original = pd.read_csv(raw / filename)
    measures = [c for c in original if c.endswith("_concentration_npq")]
    for col in measures:
        frame = pd.DataFrame({
            "subject_id": original["Sample ID"].str.extract(r"^(C\d+)_", expand=False),
            "timepoint": original["Sample ID"].str.extract(r"_([LR][+-]\d+)$", expand=False),
            "analyte": col.removesuffix("_concentration_npq"),
            "value": original[col], "unit": "NPQ (log2)", "assay": "ALAMAR",
            "source_file": filename, "source_row": np.arange(len(original)) + 2,
        })
        tables.append(frame)
    if cbc_raw is not None:
        filename = "LSDS-7_Complete_Blood_Count_CBC.upload_SUBMITTED.csv"
        original = pd.read_csv(cbc_raw / filename)
        frame = original[["ANALYTE", "VALUE", "UNITS", "SUBJECT_ID", "TEST_DATE"]].rename(
            columns={"ANALYTE": "analyte", "VALUE": "value", "UNITS": "unit", "SUBJECT_ID": "subject_id", "TEST_DATE": "timepoint"})
        frame["original_analyte"] = frame.analyte
        frame["analyte"] = frame.analyte.str.replace(r" \(FEMALE\)$", "", regex=True)
        frame["source_row"] = np.arange(len(frame)) + 2
        frame["assay"] = "CBC"
        frame["source_file"] = filename
        tables.append(frame)
    data = pd.concat(tables, ignore_index=True)
    data["analyte"] = data["analyte"].str.strip()
    data["unit"] = data["unit"].str.strip()
    data["raw_value"] = data["value"].astype(str)
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    if data[KEYS].isna().any().any():
        raise ValueError("Missing identity/time/assay/analyte")
    if data.duplicated(KEYS).any():
        raise ValueError("Duplicate observation keys: inspect, do not average silently")
    data["variable"] = data["assay"] + "::" + data["analyte"]
    if data.groupby("variable")["unit"].nunique().gt(1).any():
        raise ValueError("Multiple units in the same assay variable")
    relative = data.timepoint.map(parse_time)
    data["time_reference"] = relative.map(lambda x: x[0])
    data["relative_day"] = relative.map(lambda x: x[1])
    # An ordering index is not an exact elapsed duration across launch/return.
    visits = sorted(data.timepoint.unique(), key=lambda t: (t.startswith("R"), parse_time(t)[1]))
    data["visit_index"] = data.timepoint.map({t: i for i, t in enumerate(visits)})
    data["available_time_known"] = False
    data["quality_flag"] = np.where(data.value.isna(), "missing", "numeric_as_published")
    return data.sort_values(["subject_id", "visit_index", "variable"]).reset_index(drop=True)


def pair_correlations(wide: pd.DataFrame, assay_map: dict[str, str], min_pairs: int = 12) -> pd.DataFrame:
    """Pooled Spearman and pairwise within-person marginal-rank correlation.

    The latter is a descriptive repeated-measure adjustment, not a causal or
    independent-sample inference. Pairwise counts and between-person variation
    differ; no hypothesis-testing claims are made.
    """
    ranks = wide.rank()
    rho = wide.corr(method="spearman", min_periods=min_pairs).to_numpy()
    present = wide.notna().to_numpy(dtype=float)
    counts = present.T @ present
    m = wide.shape[1]
    covariance = np.zeros((m, m))
    variance = np.zeros((m, m))
    subjects = np.zeros((m, m), dtype=int)
    per = []
    for subject in wide.index.get_level_values("subject_id").unique():
        ranked = ranks.xs(subject, level="subject_id").to_numpy()
        indicator = np.isfinite(ranked).astype(float)
        values = np.nan_to_num(ranked)
        n = indicator.T @ indicator
        sums = values.T @ indicator
        with np.errstate(divide="ignore", invalid="ignore"):
            centered_cross = values.T @ values - np.divide(sums * sums.T, n, out=np.zeros_like(n), where=n > 0)
            centered_sq = (values ** 2).T @ indicator - np.divide(sums ** 2, n, out=np.zeros_like(n), where=n > 0)
        covariance += centered_cross
        variance += centered_sq
        subjects += n > 0
        per.append(wide.xs(subject, level="subject_id").corr(method="spearman", min_periods=4).to_numpy())
    with np.errstate(divide="ignore", invalid="ignore"):
        within = covariance / np.sqrt(np.maximum(variance, 0) * np.maximum(variance.T, 0))
    per = np.asarray(per)
    finite = np.isfinite(per)
    minimum = np.where(finite, per, np.inf).min(axis=0)
    maximum = np.where(finite, per, -np.inf).max(axis=0)
    minimum[~finite.any(axis=0)] = np.nan
    maximum[~finite.any(axis=0)] = np.nan
    i, j = np.triu_indices(m, 1)
    keep = (counts[i, j] >= min_pairs) & (subjects[i, j] >= 3) & np.isfinite(rho[i, j])
    i, j = i[keep], j[keep]
    cols = np.asarray(wide.columns)
    return pd.DataFrame({"variable_a": cols[i], "variable_b": cols[j],
                         "n_pairs": counts[i, j].astype(int), "n_subjects": subjects[i, j],
                         "spearman_pooled": rho[i, j], "within_person_rank_centered": within[i, j],
                         "subject_rho_min": minimum[i, j], "subject_rho_max": maximum[i, j],
                         "cross_assay": [assay_map[a] != assay_map[b] for a, b in zip(cols[i], cols[j])]})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/OSD-575")
    args = parser.parse_args()
    out = ROOT / "outputs/eda"
    processed = ROOT / "data/processed"
    out.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    data = load_observations(args.raw, ROOT / "data/raw/OSD-569")
    data.to_csv(processed / "inspiration4_observations.csv", index=False)
    wide = data.pivot(index=["subject_id", "visit_index", "timepoint"], columns="variable", values="value").sort_index()
    wide.to_csv(processed / "inspiration4_wide.csv")
    n_visits = len(wide)
    dictionary = data.groupby("variable").agg(
        assay=("assay", "first"), analyte=("analyte", "first"), unit=("unit", "first"),
        n_rows=("value", "size"), n_observed=("value", "count"), n_subjects=("subject_id", "nunique"),
        n_unique=("value", "nunique"), minimum=("value", "min"), maximum=("value", "max"),
        median=("value", "median"), source_file=("source_file", "first"),
    )
    dictionary["missing_fraction_of_28_visits"] = 1 - dictionary.n_observed / n_visits
    dictionary["zero_fraction_observed"] = data.assign(zero=data.value.eq(0) & data.value.notna()).groupby("variable").zero.sum() / dictionary.n_observed
    dictionary.to_csv(out / "variable_dictionary.csv")
    coverage = data.groupby(["assay", "subject_id", "timepoint", "visit_index"]).agg(
        n_present=("value", "count"), n_rows=("value", "size"))
    coverage.to_csv(out / "coverage.csv")
    mapping = dictionary.assay.to_dict()
    pairs = pair_correlations(wide, mapping)
    pairs.sort_values("within_person_rank_centered", key=abs, ascending=False).to_csv(out / "correlations.csv", index=False)
    assay_counts = dictionary.groupby("assay").agg(variables=("n_rows", "size"), observed=("n_observed", "sum"))
    samples = data[["assay", "subject_id", "timepoint", "visit_index"]].drop_duplicates()
    assay_counts["subject_visits"] = samples.groupby("assay").size()
    summary = {
        "studies": ["OSD-575", "OSD-569"], "subjects": sorted(data.subject_id.unique().tolist()),
        "subject_visits": n_visits, "timepoints": data.sort_values("visit_index").timepoint.unique().tolist(),
        "variables": len(dictionary), "rows": len(data), "observed_values": int(data.value.notna().sum()),
        "assays": assay_counts.reset_index().to_dict(orient="records"),
        "missing_fraction": float(wide.isna().to_numpy().mean()),
        "unobserved_alamar_visits": [list(k) for k in wide.index[wide.filter(like="ALAMAR::").isna().all(axis=1)]],
        "warnings": ["Four people, repeated visits: no independent-sample p-values.",
                     "No in-flight samples in these tables.",
                     "Launch-relative and return-relative days kept separate; no invented elapsed days.",
                     "Result availability times unknown: retrospective analysis only.",
                     "Published assay normalization/batch effects may be non-prospective.",
                     "EDA uses all data descriptively; later model evaluation is exploratory, not untouched confirmation."]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Readable figures: full 302x302 table is delivered as CSV, not a tiny image.
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    cmp = wide.filter(like="CMP::")
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(cmp.corr(method="spearman"), vmin=-1, vmax=1, cmap="RdBu_r")
    labels = [c.split("::", 1)[1] for c in cmp]
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_title("CMP · correlación Spearman descriptiva\n4 personas, 28 visitas; incluye variables derivadas")
    fig.colorbar(image, ax=ax, shrink=.7, label="rho")
    fig.tight_layout()
    fig.savefig(out / "cmp_correlations.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5))
    order = ["CMP", "CBC", "EVE_CARDIO", "EVE_IMMUNE", "ALAMAR"]
    completeness = pd.DataFrame({a: wide.filter(like=a + "::").notna().mean(axis=1) for a in order})
    im = ax.imshow(completeness.T, aspect="auto", vmin=0, vmax=1, cmap="Blues")
    ax.set_yticks(range(len(order)), order)
    ax.set_xticks(range(n_visits), [f"{s}\n{t}" for s, _, t in wide.index], rotation=90, fontsize=8)
    ax.set_title("Cobertura por ensayo y visita · fracción de analitos presentes")
    fig.colorbar(im, ax=ax, label="Fracción presente")
    fig.tight_layout()
    fig.savefig(out / "coverage.png", dpi=150)
    plt.close(fig)
    chosen = ["CMP::GLUCOSE", "CMP::CREATININE", "CMP::SODIUM", "EVE_CARDIO::CRP"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, variable in zip(axes.flat, chosen):
        subset = data[data.variable.eq(variable)]
        for subject, sub in subset.groupby("subject_id"):
            ax.plot(sub.visit_index, sub.value, marker="o", label=subject)
        ax.set_title(variable)
        ax.set_ylabel(subset.unit.iloc[0])
        ax.set_xticks(range(7), summary["timepoints"], rotation=30)
        ax.grid(alpha=.2)
    axes[0, 0].legend(ncol=2)
    fig.suptitle("Trayectorias observadas · eje por visita, no tiempo uniforme")
    fig.tight_layout()
    fig.savefig(out / "trajectories.png", dpi=150)
    plt.close(fig)
    # Independent intervals only within one time reference.
    gaps = samples.copy()
    gaps["reference"] = gaps.timepoint.map(lambda t: parse_time(t)[0])
    gaps["day"] = gaps.timepoint.map(lambda t: parse_time(t)[1])
    gaps = gaps.sort_values(["assay", "subject_id", "reference", "day"])
    gaps["gap_days_same_reference"] = gaps.groupby(["assay", "subject_id", "reference"]).day.diff()
    gaps.to_csv(out / "visit_intervals.csv", index=False)
    # Report avoids statistical significance and physiological judgement.
    top = pairs[pairs.cross_assay & pairs.within_person_rank_centered.notna()].copy()
    top = top.reindex(top.within_person_rank_centered.abs().sort_values(ascending=False).index).head(10)
    lines = ["# Análisis exploratorio real de Inspiration4: OSD-575 y OSD-569", "",
             "Generado por `python scripts/analyze_osd575.py`. Datos publicados, sin imputación ni interpolación.", "",
             f"Se han analizado **{len(summary['subjects'])} personas, {n_visits} visitas-persona, {len(dictionary)} variables de ensayo y {summary['observed_values']} valores numéricos observados**.", "",
             "| Ensayo | Variables | Visitas-persona | Valores observados |", "|---|---:|---:|---:|"]
    for assay, row in assay_counts.iterrows():
        lines.append(f"| {assay} | {int(row.variables)} | {int(row.subject_visits)} | {int(row.observed)} |")
    lines += ["", "## Estructura temporal y calidad", "",
              "Visitas: " + ", ".join(summary["timepoints"]) + ". No hay observaciones durante el vuelo en estas tablas. Los intervalos son irregulares: L-92→L-44 son 48 días; L-44→L-3 son 41; R+1→R+45 son 44; R+45→R+82 son 37; R+82→R+194 son 112. El salto entre L y R requiere fechas exactas adicionales y no se fabrica.", "",
              f"La matriz conjunta tiene un {100*summary['missing_fraction']:.2f} % de celdas ausentes. Alamar no contiene C003 en L-44. BUN/CREATININE RATIO solo contiene 6 valores numéricos de 28 visitas; no se completa usando futuros valores. Las cifras y rangos publicados se conservan sin convertir desviaciones en diagnósticos.", "",
              "Se excluyen todas las columnas Percent, percent y percent_normalized_value, todos los rangos de referencia como entradas y las copias transformadas de ensayos ya incluidos. Alamar se conserva en NPQ log2 y nunca se mezcla con pg/ml como si fueran la misma medición. Valores cero se conservan; ausencia de flags de límite de detección no demuestra ausencia de censura.", "",
              "## Correlaciones exploratorias", "",
              "`correlations.csv` contiene Spearman agrupado (rangos recalculados por par), correlación de rangos marginales centrados por persona (centrado sobre pares completos), número de parejas, sujetos y rango de correlaciones individuales. Los rangos marginales se calculan con todas las observaciones disponibles de cada variable, por lo que la segunda métrica no es exactamente una correlación Spearman de pares completos. Este centrado elimina diferencias de nivel entre sujetos, pero no elimina tendencia de visita, confusión ni dependencia temporal. No se publican p-valores de independencia ni se presentan asociaciones como causales. Los valores máximos entre miles de pares sufren selección múltiple.", "",
              "Los siguientes pares entre ensayos son ejemplos extremos para inspección; no una selección de features ni evidencia confirmatoria:", "",
              "| Variable A | Variable B | Pares | rho agrupado | rho centrado por persona |", "|---|---|---:|---:|---:|"]
    for _, row in top.iterrows():
        lines.append(f"| {row.variable_a} | {row.variable_b} | {row.n_pairs} | {row.spearman_pooled:.3f} | {row.within_person_rank_centered:.3f} |")
    lines += ["", "## Implicación para el modelo", "",
              "Se puede implementar y verificar la arquitectura por encoders y cabezas temporales con estas observaciones. Cuatro personas y siete visitas no sostienen una validación convincente de un transformer de salud ni predicciones continuas por horas. La primera ejecución de entrenamiento será un experimento técnico de siguiente visita con horizonte irregular, comparado con persistencia y etiquetado como exploratorio.", "",
              "Los rangos históricos de eGFR por categoría racial son variables derivadas del archivo original; no se usarán como objetivos ni para personalización. Las razones derivadas tampoco serán el criterio de agrupamiento. El agrupamiento se realiza con criterio biomédico y ensayo, y la correlación sirve para revisarlo, sin reemplazar ese criterio.", "",
              "El tiempo de disponibilidad del resultado no consta: el experimento será retrospectivo bajo un supuesto explícito. Los datos del conjunto se han explorado completos, por lo que cualquier partición posterior será una comprobación preliminar, no una prueba confirmatoria intacta.", "",
              "El hemograma se toma del archivo original de OSD-569: HEMOGLOBIN usa g/dL, corrigiendo la etiqueta errónea percent de la copia transformada sin cambiar el valor. Se armonizan los sufijos (FEMALE) de eritrocitos, hemoglobina y hematocrito, preservando la etiqueta original. No se añaden sexo ni rangos como predictores.", "",
              "Fuentes: https://osdr.nasa.gov/bio/repo/data/studies/OSD-575 y https://osdr.nasa.gov/bio/repo/data/studies/OSD-569 . URLs, tamaños y SHA-256 en `data/manifests/osdr_downloads.json`."]
    (ROOT / "docs/eda-osd575.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
