"""Describe only training rows of the expanded cohort; never open final files."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT / "data/processed/hourly_v2"
    audit = json.loads((directory / "audit.json").read_text())
    source = ROOT / audit["dataset"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != audit["dataset_sha256"]:
        raise ValueError("Audited dataset changed")
    with np.load(source, allow_pickle=False) as archive:
        data = {key:archive[key] for key in archive.files}
    expansion = json.loads((ROOT / "data/manifests/hourly_expansion_v2.json").read_text())
    final = {r["episode_id"] for r in expansion["files"] if r["role"] == "final"}
    if set(data["episode_ids"]) & final:
        raise ValueError("Final episodes leaked into nonfinal preparation")
    old_splits = json.loads((ROOT / "data/processed/hourly/splits.json").read_text())["episode_split"]
    forbidden = {name for name,role in old_splits.items() if role == "test"}
    forbidden |= {f"{p.parent.name}/{p.stem}" for p in (ROOT / "data/raw/external/PHYSIONET_CONFIRMATION").rglob("*.psv")}
    if set(data["episode_ids"]) & forbidden:
        raise ValueError("Consumed previous test episodes leaked into V2")
    train_episodes = np.flatnonzero(data["episode_split"] == "train")
    train_rows = np.concatenate([np.arange(data["episode_offsets"][e],data["episode_offsets"][e+1]) for e in train_episodes])
    values = data["values"][train_rows]
    variables = list(map(str, data["variables"]))
    train_samples = np.flatnonzero(data["episode_split"][data["sample_episode"]] == "train")
    old = np.load(ROOT / "outputs/hourly_baselines/linear_quantile.npz",allow_pickle=False)["train_target_counts"]
    records = []
    for h,horizon in enumerate(data["horizons"]):
        target = data["target_rows"][train_samples,h]
        exists = target >= 0
        labels = np.full((len(train_samples),len(variables)),np.nan)
        labels[exists] = data["values"][target[exists]]
        for v,variable in enumerate(variables):
            valid = np.isfinite(labels[:,v])
            records.append(dict(variable=variable,horizon_hours=int(horizon),
                old_train_labels=int(old[h,v]), new_train_labels=int(valid.sum()),
                new_train_episodes=int(np.unique(data["sample_episode"][train_samples[valid]]).size)))
    out = ROOT / "outputs/data_audit_v2"
    out.mkdir(parents=True,exist_ok=True)
    table = pd.DataFrame(records)
    table.to_csv(out / "training_labels_by_variable_horizon.csv",index=False)
    frame = pd.DataFrame(values,columns=variables)
    frame.corr(min_periods=30).to_csv(out / "training_pairwise_correlations.csv")
    observed = np.isfinite(values).astype(np.int64)
    pd.DataFrame(observed.T @ observed,index=variables,columns=variables).to_csv(out / "training_correlation_pair_counts.csv")
    coverage = frame.notna().mean().rename("observed_fraction").to_frame()
    coverage["observed_unique_rows"] = frame.notna().sum()
    coverage.to_csv(out / "training_row_coverage.csv")
    summary = dict(dataset_sha256=audit["dataset_sha256"], train_episodes=len(train_episodes),
        train_unique_source_rows=len(train_rows), train_origins=len(train_samples),
        training_label_pairs=int(table.new_train_labels.sum()), old_training_label_pairs=int(table.old_train_labels.sum()),
        minimum_labels_per_cell=int(table.new_train_labels.min()), cells_below_20_train_labels=int(table.new_train_labels.lt(20).sum()),
        final_not_packed=True, prior_tests_excluded=True, meta_calibration_values_analyzed=False,
        official_new_hourly_labels=0, domain="terrestrial ICU", correlation_caveat="Descriptive pairwise correlations; rows cluster within episodes, missingness is selective, and associations are not causal.")
    (out / "summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    lines = ["# Auditoría de ampliación horaria V2", "",
        "Se verificó el contenido oficial de Resources: cinco enlaces documentales, sin tablas horarias nuevas. Se descargaron las cinco fuentes y enlaces de seguimiento para contexto y recuperación documental; no se contaron como observaciones.", "",
        "La ampliación observacional independiente proviene de PhysioNet Challenge 2019, UCI terrestre. Se congelaron las identidades antes de descargarlas: 1.000 episodios nuevos de entrenamiento, 200 para aprender el ensemble y 300 para prueba final. No hay solapamiento con los 1.000 episodios originales ni los 300 de confirmación consumidos.", "",
        f"El NPZ no final reúne 1.600 episodios de entrenamiento ({len(train_samples):,} orígenes), 150 de desarrollo, 200 META y 100 de calibración. Se excluyeron la prueba original y la confirmación anterior. Los 300 finales solo tienen comprobación estructural y hashes; no se empaquetan ni evalúan antes del freeze final.", "",
        f"Entrenamiento: {len(train_rows):,} filas fuente únicas, {summary['training_label_pairs']:,} pares etiqueta–origen–horizonte frente a {summary['old_training_label_pairs']:,} antes. El mínimo por variable/horizonte pasa a {summary['minimum_labels_per_cell']} etiquetas, y quedan {summary['cells_below_20_train_labels']} combinaciones con menos de veinte. Los pares solapados no son observaciones independientes.", "",
        "Se conservan 34 variables, horizontes 1/4/8/24 h y contexto de hasta 24 horas con datos horarios. No se fabrica una resolución de 15 minutos ni se emparejan registros de personas diferentes. Se aplica la misma política de calidad congelada.", "",
        "Las tablas en `outputs/data_audit_v2/` incluyen cobertura y etiquetas por variable/horizonte, correlaciones y número de pares simultáneos. Solo se han analizado valores del entrenamiento para este EDA; META, calibración y prueba final no dirigen la exploración. Las correlaciones dependen del muestreo clínico, y no demuestran relaciones causales.", "",
        "El script de adquisición guarda roles y hashes en `data/manifests/hourly_expansion_v2.json`; el dataset, sus roles y auditoría están en `data/processed/hourly_v2/`. Se preservan los originales y todos los artefactos V1."]
    (ROOT / "docs/ampliacion-horaria-v2.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
