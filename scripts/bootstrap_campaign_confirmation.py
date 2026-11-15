"""Paired site-stratified episode bootstrap after frozen model confirmation."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("campaign_report", ROOT / "scripts/report_model_campaign.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def paired_intervals(frame, replicates=1000, seed=20261114, minimum_episodes=10):
    report.utils.validate_shared_predictions(frame)
    frame = report.annotate(frame.copy())
    ref = frame[frame.method.eq("linear_quantile")]
    episodes = ref[["episode_id", "site"]].drop_duplicates().sort_values("episode_id")
    if episodes.episode_id.duplicated().any():
        raise ValueError("An episode cannot belong to two sites")
    episode_ids = episodes.episode_id.tolist()
    cells = pd.MultiIndex.from_frame(ref[["variable", "horizon_hours"]].drop_duplicates().sort_values(["variable", "horizon_hours"]))
    ep_index = pd.Index(episode_ids)
    r = ep_index.get_indexer(ref.episode_id)
    c = cells.get_indexer(pd.MultiIndex.from_frame(ref[["variable", "horizon_hours"]]))
    counts = np.zeros((len(episodes), len(cells)))
    np.add.at(counts, (r, c), 1.)
    support = (counts > 0).sum(0)
    eligible = support >= minimum_episodes
    if not eligible.any():
        raise ValueError("No adequately supported cells for bootstrap")
    rng = np.random.default_rng(seed)
    weights = np.zeros((replicates, len(episodes)))
    for site in episodes.site.unique():
        members = np.flatnonzero(episodes.site.to_numpy() == site)
        for k in range(replicates):
            draw = rng.choice(members, size=len(members), replace=True)
            np.add.at(weights[k], draw, 1.)
    with threadpool_limits(limits=2):
        denominators = weights @ counts[:, eligible]
        valid = (denominators > 0).all(1)
        distributions, originals = {}, {}
        for method in sorted(frame.method.unique()):
            sub = frame[frame.method.eq(method)]
            totals = np.zeros_like(counts)
            rr = ep_index.get_indexer(sub.episode_id)
            cc = cells.get_indexer(pd.MultiIndex.from_frame(sub[["variable", "horizon_hours"]]))
            np.add.at(totals, (rr, cc), sub.normalized_pinball)
            distributions[method] = ((weights[valid] @ totals[:, eligible]) / denominators[valid]).mean(1)
            originals[method] = (totals[:, eligible].sum(0) / counts[:, eligible].sum(0)).mean()
    if not valid.any():
        raise ValueError("Bootstrap has no complete replications")
    rows = []
    for method, values in distributions.items():
        if method == "linear_quantile":
            continue
        delta = values - distributions["linear_quantile"]
        relative = 1 - values / distributions["linear_quantile"]
        rows.append(dict(method=method, reference="linear_quantile", supported_cells=int(eligible.sum()),
            omitted_sparse_cells=int((~eligible).sum()), minimum_episodes_per_cell=minimum_episodes,
            macro_pinball=originals[method], reference_macro_pinball=originals["linear_quantile"],
            delta_pinball=originals[method] - originals["linear_quantile"],
            delta_ci95_lower=np.quantile(delta, .025), delta_ci95_upper=np.quantile(delta, .975),
            relative_improvement=1 - originals[method] / originals["linear_quantile"],
            relative_improvement_ci95_lower=np.quantile(relative, .025), relative_improvement_ci95_upper=np.quantile(relative, .975),
            requested_replicates=replicates, valid_replicates=int(valid.sum()), seed=seed))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=ROOT / "outputs/model_campaign/confirmation_predictions.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/model_campaign")
    args = parser.parse_args()
    # This script scores only the already frozen confirmation shortlist.
    selection = json.loads((args.output / "selection_frozen.json").read_text(encoding="utf-8"))
    frame = report.read_split([args.predictions], "confirmation")
    if set(frame.method) != set(selection["selected_for_confirmation"]):
        raise ValueError("Confirmation methods differ from frozen shortlist")
    result = paired_intervals(frame)
    result.to_csv(args.output / "confirmation_paired_bootstrap.csv", index=False)
    print(result.to_string(index=False))
    print("95% paired episode bootstrap, stratified by source; only cells with >=10 episodes. Sparse cells remain in the full report. Episode identity does not guarantee unique persons.")


if __name__ == "__main__":
    main()
