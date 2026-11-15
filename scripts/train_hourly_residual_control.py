"""Train a predeclared own-summary MLP control; development only, no attention."""
import copy
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from astronaut_twin.hourly_architecture_trials import prepare_frozen_trial_data
from astronaut_twin.hourly_baselines import predict_linear
from astronaut_twin.hourly_neural import global_cell_loss
from astronaut_twin.hourly_residual_control import SummaryResidualMLP


def load_npz(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def main():
    torch.set_num_threads(2)
    started = time.monotonic()
    out = ROOT / "outputs/hourly_residual_control"
    out.mkdir(parents=True, exist_ok=True)
    packed = load_npz(ROOT / "data/processed/hourly/hourly_dataset.npz")
    episode = packed["sample_episode"]
    keep = np.isin(packed["episode_split"][episode], ["train", "dev"])
    for key in ("origins", "history_start", "target_rows", "sample_episode"):
        packed[key] = packed[key][keep]
    directory = ROOT / "outputs/hourly_baselines"
    scalers = load_npz(directory / "scalers.npz")
    neural, baseline = prepare_frozen_trial_data(packed, scalers)
    frozen = predict_linear(baseline, load_npz(directory / "linear_quantile.npz"), load_npz(directory / "persistence.npz"))
    base_z = torch.tensor((frozen - neural.mean[None, None, :, None]) / neural.scale[None, None, :, None], dtype=torch.float32)
    own = torch.from_numpy(baseline.features)
    target = (neural.labels - torch.tensor(neural.mean)) / torch.tensor(neural.scale)
    mask = torch.isfinite(target)
    target = torch.nan_to_num(target)
    train, dev = (np.flatnonzero(neural.split == split) for split in ("train", "dev"))
    counts = mask[train].sum(0)
    model = SummaryResidualMLP(neural.variables)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    rng = np.random.default_rng(42)

    def evaluate():
        sums, nums = torch.zeros(4, 34), torch.zeros(4, 34)
        model.eval()
        with torch.no_grad():
            for i in range(0, len(dev), 256):
                rows = dev[i:i+256]
                error = target[rows, ..., None] - model(None, own[rows], base_z[rows])
                q = error.new_tensor([.1, .5, .9])
                loss = torch.maximum(q * error, (q-1) * error).mean(-1)
                sums += (loss * mask[rows]).sum(0)
                nums += mask[rows].sum(0)
        active = nums > 0
        return float((sums[active] / nums[active]).mean())

    best, epoch_best, stale = evaluate(), 0, 0
    initial = best
    state = copy.deepcopy(model.state_dict())
    curves = [dict(epoch=0, dev_macro_pinball=best)]
    for epoch in range(1, 31):
        model.train()
        shuffled = rng.permutation(train)
        for i in range(0, len(train), 256):
            rows = shuffled[i:i+256]
            optimizer.zero_grad(set_to_none=True)
            loss = global_cell_loss(model(None, own[rows], base_z[rows]), target[rows], mask[rows], counts, len(train))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        score = evaluate()
        curves.append(dict(epoch=epoch, dev_macro_pinball=score))
        print(f"own_summary_mlp_residual epoch={epoch} dev={score:.9f}", flush=True)
        if score < best - 1e-6:
            best, epoch_best, stale, state = score, epoch, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= 5:
            break
    model.load_state_dict(state)
    metadata = json.loads((ROOT / "config/hourly_task_candidates.json").read_text(encoding="utf-8"))
    checkpoint = dict(state_dict=state, variables=neural.variables, horizons=(1, 4, 8, 24), hidden=32, seed=42,
        selected_epoch=epoch_best, dev_macro_pinball=best, groups=metadata["output_groups"],
        scaler_mean=torch.from_numpy(neural.mean), scaler_scale=torch.from_numpy(neural.scale))
    torch.save(checkpoint, out / "own_summary_mlp_residual.pt")
    np.savez_compressed(out / "scalers.npz", **scalers)
    pd.DataFrame(curves).to_csv(out / "learning_curves.csv", index=False)
    metadata = json.loads((ROOT / "config/hourly_task_candidates.json").read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("trials_trainer", ROOT / "scripts/train_hourly_architecture_trials.py")
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    exporter.export_dev(model, "own_summary_mlp_residual", neural, own, base_z, baseline, packed,
                        metadata["output_groups"], metadata, SimpleNamespace(batch_size=256, out=out), first=True)
    sources = [Path(__file__), ROOT / "src/astronaut_twin/hourly_residual_control.py", directory / "scalers.npz",
               directory / "linear_quantile.npz", directory / "persistence.npz", ROOT / "data/processed/hourly/hourly_dataset.npz"]
    manifest = dict(status="development_only_no_attention_control_complete", selection_split="dev", seed=42,
        method="own_summary_mlp_residual", parameter_count=sum(p.numel() for p in model.parameters()),
        initial_dev_macro_pinball=initial, dev_macro_pinball=best, selected_epoch=epoch_best,
        epochs_run=epoch, maximum_epochs=30, patience=5, learning_rate=.001, weight_decay=.001,
        batch_size=256, optimizer="AdamW", threads=2, elapsed_seconds=time.monotonic()-started,
        inputs="Nine frozen causal own-variable summaries plus horizon/24; independent per-target nonlinear MLP",
        temporal_attention=False, baseline_refitted=False, scalers_refitted=False, ensemble_trained=False,
        calibration_test_used=False, data_sha256=hashlib.sha256(sources[-1].read_bytes()).hexdigest(),
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (out / "fit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
