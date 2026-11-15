"""Inspect temporal tensor shapes on real windows; no fitting or scoring."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from astronaut_twin.hourly_neural import make_hourly_twin, prepare_hourly_neural, flatten_predictions


def main():
    torch.set_num_threads(2)
    with np.load(ROOT / "data/processed/hourly/hourly_dataset.npz", allow_pickle=False) as archive:
        packed = {key: archive[key] for key in archive.files}
    config = json.loads((ROOT / "config/hourly_task_candidates.json").read_text(encoding="utf-8"))
    groups = config["output_groups"]
    data = prepare_hourly_neural(packed)
    rows = np.flatnonzero(packed["origins"] - packed["history_start"] + 1 == 24)[:2]
    arguments, _, _ = data.batch(rows, groups)
    model = make_hourly_twin(groups, "transformer_grouped").eval()
    shapes = {"input_by_group": {name: list(value.shape) for name, value in arguments[0].items()},
              "horizons_hours": arguments[4].tolist(), "encoder_by_group": {}}
    handles = []

    def capture(name):
        def hook(_module, _args, result):
            shapes[name] = list(result.shape)
        return hook

    for group, encoder in model.group_encoders.items():
        def group_hook(_module, _args, result, group=group):
            shapes["encoder_by_group"][group] = list(result.shape)
        handles.append(encoder.register_forward_hook(group_hook))
    for name, layer in (("fused_latent_sequence", model.fusion), ("temporal_memory_sequence", model.temporal)):
        handles.append(layer.register_forward_hook(capture(name)))
    with torch.no_grad():
        predictions = model(*arguments)
        flattened = flatten_predictions(predictions, groups, data.variables)
    for handle in handles:
        handle.remove()
    shapes["outputs_by_group"] = {name: list(value.shape) for name, value in predictions.items()}
    shapes["flattened_outputs_batch_horizon_variable_quantile"] = list(flattened.shape)
    assert shapes["fused_latent_sequence"] == [2, 24, 16]
    assert shapes["temporal_memory_sequence"] == [2, 24, 16]
    assert list(flattened.shape) == [2, 4, 34, 3]
    shapes["interpretation"] = {
        "temporal_axis_preserved": True,
        "latent_dimension_is_per_time_bin": True,
        "current_source_bin_minutes": 60,
        "desired_24h_15min_bin_count": 96,
        "current_constructor_fixed_bin_count": 24,
        "base_transformer_accepts_variable_T": True,
        "weights": "Fresh instance of implemented architecture; only shapes inspected, no numeric predictions reported.",
        "training_performed": False,
    }
    out = ROOT / "outputs/architecture"
    out.mkdir(parents=True, exist_ok=True)
    (out / "temporal_shape_trace.json").write_text(json.dumps(shapes, indent=2), encoding="utf-8")
    print(json.dumps({key: shapes[key] for key in ["fused_latent_sequence", "temporal_memory_sequence", "flattened_outputs_batch_horizon_variable_quantile", "interpretation"]}, indent=2))


if __name__ == "__main__":
    main()
