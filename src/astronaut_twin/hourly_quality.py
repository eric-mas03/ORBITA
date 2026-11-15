"""Conservative support checks; never infer clinical normality from a value.

All values remain in source units. Missing NaNs remain missing; only invalid
values produce cell-level audit flags. No episode or variable is dropped.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

POLICY_PATH = Path(__file__).resolve().parents[2] / "config/hourly_quality_policy.json"


def load_quality_policy(path: str | Path | None = None) -> dict:
    """Read the versioned metadata and bounds used by :func:`apply_quality`."""
    return json.loads(Path(path or POLICY_PATH).read_text(encoding="utf-8"))


def apply_quality(
    values: np.ndarray, variables: Sequence[str]
) -> tuple[np.ndarray, list[dict]]:
    """Return a new float32 array and flags for invalid cells.

    ``values`` must be a numeric rows-by-variables array. Variable names are
    original Challenge 2019 columns, or their ``SEPSIS2019::`` prefixed keys.
    Use ``np.isfinite(cleaned)`` as the validity/observation mask. A missing
    source NaN is preserved without an invalid-value flag. Audit row numbers
    are zero-based positions in the provided array, not global source lines.
    ``original`` is a numeric float for finite values and a JSON-safe string
    for infinity. Bounds are inclusive. Unknown variable names raise instead
    of silently skipping review. Metadata ambiguities do not trigger a unit
    conversion or invalidate every measurement of the affected variable.
    """
    source = np.asarray(values, dtype=np.float64)
    names = list(variables)
    if source.ndim != 2 or source.shape[1] != len(names):
        raise ValueError("Expected a 2-D numeric array with one name per column")
    canonical = [name.removeprefix("SEPSIS2019::") for name in names]
    if len(set(canonical)) != len(canonical):
        raise ValueError("Duplicate variable names are not allowed")
    policy = load_quality_policy()
    rules = policy["variables"]
    unknown = sorted(set(canonical) - rules.keys())
    if unknown:
        raise ValueError(f"No reviewed quality policy for variables: {unknown}")

    cleaned = source.copy()
    flags: list[dict] = []

    def reject(column: int, rows: np.ndarray, reason: str) -> None:
        for row in np.flatnonzero(rows):
            original = source[row, column]
            flags.append({
                "row": int(row), "variable": names[column], "reason": reason,
                "original": float(original) if np.isfinite(original) else
                "Infinity" if original > 0 else "-Infinity",
            })
        cleaned[rows, column] = np.nan

    for column, name in enumerate(canonical):
        raw = source[:, column]
        reject(column, np.isinf(raw), "non_finite_value")
        rule = rules[name]
        lower = rule["minimum_inclusive"]
        upper = rule["maximum_inclusive"]
        finite = np.isfinite(cleaned[:, column])
        if lower is not None:
            reason = "below_absolute_zero" if name == "Temp" else "negative_physical_magnitude"
            reject(column, finite & (raw < lower), reason)
        if upper is not None:
            reason = "ph_above_support" if name == "pH" else "percentage_above_100"
            reject(column, np.isfinite(cleaned[:, column]) & (raw > upper), reason)
        reject(column, np.isfinite(cleaned[:, column]) &
               (np.abs(raw) > np.finfo(np.float32).max), "float32_overflow")

    return cleaned.astype(np.float32), flags
