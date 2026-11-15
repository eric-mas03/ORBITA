"""Data integrity for the new longitudinal analog, not medical assertions."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_mars500_audit", ROOT / "scripts/prepare_mars500.py")
mars = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mars)


def test_month_labels_keep_order_without_fabricated_elapsed_days():
    assert mars.time_fields("minus10 days") == (-10, -10., "before")
    assert mars.time_fields("520 days") == (520, 520., "isolation")
    assert mars.time_fields("520 days plus 10 days") == (530, 530., "after")
    third, sixth = mars.time_fields("520 days plus 3 month"), mars.time_fields("520 days plus 6 months")
    assert np.isnan(third[1]) and np.isnan(sixth[1]) and third[0] < sixth[0]
    with pytest.raises(ValueError):
        mars.time_fields("unknown day")


def test_isolation_counts_conservation_and_compositional_abundance():
    directory = ROOT / "data/processed/mars500"
    if not directory.exists():
        pytest.skip("Acquire OSD-191 and run scripts/prepare_mars500.py first")
    meta = pd.read_csv(directory / "sample_metadata.csv", index_col="sample_id")
    asv = pd.read_csv(directory / "asv_counts.csv", index_col=0)
    phylum = pd.read_csv(directory / "phylum_counts.csv", index_col=0)
    proportion = pd.read_csv(directory / "phylum_relative_abundance.csv", index_col=0)
    assert meta.index.is_unique and asv.index.tolist() == meta.index.tolist()
    assert meta.subject_id.nunique() == 6 and len(meta) == 159
    assert asv.shape == (159, 2916) and phylum.shape == (159, 12)
    assert meta.day_from_isolation_start.notna().sum() == 150
    assert not meta.available_time_known.any()
    assert set(meta.domain) == {"space_analog_isolation"}
    assert not asv.isna().any().any() and asv.eq(0).any().any()
    np.testing.assert_array_equal(asv.sum(axis=1), meta.sequencing_reads)
    np.testing.assert_array_equal(phylum.sum(axis=1), meta.sequencing_reads)
    np.testing.assert_allclose(proportion.to_numpy(), phylum.div(meta.sequencing_reads, axis=0).to_numpy())
    np.testing.assert_allclose(proportion.sum(axis=1), 1.)
