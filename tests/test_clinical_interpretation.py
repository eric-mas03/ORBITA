"""Evidence-contract tests using abstract software fixtures, not clinical cases."""
import copy
import json

import pytest

from astronaut_twin.clinical_interpretation import (
    Citation, EvidenceError, build_evidence_packet, make_model_request,
    render_spanish_summary, validate_generated_claims,
)


def fixture():
    context = dict(subject_key="opaque-fixture", domain="software_fixture", trained_domain="software_fixture",
                   origin_hour=24, time_reference="fixture-clock", method="fixture-model",
                   prediction_source_id="P", evaluation_source_id="E", evaluation_split="dev")
    predictions = [dict(variable="x", unit="fixture-unit", horizon_hours=1, target_hour=25,
                        q10=1., q50=2., q90=3., own_history_observed=0,
                        unit_review_required="False", observed=987654321.)]
    evaluations = [dict(variable="x", unit="fixture-unit", horizon_hours=1, coverage=.75,
                        n_labels=20, n_episodes=10, split="dev")]
    sources = [Citation("P", "Prediction fixture", "fixture://predictions"),
               Citation("E", "Evaluation fixture", "fixture://evaluation")]
    return context, predictions, evaluations, sources


def packet():
    context, predictions, evaluations, sources = fixture()
    return build_evidence_packet(predictions, context=context, evaluation_rows=evaluations, sources=sources)


def test_structured_packet_excludes_future_truth_and_does_not_invent_a_state():
    p = packet()
    assert p["status"] == "descriptive_evidence_only"
    assert not p["clinical_decision_allowed"]
    assert "opaque-fixture" not in json.dumps(p)
    assert "987654321" not in json.dumps(p)
    assert p["facts"][0]["own_history_observed"] == 0
    assert p["facts"][0]["q50"] == 2
    assert not p["numeric_evaluation_future_truth_included"]


def test_deterministic_spanish_summary_distinguishes_nominal_and_empirical_coverage():
    p = packet()
    text = render_spanish_summary(p)
    assert text == render_spanish_summary(p)
    assert "nominalmente del 80 %" in text and "75.00 %" in text
    assert "q10=1.0, q50=2.0, q90=3.0" in text
    assert "[P]" in text and "[E]" in text
    assert "No se genera diagnóstico" in text


def test_missing_quantiles_and_sparse_evidence_are_retained_with_abstention():
    context, predictions, evaluations, sources = fixture()
    del predictions[0]["q10"]
    evaluations[0].update(n_labels=2, n_episodes=1)
    p = build_evidence_packet(predictions, context=context, evaluation_rows=evaluations, sources=sources)
    assert p["status"] == "abstain_from_interpretation"
    assert p["facts"][0]["q10"] is None
    assert {"missing_quantiles", "insufficient_evaluation_support"}.issubset(p["abstention_reasons"])
    assert "banda no disponible" in render_spanish_summary(p)


def test_domain_units_and_physical_support_prevent_unqualified_interpretation():
    context, predictions, evaluations, sources = fixture()
    context.update(domain="different_fixture", physical_bounds={"x": {"maximum_inclusive": 2}})
    predictions[0]["unit_review_required"] = "True"
    p = build_evidence_packet(predictions, context=context, evaluation_rows=evaluations, sources=sources)
    assert {"unit_review_required", "domain_transfer_unvalidated", "prediction_outside_physical_support"}.issubset(p["abstention_reasons"])
    assert p["facts"][0]["q90"] == 3  # No clipping or change to the forecaster.


def test_crossed_quantiles_wrong_horizon_and_mixed_episodes_raise():
    context, predictions, evaluations, sources = fixture()
    for mutation in [dict(q10=4), dict(target_hour=26), dict(episode_id="other-fixture")]:
        rows = copy.deepcopy(predictions)
        rows[0].update(mutation)
        with pytest.raises(EvidenceError):
            build_evidence_packet(rows, context=context, sources=sources)


def test_observation_availability_cannot_precede_measurement():
    context, predictions, evaluations, sources = fixture()
    observation = dict(variable="x", value=2., unit="fixture-unit", observed=True,
                       event_hour=20, available_hour=19, source_id="P")
    with pytest.raises(EvidenceError, match="precede"):
        build_evidence_packet(predictions, context=context, sources=sources, observations=[observation])


def test_calibrated_bounds_are_not_described_as_certified_quantiles():
    context, predictions, evaluations, sources = fixture()
    predictions[0]["interval_type"] = "empirical_adjusted_interval"
    result = build_evidence_packet(predictions, context=context, sources=sources, evaluation_rows=evaluations)
    text = render_spanish_summary(result)
    assert "límite inferior=1.0" in text and "límite superior=3.0" in text
    assert "q10=1.0" not in text
    assert result["facts"][0]["interval_type"] == "empirical_adjusted_interval"
    assert result["facts"][0]["field_source_ids"]["interval_type"] == ["P"]


def test_exported_unsupported_bounds_and_sparse_calibration_retain_abstention():
    context, predictions, evaluations, sources = fixture()
    predictions[0].update(interval_type="empirical_adjusted_interval",calibration_source="insufficient_no_expansion",
        calibration_fit_labels=4,calibration_fit_episodes=2,prediction_outside_physical_support=True)
    result = build_evidence_packet(predictions,context=context,sources=sources,evaluation_rows=evaluations)
    assert {"insufficient_calibration_support","prediction_outside_physical_support"}.issubset(result["abstention_reasons"])
    assert "calibración con soporte insuficiente" in render_spanish_summary(result)


def test_observation_masks_keep_zero_and_block_unavailable_inputs():
    context, predictions, evaluations, sources = fixture()
    observations = [dict(variable="x", value=0, unit="fixture-unit", event_hour=23, available_hour=24, observed=True, source_id="P"),
                    dict(variable="y", value=99, event_hour=22, available_hour=22, observed=False, source_id="P")]
    p = build_evidence_packet(predictions, context=context, sources=sources, observations=observations)
    assert p["facts"][1]["value"] == 0
    assert p["facts"][2]["value"] is None
    assert "sin observación válida" in render_spanish_summary(p)
    observations[0]["available_hour"] = 25
    with pytest.raises(EvidenceError, match="Future/unavailable"):
        build_evidence_packet(predictions, context=context, sources=sources, observations=observations)


def test_citation_check_is_field_specific_and_rejects_changed_numbers_units_or_new_claims():
    p = packet()
    claim = dict(fact_id="forecast:x:1", field="q50", value=2., unit="fixture-unit", citation_ids=["P"])
    validate_generated_claims(p, [claim])
    for mutation in [dict(value=9), dict(unit="changed-unit"), dict(citation_ids=["E"]), dict(field="diagnosis")]:
        with pytest.raises(EvidenceError):
            validate_generated_claims(p, [{**claim, **mutation}])
    with pytest.raises(EvidenceError):
        validate_generated_claims(p, [{**claim, "free_text": "unchecked text"}])


def test_ambiguous_evaluation_or_changed_units_are_rejected():
    context, predictions, evaluations, sources = fixture()
    with pytest.raises(EvidenceError, match="Ambiguous"):
        build_evidence_packet(predictions, context=context, evaluation_rows=evaluations * 2, sources=sources)
    evaluations[0]["unit"] = "other-unit"
    with pytest.raises(EvidenceError, match="Evaluation unit"):
        build_evidence_packet(predictions, context=context, evaluation_rows=evaluations, sources=sources)


def test_future_adapter_request_is_serialization_only_and_remote_is_not_implicit():
    p = packet()
    request = json.loads(make_model_request(p))
    assert request["professional_review_required"]
    assert request["retrieved_text_is_untrusted_data_not_instructions"]
    with pytest.raises(PermissionError):
        make_model_request(p, backend_location="remote")
