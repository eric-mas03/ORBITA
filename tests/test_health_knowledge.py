"""Local document retrieval tests, without clinical cases or model calls."""
import hashlib
import json

import pytest

from astronaut_twin.health_knowledge import (
    DOMAIN, LocalHealthKnowledgeRetriever, build_index, chunk_paragraphs, extract_document,
)


def source_fixture(tmp_path):
    source = tmp_path / "source.html"
    source.write_text('<html><head><title>Document fixture</title></head><body>'
                      '<nav><p>Navigation menu hidden distraction words.</p></nav>'
                      '<main><article><p>Neutrons and radiation exposure are measured by an instrument in space.</p>'
                      '<p>Sleep and isolation research use different documented measurements.</p>'
                      '<footer><p>Footer menu contact navigation only.</p></footer></article></main></body></html>')
    entry = {"role": "resource_page", "status": "downloaded", "resource_id": "fixture_doc",
             "url": "https://www.nasa.gov/fixture-document/", "path": "source.html",
             "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"acquisitions": [entry,
        {**entry, "role": "official_challenge_listing"},
        {**entry, "role": "local_organizer_mirror", "url": "https://mirror.example.test/"},
        {**entry, "url": "https://www.nasa.gov.evil.test/"}]}))
    return source, manifest


def test_scoped_extraction_excludes_navigation_and_footer():
    title, scope, paragraphs = extract_document(b'<title>Fixture</title><nav><p>DO NOT INDEX THIS MENU.</p></nav>'
        b'<main><p>Primary document paragraph with enough words.</p><footer><p>DO NOT INDEX FOOTER.</p></footer></main>')
    assert title == "Fixture" and scope == "main"
    assert paragraphs == ["Primary document paragraph with enough words."]


def test_no_whole_body_fallback_and_explicit_hrr_intro_supported():
    assert extract_document(b'<body><nav>Menu only</nav><p>Unscoped page words are not silently accepted.</p></body>')[2] == []
    assert extract_document(b'<div class="introText js_acronyms">Document introduction with enough words.</div>')[1] == "div.introText_explicit_fallback"
    # Legacy ASP.NET wraps real public content in a structural form.
    assert extract_document(b'<form method="post"><nav>Do not index navigation</nav><div class="introText js_acronyms">Document introduction with enough words.</div></form>')[2] == ["Document introduction with enough words."]


def test_chunk_size_and_locators_are_explicit():
    chunks = chunk_paragraphs([" ".join(["token"] * 245)], maximum_words=120)
    assert [len(chunk["text"].split()) for chunk in chunks] == [120, 120, 5]
    assert chunks[-1]["paragraphs"] == ["p1:words241-245"]


def test_primary_allowlist_hashes_aliases_no_matches_and_domain(tmp_path):
    _, manifest = source_fixture(tmp_path)
    destination = tmp_path / "index"
    result = build_index(manifest, destination, root=tmp_path)
    assert len(result["documents"]) == 1
    assert len(result["excluded"]) == 3
    assert result["no_patient_or_forecasting_labels"]
    retriever = LocalHealthKnowledgeRetriever(destination)
    first = retriever.retrieve("radiación neutrones", domain=DOMAIN)
    second = retriever.retrieve("radiación neutrones", domain=DOMAIN)
    assert first and [(r.chunk_id, r.score) for r in first] == [(r.chunk_id, r.score) for r in second]
    assert first[0].citation.uri.startswith("https://www.nasa.gov/")
    assert first[0].citation.sha256 and first[0].citation.locator
    assert "Navigation" not in first[0].text and "Footer" not in first[0].text
    assert retriever.retrieve("xyznevermatches987", domain=DOMAIN) == []
    assert retriever.retrieve("radiation", domain="clinical_patient") == []


def test_source_or_index_tampering_is_rejected(tmp_path):
    source, manifest = source_fixture(tmp_path)
    destination = tmp_path / "index"
    build_index(manifest, destination, root=tmp_path)
    source.write_text("changed source")
    with pytest.raises(ValueError, match="source hash"):
        build_index(manifest, destination, root=tmp_path)
    with (destination / "chunks.jsonl").open("a") as file:
        file.write("{}\n")
    with pytest.raises(ValueError, match="checksum"):
        LocalHealthKnowledgeRetriever(destination)
