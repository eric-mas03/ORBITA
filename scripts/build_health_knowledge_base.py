"""Index only hashed public NASA/CSA background; no model or patient data."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from astronaut_twin.health_knowledge import build_index, LocalHealthKnowledgeRetriever, DOMAIN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifests/challenge_resources_v2.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/health_knowledge_v2")
    args = parser.parse_args()
    result = build_index(args.manifest, args.output, root=ROOT)
    retriever = LocalHealthKnowledgeRetriever(args.output)
    query = "radiación neutrones salud astronautas"
    hits = retriever.retrieve(query, domain=DOMAIN, top_k=3)
    demo = {"query": query, "retrieval": "TF-IDF lexical; explicit Spanish-English aliases; not semantic embeddings",
            "domain": DOMAIN, "not_patient_advice": True,
            "results": [{"chunk_id": hit.chunk_id, "score": hit.score,
                         "title": hit.citation.title, "url": hit.citation.uri,
                         "source_sha256": hit.citation.sha256, "locator": hit.citation.locator,
                         "excerpt": " ".join(hit.text.split()[:20])} for hit in hits]}
    (args.output / "demo_radiacion.json").write_text(json.dumps(demo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"documents": len(result["documents"]), "chunks": result["chunk_count"],
                      "index_sha256": result["chunks_sha256"], "demo_hits": len(hits),
                      "patient_data_used": False, "llm_executed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
