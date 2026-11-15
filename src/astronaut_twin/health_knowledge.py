"""Deterministic local lexical retrieval over hashed NASA/CSA documentation.

TF-IDF cosine retrieval, not semantic embeddings and not clinical validation.
Only public background documents are indexed; no patient/model outcomes enter.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
import unicodedata
from urllib.parse import urlparse

from .clinical_interpretation import Citation, RetrievedPassage

DOMAIN = "space_health_background"
ALIASES = {
    "radiacion": ("radiation",), "neutrones": ("neutron", "neutrons"),
    "salud": ("health",), "astronauta": ("astronaut", "astronauts"),
    "astronautas": ("astronaut", "astronauts"), "dosis": ("dose", "dosimetry"),
    "sueno": ("sleep",), "aislamiento": ("isolation",), "gravedad": ("gravity",),
    "microgravedad": ("microgravity",), "huesos": ("bone", "bones"),
    "musculos": ("muscle", "muscles"), "riesgos": ("risk", "risks"),
    "inmune": ("immune",), "inmunidad": ("immune",), "blindaje": ("shielding",),
}
STOP = set("a an and are as at be by for from in is it its of on or that the their this to was were which with y de del la las el los en un una que para por como con sobre".split())
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
SKIP = {"nav", "header", "footer", "aside", "script", "style", "button", "select", "textarea", "noscript", "svg", "iframe"}


@dataclass
class Node:
    tag: str
    attrs: dict
    children: list


class DocumentParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node("root", {}, [])
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        node = Node(tag, dict(attrs), [])
        self.stack[-1].children.append(node)
        if tag not in VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.stack[-1].children.append(Node(tag, dict(attrs), []))

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                self.stack = self.stack[:i]
                break

    def handle_data(self, text):
        self.stack[-1].children.append(text)


def _skip(node):
    marker = (node.attrs.get("id", "") + " " + node.attrs.get("class", "")).lower()
    return (node.tag in SKIP or "hidden" in node.attrs or node.attrs.get("aria-hidden") == "true" or
            "display:none" in node.attrs.get("style", "").replace(" ", "").lower() or
            node.attrs.get("role") in {"navigation", "banner", "contentinfo", "search"} or
            (node.tag == "form" and "search" in marker) or
            any(term in marker for term in ("breadcrumb", "menu-item", "social-share", "slsdnavigation", "slsdfooter", "navigatorbar", "navigatorlinks")))


def _walk(node):
    if _skip(node):
        return
    yield node
    for child in node.children:
        if isinstance(child, Node):
            yield from _walk(child)


def _text(node):
    if _skip(node):
        return ""
    return re.sub(r"\s+", " ", " ".join(_text(c) if isinstance(c, Node) else c for c in node.children)).strip()


def extract_document(payload: bytes) -> tuple[str, str, list[str]]:
    """Scope to article/main; explicit HRR intro fallback, never whole-body nav."""
    parser = DocumentParser()
    parser.feed(payload.decode("utf-8-sig", errors="replace"))
    nodes = list(_walk(parser.root))
    title_node = next((n for n in nodes if n.tag == "title"), None)
    title = _text(title_node) if title_node else "Untitled primary document"
    articles = [n for n in nodes if n.tag == "article"]
    mains = [n for n in nodes if n.tag == "main"]
    intros = [n for n in nodes if "introText" in n.attrs.get("class", "").split()]
    if articles:
        region, kind = max(articles, key=lambda n: len(_text(n))), "article"
    elif mains:
        region, kind = max(mains, key=lambda n: len(_text(n))), "main"
    elif intros:
        region, kind = max(intros, key=lambda n: len(_text(n))), "div.introText_explicit_fallback"
    else:
        return title, "no_scoped_content", []
    blocks = [_text(n) for n in _walk(region) if n.tag in {"p", "h1", "h2", "h3"}]
    blocks = [text for text in blocks if len(text.split()) >= 4]
    if not blocks and intros:
        blocks = [_text(region)]
    return title, kind, list(dict.fromkeys(blocks))


def chunk_paragraphs(paragraphs: list[str], maximum_words: int = 120) -> list[dict]:
    if maximum_words < 10:
        raise ValueError("Chunk size too small")
    chunks, buffer, locators = [], [], []
    for index, paragraph in enumerate(paragraphs, 1):
        words = paragraph.split()
        for start in range(0, len(words), maximum_words):
            segment = words[start:start + maximum_words]
            if buffer and len(buffer) + len(segment) > maximum_words:
                chunks.append({"text": " ".join(buffer), "paragraphs": locators})
                buffer, locators = [], []
            buffer.extend(segment)
            locators.append(f"p{index}:words{start+1}-{start+len(segment)}")
    if buffer:
        chunks.append({"text": " ".join(buffer), "paragraphs": locators})
    return chunks


def _primary(url):
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and (host == "nasa.gov" or host.endswith(".nasa.gov") or host == "asc-csa.gc.ca" or host.endswith(".asc-csa.gc.ca"))


def build_index(manifest_path: Path, destination: Path, *, root: Path, maximum_words=120) -> dict:
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    records, documents, excluded = [], [], []
    seen = set()
    for entry in manifest["acquisitions"]:
        if (entry.get("role") not in {"resource_page", "directly_linked_follow_up_page"} or
                entry.get("status") != "downloaded" or not _primary(entry.get("url", ""))):
            excluded.append({"role": entry.get("role"), "url": entry.get("url"), "reason": "not_downloaded_primary_resource_or_followup"})
            continue
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Knowledge source outside workspace")
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError("Knowledge source hash differs from acquisition")
        if (entry["url"], digest) in seen:
            continue
        seen.add((entry["url"], digest))
        title, scope, paragraphs = extract_document(payload)
        doc_id = entry["resource_id"]
        chunks = chunk_paragraphs(paragraphs, maximum_words)
        documents.append({"document_id": doc_id, "url": entry["url"], "sha256": digest,
                          "source_path": entry["path"], "title": title, "scope": scope,
                          "paragraph_count": len(paragraphs), "chunk_count": len(chunks), "domain": DOMAIN})
        for i, chunk in enumerate(chunks, 1):
            chunk_id = f"{doc_id}:chunk{i}"
            records.append({"chunk_id": chunk_id, "document_id": doc_id, "title": title,
                "url": entry["url"], "source_sha256": digest, "source_path": entry["path"],
                "locator": scope + "/" + ",".join(chunk["paragraphs"]), "text": chunk["text"],
                "text_sha256": hashlib.sha256(chunk["text"].encode()).hexdigest(), "domain": DOMAIN,
                "observational_data": False, "patient_labels": False})
    if not records:
        raise ValueError("No scoped primary knowledge paragraphs")
    destination.mkdir(parents=True, exist_ok=True)
    index = destination / "chunks.jsonl"
    index.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    result = {"schema_version": "2.0", "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "retrieval": "local_TFIDF_cosine_lexical_not_semantic_embeddings", "maximum_chunk_words": maximum_words,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tfidf": "tf=1+log(count);idf=log((1+chunks)/(1+document_frequency))+1;L2 vectors;cosine",
        "stopwords": sorted(STOP),
        "locator_definition": "Ordered extracted text blocks (paragraphs and headings) within the named region, with word spans; not DOM nth-child or original source lines.",
        "domain": DOMAIN, "documents": documents, "excluded": excluded, "chunk_count": len(records),
        "acquisition_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "chunks_sha256": hashlib.sha256(index.read_bytes()).hexdigest(), "spanish_query_aliases": ALIASES,
        "no_patient_or_forecasting_labels": True}
    (destination / "index_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def tokenize(text: str, *, expand_query=False) -> list[str]:
    normalized = "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))
    tokens = [word for word in re.findall(r"[a-z0-9]+", normalized) if word not in STOP]
    if expand_query:
        tokens += [alias for word in list(tokens) for alias in ALIASES.get(word, ())]
    return tokens


@dataclass(frozen=True)
class LexicalPassage(RetrievedPassage):
    score: float
    chunk_id: str


class LocalHealthKnowledgeRetriever:
    """EvidenceRetriever-compatible TF-IDF ranking, zero hits on no term match."""
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        metadata = json.loads((self.directory / "index_manifest.json").read_text(encoding="utf-8"))
        if metadata["implementation_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
            raise ValueError("Knowledge retrieval implementation changed; rebuild the versioned index")
        payload = (self.directory / "chunks.jsonl").read_bytes()
        if hashlib.sha256(payload).hexdigest() != metadata["chunks_sha256"]:
            raise ValueError("Knowledge index checksum mismatch")
        self.records = [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
        counts = [Counter(tokenize(record["text"])) for record in self.records]
        document_frequency = Counter(term for count in counts for term in count)
        self.idf = {term: math.log((1 + len(counts)) / (1 + frequency)) + 1 for term, frequency in document_frequency.items()}
        self.vectors = [self._vector(count) for count in counts]

    def _vector(self, counts):
        values = {term: (1 + math.log(count)) * self.idf[term] for term, count in counts.items() if term in self.idf}
        norm = math.sqrt(sum(value * value for value in values.values()))
        return {term: value / norm for term, value in values.items()} if norm else {}

    def retrieve(self, query: str, *, domain: str = DOMAIN, top_k: int = 4) -> list[LexicalPassage]:
        if top_k < 1 or domain != DOMAIN:
            return []
        vector = self._vector(Counter(tokenize(query, expand_query=True)))
        if not vector:
            return []
        ranked = []
        for record, document in zip(self.records, self.vectors):
            score = sum(weight * document.get(term, 0.) for term, weight in vector.items())
            if score > 0 and record["domain"] == domain:
                ranked.append((score, record["chunk_id"], record))
        ranked.sort(key=lambda row: (-row[0], row[1]))
        now = datetime.now(timezone.utc).isoformat()
        return [LexicalPassage(citation=Citation(row["chunk_id"], row["title"], row["url"],
                    sha256=row["source_sha256"], locator=row["locator"], version=row["source_sha256"]),
                    text=row["text"], domain=DOMAIN, retrieved_at=now, score=score, chunk_id=row["chunk_id"])
                for score, _, row in ranked[:top_k]]
