"""
Load the cinematography knowledge corpus.

Each file in knowledge/ is markdown with a YAML frontmatter block:

    ---
    title: Three-Point Lighting
    domain: lighting_color
    source: Original explanatory note
    license: original
    url: ""
    ---
    body text...

`domain` is the important field: it maps each note to one of the feature areas
the CV pipeline measures, so retrieval can be filtered to what a frame actually
shows. Valid domains:
    lighting_color | shot_composition | blocking_mise_en_scene | genre_director
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

VALID_DOMAINS = {
    "lighting_color",
    "shot_composition",
    "blocking_mise_en_scene",
    "genre_director",
}


@dataclass
class KnowledgeDoc:
    doc_id: str
    title: str
    domain: str
    source: str
    license: str
    url: str
    body: str


def _split_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        raise ValueError("Missing frontmatter block (file must start with '---').")
    parts = text.split("---", 2)
    # parts[0] is empty, parts[1] is yaml, parts[2] is body
    meta = yaml.safe_load(parts[1]) or {}
    body = parts[2].strip()
    return meta, body


def load_corpus(knowledge_dir: str | Path = "knowledge") -> list[KnowledgeDoc]:
    knowledge_dir = Path(knowledge_dir)
    if not knowledge_dir.exists():
        raise FileNotFoundError(f"Knowledge directory not found: {knowledge_dir}")

    docs: list[KnowledgeDoc] = []
    for path in sorted(knowledge_dir.glob("*.md")):
        meta, body = _split_frontmatter(path.read_text(encoding="utf-8"))
        domain = str(meta.get("domain", "")).strip()
        if domain not in VALID_DOMAINS:
            raise ValueError(
                f"{path.name}: domain '{domain}' is invalid. "
                f"Use one of {sorted(VALID_DOMAINS)}."
            )
        docs.append(
            KnowledgeDoc(
                doc_id=path.stem,
                title=str(meta.get("title", path.stem)),
                domain=domain,
                source=str(meta.get("source", "unknown")),
                license=str(meta.get("license", "unknown")),
                url=str(meta.get("url", "")),
                body=body,
            )
        )
    if not docs:
        raise ValueError(f"No .md files found in {knowledge_dir}")
    return docs
