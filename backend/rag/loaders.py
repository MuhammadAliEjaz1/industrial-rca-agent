"""
Pure-Python loaders for the RAG knowledge base (no heavy ML imports, easy to test).

Two sources:
  1. Manuals  - authored markdown files in rag/knowledge_base/ (SYNTHETIC, see each file's header).
                Split by "## " section so every chunk is one section and citations can name it.
  2. Incidents - rows from the failure_incidents table (REAL failure rows from AI4I 2020,
                with SYNTHETIC machine_id/timestamp from Phase 1).
"""

import re
import sqlite3
from pathlib import Path

RAG_DIR = Path(__file__).resolve().parent
KB_DIR = RAG_DIR / "knowledge_base"
DB_PATH = RAG_DIR.parent / "industrial_rca.db"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def load_manual_chunks(kb_dir: Path = KB_DIR) -> list[dict]:
    chunks = []
    for path in sorted(kb_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")

        title_match = re.search(r"^# (.+)$", text, re.M)
        title = title_match.group(1).strip() if title_match else path.stem

        type_match = re.search(r"^Failure type:\s*(.+)$", text, re.M)
        failure_type = type_match.group(1).strip() if type_match else "General"

        # parts[0] is the header preamble (title, status, failure type) - skip it
        parts = re.split(r"^## ", text, flags=re.M)
        for part in parts[1:]:
            heading, _, body = part.partition("\n")
            heading, body = heading.strip(), body.strip()
            if not body:
                continue
            chunks.append({
                "id": f"{path.stem}::{_slug(heading)}",
                # Title + section go INTO the embedded text so the chunk keeps its context
                "text": f"{title} | {heading}\n{body}",
                "metadata": {
                    "source": path.name,
                    "title": title,
                    "section": heading,
                    "failure_type": failure_type,
                    "doc_type": "general_guide" if failure_type in ("General", "Compound Failure") else "failure_manual",
                    "synthetic": True,
                },
            })
    return chunks


def load_incident_records(db_path: Path = DB_PATH) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT incident_id, machine_id, date, failure_type, description FROM failure_incidents"
    ).fetchall()
    conn.close()

    records = []
    for r in rows:
        records.append({
            "id": f"incident-{r['incident_id']}",
            "text": r["description"],
            "metadata": {
                "source": "failure_incidents table",
                "incident_id": int(r["incident_id"]),
                "machine_id": r["machine_id"],
                "date": str(r["date"]),
                "failure_type": r["failure_type"],
            },
        })
    return records


if __name__ == "__main__":
    manual_chunks = load_manual_chunks()
    print(f"Manual chunks: {len(manual_chunks)} from {len({c['metadata']['source'] for c in manual_chunks})} documents")
    for c in manual_chunks[:3]:
        print(" ", c["id"], "|", c["metadata"]["failure_type"])
    if DB_PATH.exists():
        print(f"Incident records: {len(load_incident_records())}")
    else:
        print(f"(no database at {DB_PATH} - run Phase 1 first)")
