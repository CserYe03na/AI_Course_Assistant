#!/usr/bin/env python3
"""Build FAISS indexes from atomic and semantic embedding JSON files."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    import faiss
except ImportError as exc:
    raise ImportError("faiss is not installed. Install it with: pip install faiss-cpu") from exc


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def load_vectors(path: Path) -> tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    payload = load_json(path)
    rows: List[List[float]] = []
    items: List[Dict[str, Any]] = []

    for vector in payload.get("vectors", []):
        values = vector.get("values")
        if not isinstance(values, list) or not values:
            continue

        rows.append(values)
        items.append(
            {
                "id": vector.get("id"),
                "metadata": vector.get("metadata") or {},
                "document": vector.get("document") or {},
            }
        )

    if not rows:
        raise ValueError(f"No usable vectors found in {path}")

    return np.asarray(rows, dtype="float32"), items, payload


def build_faiss_index(matrix: np.ndarray) -> faiss.IndexFlatIP:
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return index


def build_one(
    *,
    course_id: str,
    level: str,
    embeddings_path: Path,
    output_dir: Path,
) -> None:
    matrix, items, source_payload = load_vectors(embeddings_path)
    index = build_faiss_index(matrix)

    index_path = output_dir / f"{course_id}_{level}.faiss"
    metadata_path = output_dir / f"{course_id}_{level}_metadata.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    save_json(
        metadata_path,
        {
            "course_id": course_id,
            "level": level,
            "source_embeddings_path": str(embeddings_path.as_posix()),
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "embedding_model": source_payload.get("embedding_model"),
            "vector_count": len(items),
            "embedding_dim": int(matrix.shape[1]),
            "items": items,
        },
    )

    print(f"Wrote {level} FAISS index to {index_path}")
    print(f"Wrote {level} metadata to {metadata_path}")
    print(f"  vectors={len(items)}, dim={matrix.shape[1]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course-id", default="5703")
    parser.add_argument("--chunk-dir", default="data/chunk")
    parser.add_argument("--output-dir", default="data/retrieval")
    parser.add_argument("--target", choices=["atomic", "semantic", "both"], default="both")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunk_dir = Path(args.chunk_dir)
    output_dir = Path(args.output_dir)

    jobs = []
    if args.target in {"atomic", "both"}:
        jobs.append(("atomic", chunk_dir / f"{args.course_id}_atomic_embeddings.json"))
    if args.target in {"semantic", "both"}:
        jobs.append(("semantic", chunk_dir / f"{args.course_id}_semantic_embeddings.json"))

    for level, path in jobs:
        if not path.exists():
            raise FileNotFoundError(f"Embedding file not found: {path}")
        build_one(
            course_id=args.course_id,
            level=level,
            embeddings_path=path,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
