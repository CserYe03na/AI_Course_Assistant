#!/usr/bin/env python3
"""Create embeddings for semantic chunks."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None


DEFAULT_INPUT_GLOB = "data/chunk/*_semantic_chunks.json"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_BATCH_SIZE = 128
DEFAULT_COURSE_ID = "adl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--course-id",
        action="append",
        help="Course identifier, for example adl or eods. Can be passed multiple times.",
    )
    parser.add_argument(
        "--input-path",
        action="append",
        help="Explicit semantic chunk JSON path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--input-glob",
        default=DEFAULT_INPUT_GLOB,
        help=f"Glob used when --input-path is not provided. Default: {DEFAULT_INPUT_GLOB}",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional output directory. Defaults to the same directory as each input file.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Embedding model to use. Default: {DEFAULT_EMBEDDING_MODEL}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of texts per embeddings API call. Default: {DEFAULT_BATCH_SIZE}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed all eligible chunks instead of resuming from an existing output file.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def init_openai_client() -> Any:
    if OpenAI is None or not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OpenAI client is not available")
    return OpenAI()


def resolve_input_paths(explicit_paths: Sequence[str] | None, input_glob: str) -> List[Path]:
    if explicit_paths:
        return [Path(path) for path in explicit_paths]
    return sorted(Path.cwd().glob(input_glob))


def resolve_course_input_paths(course_ids: Sequence[str]) -> List[Path]:
    return [Path("data/chunk") / f"{course_id}_semantic_chunks.json" for course_id in course_ids]


def resolve_output_path(input_path: Path, output_dir: str | None) -> Path:
    if input_path.name.endswith("_semantic_chunks.json"):
        output_name = input_path.name.replace("_semantic_chunks.json", "_semantic_embeddings.json")
    else:
        output_name = f"{input_path.stem}_embeddings.json"
    base_dir = Path(output_dir) if output_dir else input_path.parent
    return base_dir / output_name


def load_existing_output(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = load_json(path)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def chunked(items: Sequence[Dict[str, Any]], size: int) -> Iterable[Sequence[Dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def select_embedding_candidates(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for chunk in payload.get("semantic_chunks", []):
        if not isinstance(chunk, dict):
            continue
        text = chunk.get("content_for_embedding")
        if not isinstance(text, str) or not text.strip():
            continue
        selected.append(chunk)
    return selected


def index_existing_vectors(existing_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for item in existing_payload.get("vectors", []):
        if not isinstance(item, dict):
            continue
        chunk_id = item.get("id")
        if not isinstance(chunk_id, str) or not chunk_id:
            continue
        indexed[chunk_id] = item
    return indexed


def create_embeddings(
    *,
    client: Any,
    model: str,
    candidates: Sequence[Dict[str, Any]],
    batch_size: int,
) -> List[Dict[str, Any]]:
    vectors: List[Dict[str, Any]] = []
    for batch in chunked(candidates, batch_size):
        texts = [chunk["content_for_embedding"] for chunk in batch]
        response = client.embeddings.create(model=model, input=texts)
        for chunk, item in zip(batch, response.data):
            metadata = dict(chunk.get("metadata") or {})
            metadata.update(
                {
                    "semantic_chunk_id": chunk.get("semantic_chunk_id"),
                    "atomic_chunk_ids": chunk.get("atomic_chunk_ids"),
                    "auxiliary_chunk_ids": chunk.get("auxiliary_chunk_ids"),
                }
            )
            vectors.append(
                {
                    "id": chunk.get("semantic_chunk_id"),
                    "values": item.embedding,
                    "metadata": metadata,
                    "document": {
                        "text": chunk.get("content_for_embedding"),
                        "content_for_generation": chunk.get("content_for_generation"),
                    },
                }
            )
    return vectors


def build_output_payload(
    *,
    input_path: Path,
    semantic_chunk_payload: Dict[str, Any],
    embedding_model: str,
    selected_candidates: Sequence[Dict[str, Any]],
    vectors: Sequence[Dict[str, Any]],
    resumed_vector_count: int,
) -> Dict[str, Any]:
    return {
        "source_semantic_chunk_path": str(input_path.as_posix()),
        "course_id": semantic_chunk_payload.get("course_id"),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "embedding_model": embedding_model,
        "candidate_chunk_count": len(selected_candidates),
        "embedded_chunk_count": len(vectors),
        "newly_embedded_chunk_count": len(vectors) - resumed_vector_count,
        "resumed_vector_count": resumed_vector_count,
        "embedding_dim": len(vectors[0]["values"]) if vectors else None,
        "config": semantic_chunk_payload.get("config"),
        "similarity_distribution": semantic_chunk_payload.get("similarity_distribution"),
        "vectors": list(vectors),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    if args.input_path:
        input_paths = resolve_input_paths(args.input_path, args.input_glob)
    elif args.course_id:
        input_paths = resolve_course_input_paths(args.course_id)
    else:
        input_paths = resolve_course_input_paths([DEFAULT_COURSE_ID])

    if not input_paths:
        raise FileNotFoundError("No semantic chunk JSON files found")

    missing_paths = [path for path in input_paths if not path.exists()]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Semantic chunk JSON file(s) not found: {missing}")

    client = init_openai_client()

    for input_path in input_paths:
        payload = load_json(input_path)
        all_candidates = select_embedding_candidates(payload)
        output_path = resolve_output_path(input_path, args.output_dir)
        existing_payload = {} if args.force else load_existing_output(output_path)
        existing_vectors_by_id = {} if args.force else index_existing_vectors(existing_payload)

        candidates = [
            chunk
            for chunk in all_candidates
            if str(chunk.get("semantic_chunk_id")) not in existing_vectors_by_id
        ]
        print(
            f"Embedding {len(candidates)} new semantic chunks from {input_path} "
            f"({len(existing_vectors_by_id)} existing vectors reused)",
            flush=True,
        )
        new_vectors = (
            create_embeddings(
                client=client,
                model=args.embedding_model,
                candidates=candidates,
                batch_size=args.batch_size,
            )
            if candidates
            else []
        )

        new_vectors_by_id = {str(vector.get('id')): vector for vector in new_vectors}
        merged_vectors: List[Dict[str, Any]] = []
        for chunk in all_candidates:
            chunk_id = str(chunk.get("semantic_chunk_id"))
            if chunk_id in existing_vectors_by_id:
                merged_vectors.append(existing_vectors_by_id[chunk_id])
                continue
            if chunk_id in new_vectors_by_id:
                merged_vectors.append(new_vectors_by_id[chunk_id])

        output_payload = build_output_payload(
            input_path=input_path,
            semantic_chunk_payload=payload,
            embedding_model=args.embedding_model,
            selected_candidates=all_candidates,
            vectors=merged_vectors,
            resumed_vector_count=len(existing_vectors_by_id),
        )
        save_json(output_path, output_payload)
        print(f"Wrote {len(merged_vectors)} embeddings to {output_path}", flush=True)


if __name__ == "__main__":
    main()
