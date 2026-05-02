#!/usr/bin/env python3
"""Hybrid retrieval with FAISS semantic search and BM25 keyword search."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import faiss
except ImportError as exc:
    raise ImportError("faiss is not installed. Install it with: pip install faiss-cpu") from exc

try:
    from openai import OpenAI
except ImportError as exc:
    raise ImportError("openai is not installed. Install it with: pip install openai") from exc


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def get_client() -> OpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI()


def embed_query(query: str, model: str) -> np.ndarray:
    response = get_client().embeddings.create(model=model, input=[query])
    query_vector = np.asarray([response.data[0].embedding], dtype="float32")
    faiss.normalize_L2(query_vector)
    return query_vector


def load_index(index_dir: Path, course_id: str, level: str) -> Tuple[Any, Dict[str, Any]]:
    index_path = index_dir / f"{course_id}_{level}.faiss"
    metadata_path = index_dir / f"{course_id}_{level}_metadata.json"

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    return faiss.read_index(str(index_path)), load_json(metadata_path)


def item_text(item: Dict[str, Any]) -> str:
    document = item.get("document") or {}
    return str(document.get("text") or "")


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def build_bm25_stats(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    tokenized_docs = [tokenize(item_text(item)) for item in items]
    doc_count = len(tokenized_docs)
    avg_doc_len = sum(len(doc) for doc in tokenized_docs) / max(doc_count, 1)

    doc_freq: Dict[str, int] = {}
    for doc in tokenized_docs:
        for token in set(doc):
            doc_freq[token] = doc_freq.get(token, 0) + 1

    return {
        "tokenized_docs": tokenized_docs,
        "doc_count": doc_count,
        "avg_doc_len": avg_doc_len,
        "doc_freq": doc_freq,
    }


def bm25_score(
    query_tokens: List[str],
    doc_tokens: List[str],
    *,
    doc_freq: Dict[str, int],
    doc_count: int,
    avg_doc_len: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0

    score = 0.0
    doc_len = len(doc_tokens)
    term_counts: Dict[str, int] = {}

    for token in doc_tokens:
        term_counts[token] = term_counts.get(token, 0) + 1

    for token in query_tokens:
        if token not in term_counts:
            continue

        df = doc_freq.get(token, 0)
        idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
        tf = term_counts[token]
        denom = tf + k1 * (1 - b + b * doc_len / max(avg_doc_len, 1e-9))
        score += idf * (tf * (k1 + 1)) / denom

    return score


def faiss_search(
    *,
    query_vector: np.ndarray,
    index: Any,
    metadata: Dict[str, Any],
    level: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    scores, row_ids = index.search(query_vector, top_k)
    items = metadata.get("items", [])
    results: List[Dict[str, Any]] = []

    for rank, (score, row_id) in enumerate(zip(scores[0], row_ids[0]), start=1):
        if row_id < 0 or row_id >= len(items):
            continue

        item = items[row_id]
        results.append(
            {
                "key": f"{level}:{item.get('id')}",
                "level": level,
                "rank": rank,
                "score": float(score),
                "item": item,
            }
        )

    return results


def bm25_search(
    *,
    query: str,
    metadata: Dict[str, Any],
    level: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    items = metadata.get("items", [])
    stats = build_bm25_stats(items)
    query_tokens = tokenize(query)

    scored: List[Dict[str, Any]] = []
    for row_id, item in enumerate(items):
        score = bm25_score(
            query_tokens,
            stats["tokenized_docs"][row_id],
            doc_freq=stats["doc_freq"],
            doc_count=stats["doc_count"],
            avg_doc_len=stats["avg_doc_len"],
        )
        if score <= 0:
            continue

        scored.append(
            {
                "key": f"{level}:{item.get('id')}",
                "level": level,
                "score": float(score),
                "item": item,
            }
        )

    scored.sort(key=lambda result: result["score"], reverse=True)

    for rank, result in enumerate(scored[:top_k], start=1):
        result["rank"] = rank

    return scored[:top_k]


def add_result(
    results: Dict[str, Dict[str, Any]],
    *,
    source: str,
    result: Dict[str, Any],
    rrf_k: int,
    source_weight: float,
) -> None:
    key = result["key"]

    if key not in results:
        results[key] = {
            "level": result["level"],
            "id": result["item"].get("id"),
            "item": result["item"],
            "faiss_rank": None,
            "faiss_score": None,
            "bm25_rank": None,
            "bm25_score": None,
            "combined_score": 0.0,
        }

    entry = results[key]
    rank = result["rank"]
    score = result["score"]

    if source == "faiss":
        entry["faiss_rank"] = rank
        entry["faiss_score"] = score
    elif source == "bm25":
        entry["bm25_rank"] = rank
        entry["bm25_score"] = score

    entry["combined_score"] += source_weight / (rrf_k + rank)


def combine_results(
    *,
    faiss_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    top_k: int,
    rrf_k: int,
    faiss_weight: float,
    bm25_weight: float,
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}

    for result in faiss_results:
        add_result(
            merged,
            source="faiss",
            result=result,
            rrf_k=rrf_k,
            source_weight=faiss_weight,
        )

    for result in bm25_results:
        add_result(
            merged,
            source="bm25",
            result=result,
            rrf_k=rrf_k,
            source_weight=bm25_weight,
        )

    final_results = list(merged.values())
    final_results.sort(key=lambda result: result["combined_score"], reverse=True)
    return final_results[:top_k]


def format_output_result(result: Dict[str, Any]) -> Dict[str, Any]:
    item = result.get("item") or {}
    metadata = item.get("metadata") or {}
    document = item.get("document") or {}

    return {
        "level": result.get("level"),
        "combined_score": result.get("combined_score"),
        "faiss_rank": result.get("faiss_rank"),
        "faiss_score": result.get("faiss_score"),
        "bm25_rank": result.get("bm25_rank"),
        "bm25_score": result.get("bm25_score"),
        "id": item.get("id"),
        "doc_id": metadata.get("doc_id"),
        "page_no": metadata.get("page_no"),
        "page_start": metadata.get("page_start"),
        "page_end": metadata.get("page_end"),
        "chunk_type": metadata.get("chunk_type", result.get("level")),
        "text": document.get("text"),
        "metadata": metadata,
        "content_for_generation": document.get("content_for_generation"),
    }


def compact_text(text: str | None, limit: int = 260) -> str:
    if not text:
        return ""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def location(result: Dict[str, Any]) -> str:
    if result.get("level") == "semantic":
        return f"{result.get('doc_id')} p{result.get('page_start')}-{result.get('page_end')}"
    return f"{result.get('doc_id')} p{result.get('page_no')}"


def print_results(results: List[Dict[str, Any]]) -> None:
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. [{result['level']}] combined={result['combined_score']:.4f} "
            f"faiss_rank={result.get('faiss_rank')} bm25_rank={result.get('bm25_rank')} "
            f"id={result['id']} type={result.get('chunk_type')} {location(result)}"
        )
        print(f"   text: {compact_text(result.get('text'))}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course-id", default="5703")
    parser.add_argument("--index-dir", default="data/retrieval")
    parser.add_argument("--query", required=True)
    parser.add_argument("--target", choices=["atomic", "semantic", "both"], default="both")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--faiss-weight", type=float, default=1.0)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--output-json", help="Optional path to save retrieval results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.candidate_k < args.top_k:
        raise ValueError("--candidate-k must be >= --top-k")

    index_dir = Path(args.index_dir)
    query_vector = embed_query(args.query, args.embedding_model)
    levels = ["atomic", "semantic"] if args.target == "both" else [args.target]

    all_faiss_results: List[Dict[str, Any]] = []
    all_bm25_results: List[Dict[str, Any]] = []

    for level in levels:
        index, metadata = load_index(index_dir, args.course_id, level)

        all_faiss_results.extend(
            faiss_search(
                query_vector=query_vector,
                index=index,
                metadata=metadata,
                level=level,
                top_k=args.candidate_k,
            )
        )

        all_bm25_results.extend(
            bm25_search(
                query=args.query,
                metadata=metadata,
                level=level,
                top_k=args.candidate_k,
            )
        )

    combined = combine_results(
        faiss_results=all_faiss_results,
        bm25_results=all_bm25_results,
        top_k=args.top_k,
        rrf_k=args.rrf_k,
        faiss_weight=args.faiss_weight,
        bm25_weight=args.bm25_weight,
    )

    output_results = [format_output_result(result) for result in combined]

    print(f"Query: {args.query}")
    print_results(output_results)

    if args.output_json:
        save_json(
            Path(args.output_json),
            {
                "course_id": args.course_id,
                "query": args.query,
                "target": args.target,
                "top_k": args.top_k,
                "candidate_k": args.candidate_k,
                "method": "hybrid_faiss_bm25_rrf",
                "results": output_results,
            },
        )
        print(f"Wrote hybrid retrieval results to {args.output_json}")


if __name__ == "__main__":
    main()
