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


DEFAULT_DENSE_POOL_MULTIPLIER = 4


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


def bm25_scores_for_items(
    *,
    query: str,
    items: List[Dict[str, Any]],
) -> Dict[str, float]:
    stats = build_bm25_stats(items)
    query_tokens = tokenize(query)
    scores: Dict[str, float] = {}

    for row_id, item in enumerate(items):
        item_id = item.get("id")
        if not item_id:
            continue
        scores[str(item_id)] = bm25_score(
            query_tokens,
            stats["tokenized_docs"][row_id],
            doc_freq=stats["doc_freq"],
            doc_count=stats["doc_count"],
            avg_doc_len=stats["avg_doc_len"],
        )

    return scores


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
        "combined_score": result.get("combined_score", result.get("score")),
        "faiss_rank": result.get("faiss_rank", result.get("rank")),
        "faiss_score": result.get("faiss_score", result.get("score")),
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


def level_priority(level: str | None) -> int:
    return 0 if level == "semantic" else 1


def normalize_score_map(score_map: Dict[str, float]) -> Dict[str, float]:
    if not score_map:
        return {}
    values = list(score_map.values())
    max_value = max(values)
    min_value = min(values)
    if math.isclose(max_value, min_value):
        return {key: 1.0 for key in score_map}
    return {
        key: (value - min_value) / (max_value - min_value)
        for key, value in score_map.items()
    }


def query_preferences(query: str) -> Dict[str, bool]:
    normalized = query.lower()
    formula_tokens = {
        "formula",
        "equation",
        "objective",
        "loss",
        "cost",
        "compute",
        "complexity",
        "defined",
        "definition",
        "mse",
        "auc",
        "dcg",
        "ndcg",
        "mrr",
        "precision",
        "recall",
        "flops",
    }
    figure_tokens = {
        "figure",
        "diagram",
        "architecture",
        "illustrated",
        "shown",
        "visual",
        "pipeline",
        "left part",
        "right part",
    }
    return {
        "prefer_formula": any(token in normalized for token in formula_tokens),
        "prefer_figure": any(token in normalized for token in figure_tokens),
    }


def page_signature(result: Dict[str, Any]) -> tuple[Any, Any, Any]:
    return (
        result.get("doc_id"),
        result.get("page_no") if result.get("level") == "atomic" else result.get("page_start"),
        result.get("level"),
    )


def diversity_rerank(
    *,
    query: str,
    dense_candidates: List[Dict[str, Any]],
    top_k: int,
    target: str,
    dense_weight: float,
    bm25_weight: float,
) -> List[Dict[str, Any]]:
    dense_score_map = {
        str(result["item"].get("id")): float(result.get("score", 0.0))
        for result in dense_candidates
        if result.get("item", {}).get("id")
    }
    bm25_score_map = bm25_scores_for_items(
        query=query,
        items=[result["item"] for result in dense_candidates],
    )
    normalized_dense = normalize_score_map(dense_score_map)
    normalized_bm25 = normalize_score_map(bm25_score_map)
    prefs = query_preferences(query)

    remaining = list(dense_candidates)
    selected: List[Dict[str, Any]] = []
    seen_pages: set[tuple[Any, Any, Any]] = set()
    seen_levels: set[str] = set()

    while remaining and len(selected) < top_k:
        best_idx = 0
        best_score = float("-inf")

        for idx, result in enumerate(remaining):
            item = result["item"]
            item_id = str(item.get("id"))
            metadata = item.get("metadata") or {}
            chunk_type = str(metadata.get("chunk_type") or result.get("level") or "")
            level = str(result.get("level") or "")
            score = dense_weight * normalized_dense.get(item_id, 0.0) + bm25_weight * normalized_bm25.get(item_id, 0.0)

            if target == "both":
                if "semantic" not in seen_levels and level == "semantic":
                    score += 0.18
                if "atomic" not in seen_levels and level == "atomic":
                    score += 0.15

            if prefs["prefer_formula"] and chunk_type in {"formula", "text_inline_math"}:
                score += 0.12
            if prefs["prefer_figure"] and chunk_type == "figure":
                score += 0.12

            if chunk_type == "text":
                score += 0.01
            if level == "semantic":
                score += 0.02

            signature = page_signature(format_output_result(result))
            if signature in seen_pages:
                score -= 0.10

            if score > best_score:
                best_score = score
                best_idx = idx

        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        chosen_formatted = format_output_result(chosen)
        seen_pages.add(page_signature(chosen_formatted))
        seen_levels.add(str(chosen.get("level") or ""))

    reranked: List[Dict[str, Any]] = []
    for rank, result in enumerate(selected, start=1):
        item_id = str(result["item"].get("id"))
        reranked.append(
            {
                "level": result["level"],
                "item": result["item"],
                "rank": rank,
                "score": float(result.get("score", 0.0)),
                "faiss_rank": result.get("rank"),
                "faiss_score": float(result.get("score", 0.0)),
                "bm25_rank": None,
                "bm25_score": bm25_score_map.get(item_id, 0.0),
                "combined_score": (
                    dense_weight * normalized_dense.get(item_id, 0.0)
                    + bm25_weight * normalized_bm25.get(item_id, 0.0)
                ),
            }
        )

    return reranked


def retrieve_results(
    *,
    course_id: str,
    index_dir: Path,
    query: str,
    target: str,
    top_k: int,
    candidate_k: int,
    embedding_model: str,
    method: str,
    rrf_k: int,
    faiss_weight: float,
    bm25_weight: float,
    dense_pool_multiplier: int,
    dense_rerank_dense_weight: float,
    dense_rerank_bm25_weight: float,
) -> List[Dict[str, Any]]:
    query_vector: np.ndarray | None = None
    if method in {"dense", "hybrid", "dense_rerank"}:
        query_vector = embed_query(query, embedding_model)
    levels = ["atomic", "semantic"] if target == "both" else [target]

    all_faiss_results: List[Dict[str, Any]] = []
    all_bm25_results: List[Dict[str, Any]] = []

    faiss_top_k = candidate_k
    if method == "dense_rerank":
        faiss_top_k = max(candidate_k * max(dense_pool_multiplier, 1), candidate_k)

    for level in levels:
        index, metadata = load_index(index_dir, course_id, level)

        if method in {"dense", "hybrid", "dense_rerank"}:
            if query_vector is None:
                raise RuntimeError("query_vector is required for dense retrieval methods")
            all_faiss_results.extend(
                faiss_search(
                    query_vector=query_vector,
                    index=index,
                    metadata=metadata,
                    level=level,
                    top_k=faiss_top_k,
                )
            )

        if method in {"bm25", "hybrid"}:
            all_bm25_results.extend(
                bm25_search(
                    query=query,
                    metadata=metadata,
                    level=level,
                    top_k=candidate_k,
                )
            )

    if method == "bm25":
        all_bm25_results.sort(key=lambda result: result["score"], reverse=True)
        return [format_output_result(result) for result in all_bm25_results[:top_k]]

    if method == "dense":
        all_faiss_results.sort(
            key=lambda result: (
                -result["score"],
                level_priority(result.get("level")),
            )
        )
        return [format_output_result(result) for result in all_faiss_results[:top_k]]

    if method == "dense_rerank":
        all_faiss_results.sort(
            key=lambda result: (
                -result["score"],
                level_priority(result.get("level")),
            )
        )
        reranked = diversity_rerank(
            query=query,
            dense_candidates=all_faiss_results[:faiss_top_k],
            top_k=top_k,
            target=target,
            dense_weight=dense_rerank_dense_weight,
            bm25_weight=dense_rerank_bm25_weight,
        )
        return [format_output_result(result) for result in reranked]

    combined = combine_results(
        faiss_results=all_faiss_results,
        bm25_results=all_bm25_results,
        top_k=top_k,
        rrf_k=rrf_k,
        faiss_weight=faiss_weight,
        bm25_weight=bm25_weight,
    )
    return [format_output_result(result) for result in combined]


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
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--candidate-k", type=int, default=4)
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument(
        "--method",
        choices=["bm25", "dense", "hybrid", "dense_rerank"],
        default="dense_rerank",
    )
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--faiss-weight", type=float, default=1.0)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--dense-rerank-dense-weight", type=float, default=0.65)
    parser.add_argument("--dense-rerank-bm25-weight", type=float, default=0.35)
    parser.add_argument("--dense-pool-multiplier", type=int, default=DEFAULT_DENSE_POOL_MULTIPLIER)
    parser.add_argument("--output-json", help="Optional path to save retrieval results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.candidate_k < args.top_k:
        raise ValueError("--candidate-k must be >= --top-k")

    index_dir = Path(args.index_dir)
    output_results = retrieve_results(
        course_id=args.course_id,
        index_dir=index_dir,
        query=args.query,
        target=args.target,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        embedding_model=args.embedding_model,
        method=args.method,
        rrf_k=args.rrf_k,
        faiss_weight=args.faiss_weight,
        bm25_weight=args.bm25_weight,
        dense_pool_multiplier=args.dense_pool_multiplier,
        dense_rerank_dense_weight=args.dense_rerank_dense_weight,
        dense_rerank_bm25_weight=args.dense_rerank_bm25_weight,
    )

    print(f"Query: {args.query}")
    print_results(output_results)

    if args.output_json:
        save_json(
            Path(args.output_json),
            {
                "course_id": args.course_id,
                "query": args.query,
                "target": args.target,
                "method": args.method,
                "top_k": args.top_k,
                "candidate_k": args.candidate_k,
                "dense_pool_multiplier": args.dense_pool_multiplier,
                "results": output_results,
            },
        )
        print(f"Wrote retrieval results to {args.output_json}")


if __name__ == "__main__":
    main()
