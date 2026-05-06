#!/usr/bin/env python3
"""Evaluate ADL retrieval on a gold question-evidence-answer set."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None


DEFAULT_COURSE_ID = "adl"
DEFAULT_EVAL_JSON = "data/test/adl_retrieval_eval_40.json"
DEFAULT_TARGET = "both"
DEFAULT_METHOD = "dense_rerank"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CANDIDATE_K = 4
DEFAULT_RRF_K = 60
CUTOFFS = (2, 3, 4)
DEFAULT_DENSE_POOL_MULTIPLIER = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course-id", default=DEFAULT_COURSE_ID)
    parser.add_argument("--eval-json", default=DEFAULT_EVAL_JSON)
    parser.add_argument(
        "--target",
        choices=["atomic", "semantic", "both"],
        default=DEFAULT_TARGET,
        help=f"Which corpus to evaluate. Default: {DEFAULT_TARGET}",
    )
    parser.add_argument(
        "--method",
        choices=["bm25", "dense", "hybrid", "dense_rerank"],
        default=DEFAULT_METHOD,
        help=f"Retrieval method. Default: {DEFAULT_METHOD}",
    )
    parser.add_argument(
        "--atomic-embeddings-path",
        help="Optional explicit path to *_atomic_embeddings.json.",
    )
    parser.add_argument(
        "--semantic-embeddings-path",
        help="Optional explicit path to *_semantic_embeddings.json.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Query embedding model for dense retrieval. Default: {DEFAULT_EMBEDDING_MODEL}",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=DEFAULT_CANDIDATE_K,
        help=f"How many candidates to retrieve before evaluation. Default: {DEFAULT_CANDIDATE_K}",
    )
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument("--faiss-weight", type=float, default=1.0)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--dense-rerank-dense-weight", type=float, default=0.65)
    parser.add_argument("--dense-rerank-bm25-weight", type=float, default=0.35)
    parser.add_argument("--dense-rerank-semantic-bonus", type=float, default=0.18)
    parser.add_argument("--dense-rerank-atomic-bonus", type=float, default=0.15)
    parser.add_argument("--dense-rerank-formula-bonus", type=float, default=0.12)
    parser.add_argument("--dense-rerank-figure-bonus", type=float, default=0.12)
    parser.add_argument("--dense-rerank-text-bonus", type=float, default=0.01)
    parser.add_argument("--dense-rerank-same-page-penalty", type=float, default=0.10)
    parser.add_argument(
        "--dense-pool-multiplier",
        type=int,
        default=DEFAULT_DENSE_POOL_MULTIPLIER,
        help=(
            "For dense_rerank, retrieve this many multiples of candidate-k from dense search "
            "before reranking. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print top hits for every query.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def resolve_atomic_embeddings_path(course_id: str, explicit_path: str | None) -> Path:
    if explicit_path:
        return Path(explicit_path)
    return Path("data/chunk") / f"{course_id}_atomic_embeddings.json"


def resolve_semantic_embeddings_path(course_id: str, explicit_path: str | None) -> Path:
    if explicit_path:
        return Path(explicit_path)
    return Path("data/chunk") / f"{course_id}_semantic_embeddings.json"


def embedding_payload_to_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    for vector in payload.get("vectors") or []:
        items.append(
            {
                "id": vector.get("id"),
                "metadata": vector.get("metadata") or {},
                "document": vector.get("document") or {},
            }
        )
    return {"items": items}


def init_openai_client() -> Any:
    if OpenAI is None or not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for dense or hybrid retrieval.")
    return OpenAI()


def embed_query(client: Any, query: str, model: str) -> List[float]:
    response = client.embeddings.create(model=model, input=[query])
    return list(response.data[0].embedding)


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(a: Sequence[float]) -> float:
    return math.sqrt(dot(a, a))


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    denom = norm(a) * norm(b)
    if denom == 0:
        return 0.0
    return dot(a, b) / denom


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

    scored.sort(key=lambda item: item["score"], reverse=True)
    for rank, item in enumerate(scored[:top_k], start=1):
        item["rank"] = rank
    return scored[:top_k]


def bm25_scores_for_items(
    *,
    query: str,
    items: Sequence[Dict[str, Any]],
) -> Dict[str, float]:
    stats = build_bm25_stats(list(items))
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
            "dense_rank": None,
            "dense_score": None,
            "bm25_rank": None,
            "bm25_score": None,
            "combined_score": 0.0,
        }

    entry = results[key]
    rank = result["rank"]
    score = result["score"]
    if source == "dense":
        entry["dense_rank"] = rank
        entry["dense_score"] = score
    elif source == "bm25":
        entry["bm25_rank"] = rank
        entry["bm25_score"] = score

    entry["combined_score"] += source_weight / (rrf_k + rank)


def combine_results(
    *,
    dense_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    top_k: int,
    rrf_k: int,
    dense_weight: float,
    bm25_weight: float,
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for result in dense_results:
        add_result(
            merged,
            source="dense",
            result=result,
            rrf_k=rrf_k,
            source_weight=dense_weight,
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
    final_results.sort(key=lambda item: item["combined_score"], reverse=True)
    return final_results[:top_k]


def merge_candidate_results(
    *,
    dense_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    rrf_k: int,
    dense_weight: float,
    bm25_weight: float,
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for result in dense_results:
        add_result(
            merged,
            source="dense",
            result=result,
            rrf_k=rrf_k,
            source_weight=dense_weight,
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
    final_results.sort(
        key=lambda item: (
            -(item.get("combined_score") or 0.0),
            item.get("dense_rank") is None,
            item.get("bm25_rank") is None,
        )
    )
    return final_results


def format_output_result(result: Dict[str, Any]) -> Dict[str, Any]:
    item = result.get("item") or {}
    metadata = item.get("metadata") or {}
    document = item.get("document") or {}
    return {
        "level": result.get("level"),
        "combined_score": result.get("combined_score", result.get("score")),
        "dense_rank": result.get("dense_rank", result.get("rank")),
        "dense_score": result.get("dense_score", result.get("score")),
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


def dense_search(
    *,
    query_embedding: Sequence[float],
    vectors: Sequence[Dict[str, Any]],
    level: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    for vector in vectors:
        values = vector.get("values") or []
        if not values:
            continue
        scored.append(
            {
                "key": f"{level}:{vector.get('id')}",
                "level": level,
                "score": float(cosine_similarity(query_embedding, values)),
                "item": {
                    "id": vector.get("id"),
                    "metadata": vector.get("metadata") or {},
                    "document": vector.get("document") or {},
                },
            }
        )
    scored.sort(key=lambda item: item["score"], reverse=True)
    for rank, item in enumerate(scored[:top_k], start=1):
        item["rank"] = rank
    return scored[:top_k]


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


def normalize_whitespace(text: Any) -> str:
    return " ".join(str(text or "").split())


def low_information_text_penalty(item: Dict[str, Any], prefs: Dict[str, bool]) -> float:
    metadata = item.get("metadata") or {}
    chunk_type = str(metadata.get("chunk_type") or "")
    if chunk_type != "text":
        return 0.0

    document = item.get("document") or {}
    content = document.get("content_for_generation") or {}
    text = normalize_whitespace(content.get("text") or content.get("text_cleaned") or document.get("text"))
    section_title = normalize_whitespace(content.get("section_title"))
    lowered = text.lower()

    penalty = 0.0
    generic_prefixes = (
        "outline.",
        "outline",
        "we have",
        "this means",
        "for example",
        "back to",
        "note:",
        "similarly,",
        "and",
        "or",
    )

    if len(text.split()) <= 3:
        penalty += 0.16
    elif len(text.split()) <= 6:
        penalty += 0.08

    if any(lowered.startswith(prefix) for prefix in generic_prefixes):
        penalty += 0.14

    if section_title:
        section_lower = section_title.lower()
        if lowered == section_lower:
            penalty += 0.18
        elif lowered.startswith(section_lower):
            remainder = normalize_whitespace(text[len(section_title):])
            if len(remainder.split()) <= 4:
                penalty += 0.12

    if prefs["prefer_formula"] or prefs["prefer_figure"]:
        if len(text.split()) <= 8:
            penalty += 0.08

    return penalty


def query_mismatch_penalty(item: Dict[str, Any], query: str, prefs: Dict[str, bool]) -> float:
    normalized_query = query.lower()
    document = item.get("document") or {}
    text = normalize_whitespace(document.get("text")).lower()
    if not text:
        return 0.0

    penalty = 0.0
    if prefs["prefer_figure"] and "regularized" not in normalized_query and "regularized" in text:
        penalty += 0.18
    if prefs["prefer_formula"] and "regularized" not in normalized_query and "regularized" in text:
        penalty += 0.12
    return penalty


def query_preferences(query: str) -> Dict[str, bool]:
    normalized = query.lower()
    formula_tokens = {
        "formula",
        "equation",
        "objective",
        "loss",
        "cost",
        "likelihood",
        "log-likelihood",
        "log likelihood",
        "sigmoid",
        "probability",
        "probabilities",
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
        "graph",
        "computational graph",
        "architecture",
        "illustrated",
        "shown",
        "visual",
        "pipeline",
        "flow",
        "flow of data",
        "network",
        "chart",
        "plot",
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
    candidate_k: int,
    target: str,
    dense_weight: float,
    bm25_weight: float,
    semantic_bonus: float,
    atomic_bonus: float,
    formula_bonus: float,
    figure_bonus: float,
    text_bonus: float,
    same_page_penalty: float,
) -> List[Dict[str, Any]]:
    dense_score_map = {
        str(result["item"].get("id")): float(result.get("dense_score", result.get("score", 0.0)) or 0.0)
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

    while remaining and len(selected) < candidate_k:
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
                    score += semantic_bonus
                if "atomic" not in seen_levels and level == "atomic":
                    score += atomic_bonus

            if prefs["prefer_formula"] and chunk_type == "formula":
                score += formula_bonus + 0.16
            elif prefs["prefer_formula"] and chunk_type == "text_inline_math":
                score += formula_bonus + 0.08
            if prefs["prefer_figure"] and chunk_type == "figure":
                score += figure_bonus + 0.20

            if chunk_type == "text":
                score += text_bonus
            if level == "semantic":
                score += 0.02

            score -= low_information_text_penalty(item, prefs)
            score -= query_mismatch_penalty(item, query, prefs)

            signature = page_signature(format_output_result(result))
            if signature in seen_pages:
                score -= same_page_penalty

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
                "score": float(result.get("dense_score", result.get("score", 0.0)) or 0.0),
                "rank": rank,
                "dense_rank": result.get("dense_rank", result.get("rank")),
                "dense_score": float(result.get("dense_score", result.get("score", 0.0)) or 0.0),
                "bm25_rank": result.get("bm25_rank"),
                "bm25_score": result.get("bm25_score", bm25_score_map.get(item_id, 0.0)),
                "combined_score": (
                    dense_weight * normalized_dense.get(item_id, 0.0)
                    + bm25_weight * normalized_bm25.get(item_id, 0.0)
                ),
            }
        )
    return reranked


def ensure_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def normalize_gold_item(raw_item: Dict[str, Any]) -> Dict[str, Any]:
    gold_doc_ids = ensure_list(raw_item.get("gold_doc_id"))
    gold_page_nos = ensure_list(raw_item.get("gold_page_no"))
    gold_chunk_ids = ensure_list(raw_item.get("gold_chunk_id"))
    gold_chunk_texts = ensure_list(raw_item.get("gold_chunk_text"))

    return {
        **raw_item,
        "gold_doc_ids": [str(value) for value in gold_doc_ids],
        "gold_page_nos": [int(value) for value in gold_page_nos if value is not None],
        "gold_chunk_ids": [str(value) for value in gold_chunk_ids],
        "gold_chunk_texts": [str(value) for value in gold_chunk_texts],
    }


def result_atomic_chunk_ids(result: Dict[str, Any]) -> set[str]:
    metadata = result.get("metadata") or {}
    atomic_chunk_ids = metadata.get("atomic_chunk_ids")
    if isinstance(atomic_chunk_ids, list):
        return {str(value) for value in atomic_chunk_ids}

    content = result.get("content_for_generation") or {}
    atomic_chunks = content.get("atomic_chunks") if isinstance(content, dict) else None
    if isinstance(atomic_chunks, list):
        return {
            str(chunk.get("chunk_id"))
            for chunk in atomic_chunks
            if isinstance(chunk, dict) and chunk.get("chunk_id")
        }
    return set()


def matched_gold_chunk_ids(result: Dict[str, Any], gold_item: Dict[str, Any]) -> set[str]:
    result_id = str(result.get("id"))
    gold_chunk_ids = set(gold_item["gold_chunk_ids"])
    if result_id in gold_chunk_ids:
        return {result_id}

    atomic_chunk_ids = result_atomic_chunk_ids(result)
    matched_by_atomic = atomic_chunk_ids & gold_chunk_ids
    if matched_by_atomic:
        return matched_by_atomic

    if result.get("level") == "atomic":
        return set()

    result_doc_id = result.get("doc_id")
    gold_doc_ids = gold_item["gold_doc_ids"]
    gold_page_nos = gold_item["gold_page_nos"]
    if not gold_doc_ids or not gold_page_nos:
        return set()

    if result_doc_id not in set(gold_doc_ids):
        return set()

    start = result.get("page_start")
    end = result.get("page_end")
    if start is None:
        return set()
    if end is None:
        end = start
    matches: set[str] = set()
    gold_chunk_id_list = gold_item["gold_chunk_ids"]
    for doc_id, page_no, chunk_id in zip(gold_doc_ids, gold_page_nos, gold_chunk_id_list):
        if doc_id == result_doc_id and start <= page_no <= end:
            matches.add(chunk_id)
    return matches


def is_relevant(result: Dict[str, Any], gold_item: Dict[str, Any]) -> bool:
    return bool(matched_gold_chunk_ids(result, gold_item))


def novelty_relevance(results: Sequence[Dict[str, Any]], gold_item: Dict[str, Any], limit: int) -> List[int]:
    seen_gold_ids: set[str] = set()
    gains: List[int] = []
    for result in results[:limit]:
        matched = matched_gold_chunk_ids(result, gold_item) - seen_gold_ids
        if matched:
            gains.append(1)
            seen_gold_ids.update(matched)
        else:
            gains.append(0)
    return gains


def reciprocal_rank_at_k(relevance: Sequence[int], k: int) -> float:
    for rank, rel in enumerate(relevance[:k], start=1):
        if rel:
            return 1.0 / rank
    return 0.0


def dcg_at_k(relevance: Sequence[int], k: int) -> float:
    total = 0.0
    for rank, rel in enumerate(relevance[:k], start=1):
        if not rel:
            continue
        total += rel / math.log2(rank + 1)
    return total


def ndcg_at_k(relevance: Sequence[int], gold_count: int, k: int) -> float:
    if gold_count <= 0:
        return 0.0
    ideal_relevance = [1] * min(gold_count, k)
    ideal_dcg = dcg_at_k(ideal_relevance, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(relevance, k) / ideal_dcg


def precision_at_k(relevance: Sequence[int], k: int) -> float:
    return sum(relevance[:k]) / k


def recall_at_k(relevance: Sequence[int], gold_count: int, k: int) -> float:
    if gold_count <= 0:
        return 0.0
    return sum(relevance[:k]) / gold_count


def compact_text(text: str | None, limit: int = 160) -> str:
    if not text:
        return ""
    normalized = " ".join(str(text).split())
    return normalized if len(normalized) <= limit else normalized[: limit - 3] + "..."


def print_query_hits(query: str, results: Sequence[Dict[str, Any]], gold_item: Dict[str, Any], limit: int = 5) -> None:
    print(f"\nQuery: {query}")
    print(f"Gold chunk ids: {gold_item['gold_chunk_ids']}")
    for rank, result in enumerate(results[:limit], start=1):
        rel = "Y" if is_relevant(result, gold_item) else "N"
        page = result.get("page_no") if result.get("level") == "atomic" else f"{result.get('page_start')}-{result.get('page_end')}"
        print(
            f"{rank}. rel={rel} level={result.get('level')} id={result.get('id')} "
            f"{result.get('doc_id')} p{page} score={result.get('combined_score', result.get('score', 0.0)):.4f}"
        )
        print(f"   {compact_text(result.get('text'))}")


def retrieve_for_query(
    *,
    query: str,
    target: str,
    method: str,
    candidate_k: int,
    embedding_model: str,
    rrf_k: int,
    faiss_weight: float,
    bm25_weight: float,
    dense_pool_multiplier: int,
    dense_rerank_dense_weight: float,
    dense_rerank_bm25_weight: float,
    dense_rerank_semantic_bonus: float,
    dense_rerank_atomic_bonus: float,
    dense_rerank_formula_bonus: float,
    dense_rerank_figure_bonus: float,
    dense_rerank_text_bonus: float,
    dense_rerank_same_page_penalty: float,
    target_payloads: Dict[str, Dict[str, Any]],
    query_client: Any | None,
) -> List[Dict[str, Any]]:
    levels = ["atomic", "semantic"] if target == "both" else [target]

    all_dense_results: List[Dict[str, Any]] = []
    all_bm25_results: List[Dict[str, Any]] = []

    query_embedding: Sequence[float] | None = None
    if method in {"dense", "hybrid", "dense_rerank"}:
        if query_client is None:
            raise RuntimeError("Dense or hybrid retrieval requires OPENAI_API_KEY for query embeddings.")
        query_embedding = embed_query(query_client, query, embedding_model)

    dense_top_k = candidate_k
    if method == "dense_rerank":
        dense_top_k = max(candidate_k * max(dense_pool_multiplier, 1), candidate_k)

    for level in levels:
        payload = target_payloads[level]
        vectors = payload.get("vectors") or []
        metadata = embedding_payload_to_metadata(payload)

        if method in {"dense", "hybrid", "dense_rerank"} and query_embedding is not None:
            all_dense_results.extend(
                dense_search(
                    query_embedding=query_embedding,
                    vectors=vectors,
                    level=level,
                    top_k=dense_top_k,
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
        all_bm25_results.sort(key=lambda item: item["score"], reverse=True)
        return [format_output_result(item) for item in all_bm25_results[:candidate_k]]

    if method == "dense":
        all_dense_results.sort(key=lambda item: item["score"], reverse=True)
        return [format_output_result(item) for item in all_dense_results[:candidate_k]]

    if method == "dense_rerank":
        all_dense_results.sort(
            key=lambda item: (
                -item["score"],
                level_priority(item.get("level")),
            )
        )
        merged_candidates = merge_candidate_results(
            dense_results=all_dense_results[:dense_top_k],
            bm25_results=all_bm25_results[:candidate_k],
            rrf_k=rrf_k,
            dense_weight=1.0,
            bm25_weight=1.0,
        )
        reranked = diversity_rerank(
            query=query,
            dense_candidates=merged_candidates,
            candidate_k=candidate_k,
            target=target,
            dense_weight=dense_rerank_dense_weight,
            bm25_weight=dense_rerank_bm25_weight,
            semantic_bonus=dense_rerank_semantic_bonus,
            atomic_bonus=dense_rerank_atomic_bonus,
            formula_bonus=dense_rerank_formula_bonus,
            figure_bonus=dense_rerank_figure_bonus,
            text_bonus=dense_rerank_text_bonus,
            same_page_penalty=dense_rerank_same_page_penalty,
        )
        return [format_output_result(item) for item in reranked]

    combined = combine_results(
        dense_results=all_dense_results,
        bm25_results=all_bm25_results,
        top_k=candidate_k,
        rrf_k=rrf_k,
        dense_weight=faiss_weight,
        bm25_weight=bm25_weight,
    )
    return [format_output_result(item) for item in combined]


def main() -> None:
    args = parse_args()
    max_cutoff = max(CUTOFFS)
    if args.candidate_k < max_cutoff:
        raise ValueError(f"--candidate-k must be >= {max_cutoff}")

    eval_items = [normalize_gold_item(item) for item in load_json(Path(args.eval_json))]
    if not eval_items:
        raise ValueError(f"No eval items found in {args.eval_json}")

    target_payloads: Dict[str, Dict[str, Any]] = {}
    if args.target in {"atomic", "both"}:
        atomic_path = resolve_atomic_embeddings_path(args.course_id, args.atomic_embeddings_path)
        if not atomic_path.exists():
            raise FileNotFoundError(f"Atomic embeddings file not found: {atomic_path}")
        target_payloads["atomic"] = load_json(atomic_path)

    if args.target in {"semantic", "both"}:
        semantic_path = resolve_semantic_embeddings_path(args.course_id, args.semantic_embeddings_path)
        if not semantic_path.exists():
            raise FileNotFoundError(f"Semantic embeddings file not found: {semantic_path}")
        target_payloads["semantic"] = load_json(semantic_path)

    query_client = init_openai_client() if args.method in {"dense", "hybrid", "dense_rerank"} else None

    totals = {
        cutoff: {"recall": 0.0, "precision": 0.0, "mrr": 0.0, "ndcg": 0.0}
        for cutoff in CUTOFFS
    }

    for item in eval_items:
        results = retrieve_for_query(
            query=item["question"],
            target=args.target,
            method=args.method,
            candidate_k=args.candidate_k,
            embedding_model=args.embedding_model,
            rrf_k=args.rrf_k,
            faiss_weight=args.faiss_weight,
            bm25_weight=args.bm25_weight,
            dense_pool_multiplier=args.dense_pool_multiplier,
            dense_rerank_dense_weight=args.dense_rerank_dense_weight,
            dense_rerank_bm25_weight=args.dense_rerank_bm25_weight,
            dense_rerank_semantic_bonus=args.dense_rerank_semantic_bonus,
            dense_rerank_atomic_bonus=args.dense_rerank_atomic_bonus,
            dense_rerank_formula_bonus=args.dense_rerank_formula_bonus,
            dense_rerank_figure_bonus=args.dense_rerank_figure_bonus,
            dense_rerank_text_bonus=args.dense_rerank_text_bonus,
            dense_rerank_same_page_penalty=args.dense_rerank_same_page_penalty,
            target_payloads=target_payloads,
            query_client=query_client,
        )

        relevance = novelty_relevance(results, item, args.candidate_k)
        gold_count = len(item["gold_chunk_ids"])

        if args.verbose:
            print_query_hits(item["question"], results, item, limit=max_cutoff)

        for cutoff in CUTOFFS:
            totals[cutoff]["recall"] += recall_at_k(relevance, gold_count, cutoff)
            totals[cutoff]["precision"] += precision_at_k(relevance, cutoff)
            totals[cutoff]["mrr"] += reciprocal_rank_at_k(relevance, cutoff)
            totals[cutoff]["ndcg"] += ndcg_at_k(relevance, gold_count, cutoff)

    query_count = len(eval_items)
    print(
        f"Evaluated {query_count} queries | target={args.target} | method={args.method} | eval_json={args.eval_json}"
    )
    print("")
    print("k\trecall\tprecision\tmrr\tndcg")
    for cutoff in CUTOFFS:
        print(
            f"{cutoff}\t"
            f"{totals[cutoff]['recall'] / query_count:.4f}\t"
            f"{totals[cutoff]['precision'] / query_count:.4f}\t"
            f"{totals[cutoff]['mrr'] / query_count:.4f}\t"
            f"{totals[cutoff]['ndcg'] / query_count:.4f}"
        )


if __name__ == "__main__":
    main()
