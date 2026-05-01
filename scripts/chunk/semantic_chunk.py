#!/usr/bin/env python3
"""Build semantic chunks from atomic chunks and atomic embeddings."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


DEFAULT_COURSE_ID = "adl"
DEFAULT_MIN_TOKENS = 400
DEFAULT_MAX_TOKENS = 800
DEFAULT_THRESHOLD_QUANTILE = 0.45
DEFAULT_HIGH_SIMILARITY_QUANTILE = 0.80
DEFAULT_SECTION_TRANSITION_PENALTY = 0.12
DEFAULT_AUX_HOST_MIN_TOKENS = 60
DEFAULT_SAME_SECTION_REJECT_BELOW = 0.3
DEFAULT_DIFFERENT_SECTION_REJECT_BELOW = 0.8


@dataclass
class AtomicRecord:
    chunk_id: str
    chunk_type: str
    doc_id: str
    page_no: int
    block_no: int
    token_count: int
    content_for_embedding: str
    content_for_generation: Dict[str, Any]
    metadata: Dict[str, Any]
    raw_fields: Dict[str, Any]
    embedding: List[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--course-id",
        default=DEFAULT_COURSE_ID,
        help=f"Course identifier. Default: {DEFAULT_COURSE_ID}",
    )
    parser.add_argument("--atomic-chunks-path", help="Optional explicit *_atomic_chunks.json path.")
    parser.add_argument("--atomic-embeddings-path", help="Optional explicit *_atomic_embeddings.json path.")
    parser.add_argument(
        "--output-dir",
        help="Optional output directory. Defaults to data/chunk.",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=DEFAULT_MIN_TOKENS,
        help=f"Lower bound for dynamic semantic chunk token budget. Default: {DEFAULT_MIN_TOKENS}",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Upper bound for dynamic semantic chunk token budget. Default: {DEFAULT_MAX_TOKENS}",
    )
    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=DEFAULT_THRESHOLD_QUANTILE,
        help=f"Quantile of adjacent similarity scores used as similarity_threshold. Default: {DEFAULT_THRESHOLD_QUANTILE}",
    )
    parser.add_argument(
        "--high-similarity-quantile",
        type=float,
        default=DEFAULT_HIGH_SIMILARITY_QUANTILE,
        help=f"Quantile of adjacent similarity scores used as high_similarity. Default: {DEFAULT_HIGH_SIMILARITY_QUANTILE}",
    )
    parser.add_argument(
        "--section-transition-penalty",
        type=float,
        default=DEFAULT_SECTION_TRANSITION_PENALTY,
        help=f"Extra similarity required when merging across different section titles. Default: {DEFAULT_SECTION_TRANSITION_PENALTY}",
    )
    parser.add_argument(
        "--aux-host-min-tokens",
        type=int,
        default=DEFAULT_AUX_HOST_MIN_TOKENS,
        help=f"Minimum semantic chunk token count required before attaching poor figures. Default: {DEFAULT_AUX_HOST_MIN_TOKENS}",
    )
    parser.add_argument(
        "--same-section-reject-below",
        type=float,
        default=DEFAULT_SAME_SECTION_REJECT_BELOW,
        help=f"Reject same-section merges only when similarity is below this value. Default: {DEFAULT_SAME_SECTION_REJECT_BELOW}",
    )
    parser.add_argument(
        "--different-section-reject-below",
        type=float,
        default=DEFAULT_DIFFERENT_SECTION_REJECT_BELOW,
        help=f"Reject cross-section merges only when similarity is below this value. Default: {DEFAULT_DIFFERENT_SECTION_REJECT_BELOW}",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def resolve_atomic_chunks_path(course_id: str, explicit_path: Optional[str]) -> Path:
    if explicit_path:
        return Path(explicit_path)
    return Path("data/chunk") / f"{course_id}_atomic_chunks.json"


def resolve_atomic_embeddings_path(course_id: str, explicit_path: Optional[str]) -> Path:
    if explicit_path:
        return Path(explicit_path)
    return Path("data/chunk") / f"{course_id}_atomic_embeddings.json"


def resolve_output_path(course_id: str, output_dir: Optional[str]) -> Path:
    base_dir = Path(output_dir) if output_dir else Path("data/chunk")
    return base_dir / f"{course_id}_semantic_chunks.json"


def estimate_token_count(text: str) -> int:
    if not text:
        return 0
    token_like = re.findall(r"\S+", text)
    return len(token_like)


def clean_text(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def parse_block_no(block_id: str) -> int:
    match = re.search(r"_b(\d+)$", block_id)
    if match:
        return int(match.group(1))
    return 10**9


def normalize_section_title(title: Optional[str]) -> Optional[str]:
    cleaned = clean_text(title)
    if not cleaned:
        return None
    cleaned = cleaned.lower()
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or None


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(a: Sequence[float]) -> float:
    return math.sqrt(dot(a, a))


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    denom = norm(a) * norm(b)
    if denom == 0:
        return 0.0
    return dot(a, b) / denom


def average_embedding(vectors: Sequence[Sequence[float]]) -> List[float]:
    if not vectors:
        return []
    dimension = len(vectors[0])
    sums = [0.0] * dimension
    for vector in vectors:
        for index, value in enumerate(vector):
            sums[index] += value
    return [value / len(vectors) for value in sums]


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    q = min(max(q, 0.0), 1.0)
    sorted_values = sorted(float(value) for value in values)
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    weight = position - lower
    return lower_value + (upper_value - lower_value) * weight


def dynamic_token_limit(
    *,
    similarity: float,
    threshold: float,
    high_similarity: float,
    min_tokens: int,
    max_tokens: int,
) -> int:
    if max_tokens <= min_tokens:
        return min_tokens
    if similarity <= threshold:
        return min_tokens
    if similarity >= high_similarity:
        return max_tokens
    span = max(high_similarity - threshold, 1e-6)
    ratio = (similarity - threshold) / span
    return int(round(min_tokens + ratio * (max_tokens - min_tokens)))


def index_atomic_chunks(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(chunk.get("chunk_id")): chunk
        for chunk in payload.get("chunks", [])
        if isinstance(chunk, dict) and chunk.get("chunk_id")
    }


def build_atomic_records(
    *,
    atomic_chunk_payload: Dict[str, Any],
    atomic_embedding_payload: Dict[str, Any],
) -> Tuple[List[AtomicRecord], List[Dict[str, Any]]]:
    chunk_map = index_atomic_chunks(atomic_chunk_payload)
    atomic_records: List[AtomicRecord] = []

    for vector in atomic_embedding_payload.get("vectors", []):
        if not isinstance(vector, dict):
            continue
        chunk_id = str(vector.get("id") or "")
        atomic_chunk = chunk_map.get(chunk_id)
        if not atomic_chunk:
            continue

        metadata = dict(atomic_chunk.get("metadata") or {})
        block_id = str(metadata.get("block_id") or "")
        content_for_embedding = str(atomic_chunk.get("content_for_embedding") or "")
        if not content_for_embedding:
            continue

        atomic_records.append(
            AtomicRecord(
                chunk_id=chunk_id,
                chunk_type=str(atomic_chunk.get("chunk_type") or ""),
                doc_id=str(metadata.get("doc_id") or ""),
                page_no=int(metadata.get("page_no") or 0),
                block_no=parse_block_no(block_id),
                token_count=estimate_token_count(content_for_embedding),
                content_for_embedding=content_for_embedding,
                content_for_generation=dict(atomic_chunk.get("content_for_generation") or {}),
                metadata=metadata,
                raw_fields=dict(atomic_chunk.get("raw_fields") or {}),
                embedding=list(vector.get("values") or []),
            )
        )

    auxiliary_chunks: List[Dict[str, Any]] = []
    for chunk in atomic_chunk_payload.get("chunks", []):
        if not isinstance(chunk, dict):
            continue
        if chunk.get("chunk_type") != "figure":
            continue
        if chunk.get("indexable") is not False:
            continue
        auxiliary_chunks.append(chunk)

    atomic_records.sort(key=lambda item: (item.doc_id, item.page_no, item.block_no, item.chunk_id))
    auxiliary_chunks.sort(
        key=lambda chunk: (
            str((chunk.get("metadata") or {}).get("doc_id") or ""),
            int((chunk.get("metadata") or {}).get("page_no") or 0),
            parse_block_no(str((chunk.get("metadata") or {}).get("block_id") or "")),
            str(chunk.get("chunk_id") or ""),
        )
    )
    return atomic_records, auxiliary_chunks


def adjacent_similarity_scores(records: Sequence[AtomicRecord]) -> List[float]:
    scores: List[float] = []
    for left, right in zip(records, records[1:]):
        if left.doc_id != right.doc_id:
            continue
        if not left.embedding or not right.embedding:
            continue
        scores.append(cosine_similarity(left.embedding, right.embedding))
    return scores


def compute_similarity_thresholds(
    records: Sequence[AtomicRecord],
    *,
    threshold_quantile: float,
    high_similarity_quantile: float,
) -> Tuple[float, float, Dict[str, Any]]:
    scores = adjacent_similarity_scores(records)
    if not scores:
        threshold = 0.0
        high_similarity = 0.0
    else:
        threshold = quantile(scores, threshold_quantile)
        high_similarity = quantile(scores, high_similarity_quantile)
        if high_similarity < threshold:
            high_similarity = threshold

    summary = {
        "adjacent_pair_count": len(scores),
        "min": min(scores) if scores else None,
        "p10": quantile(scores, 0.10) if scores else None,
        "p25": quantile(scores, 0.25) if scores else None,
        "p50": quantile(scores, 0.50) if scores else None,
        "p60": quantile(scores, 0.60) if scores else None,
        "p75": quantile(scores, 0.75) if scores else None,
        "p85": quantile(scores, 0.85) if scores else None,
        "p90": quantile(scores, 0.90) if scores else None,
        "max": max(scores) if scores else None,
    }
    return threshold, high_similarity, summary


def record_section_title(record: AtomicRecord) -> Optional[str]:
    return clean_text((record.content_for_generation or {}).get("section_title"))


def current_chunk_section_title(current_chunk: Dict[str, Any]) -> Optional[str]:
    records: List[AtomicRecord] = current_chunk["atomic_records"]
    for record in reversed(records):
        title = record_section_title(record)
        if title:
            return title
    return None


def start_semantic_chunk(record: AtomicRecord) -> Dict[str, Any]:
    return {
        "doc_id": record.doc_id,
        "page_start": record.page_no,
        "page_end": record.page_no,
        "block_start": record.block_no,
        "block_end": record.block_no,
        "atomic_records": [record],
        "token_count": record.token_count,
        "embeddings": [record.embedding],
        "merge_scores": [],
    }


def should_merge(
    *,
    current_chunk: Dict[str, Any],
    next_record: AtomicRecord,
    min_tokens: int,
    max_tokens: int,
    threshold: float,
    high_similarity: float,
    same_section_reject_below: float,
    different_section_reject_below: float,
) -> Tuple[bool, float, int, float]:
    if current_chunk["doc_id"] != next_record.doc_id:
        return False, 0.0, min_tokens, threshold

    current_avg = average_embedding(current_chunk["embeddings"])
    similarity = cosine_similarity(current_avg, next_record.embedding)
    current_title = normalize_section_title(current_chunk_section_title(current_chunk))
    next_title = normalize_section_title(record_section_title(next_record))
    same_section = not current_title or not next_title or current_title == next_title
    effective_threshold = same_section_reject_below if same_section else different_section_reject_below
    token_limit = dynamic_token_limit(
        similarity=similarity,
        threshold=threshold,
        high_similarity=high_similarity,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
    )
    merged_token_count = current_chunk["token_count"] + next_record.token_count

    can_merge = similarity >= effective_threshold and merged_token_count <= token_limit
    return can_merge, similarity, token_limit, effective_threshold


def finalize_semantic_chunk(current_chunk: Dict[str, Any], ordinal: int) -> Dict[str, Any]:
    records: List[AtomicRecord] = current_chunk["atomic_records"]
    doc_id = records[0].doc_id
    semantic_chunk_id = f"{doc_id}_semantic_{ordinal:04d}"

    atomic_chunks_generation: List[Dict[str, Any]] = []
    for record in records:
        atomic_chunks_generation.append(
            {
                "chunk_id": record.chunk_id,
                "chunk_type": record.chunk_type,
                "content_for_generation": record.content_for_generation,
                "metadata": record.metadata,
                "raw_fields": record.raw_fields,
            }
        )

    return {
        "semantic_chunk_id": semantic_chunk_id,
        "atomic_chunk_ids": [record.chunk_id for record in records],
        "auxiliary_chunk_ids": [],
        "content_for_embedding": " ".join(record.content_for_embedding for record in records).strip(),
        "content_for_generation": {
            "atomic_chunks": atomic_chunks_generation,
            "auxiliary_chunks": [],
        },
        "metadata": {
            "doc_id": doc_id,
            "page_start": current_chunk["page_start"],
            "page_end": current_chunk["page_end"],
            "token_count": current_chunk["token_count"],
            "atomic_chunk_count": len(records),
            "merge_scores": current_chunk["merge_scores"],
        },
    }


def build_semantic_chunks(
    records: Sequence[AtomicRecord],
    *,
    min_tokens: int,
    max_tokens: int,
    threshold: float,
    high_similarity: float,
    same_section_reject_below: float,
    different_section_reject_below: float,
) -> List[Dict[str, Any]]:
    if not records:
        return []

    semantic_chunks: List[Dict[str, Any]] = []
    current_chunk = start_semantic_chunk(records[0])
    ordinal = 1

    for next_record in records[1:]:
        can_merge, similarity, _token_limit, effective_threshold = should_merge(
            current_chunk=current_chunk,
            next_record=next_record,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            threshold=threshold,
            high_similarity=high_similarity,
            same_section_reject_below=same_section_reject_below,
            different_section_reject_below=different_section_reject_below,
        )
        if can_merge:
            current_chunk["atomic_records"].append(next_record)
            current_chunk["token_count"] += next_record.token_count
            current_chunk["embeddings"].append(next_record.embedding)
            current_chunk["page_end"] = next_record.page_no
            current_chunk["block_end"] = next_record.block_no
            current_chunk["merge_scores"].append(
                {
                    "similarity": similarity,
                    "effective_threshold": effective_threshold,
                }
            )
            continue

        semantic_chunks.append(finalize_semantic_chunk(current_chunk, ordinal))
        ordinal += 1
        current_chunk = start_semantic_chunk(next_record)

    semantic_chunks.append(finalize_semantic_chunk(current_chunk, ordinal))
    return semantic_chunks


def semantic_chunk_position_key(chunk: Dict[str, Any]) -> Tuple[int, int]:
    metadata = chunk.get("metadata") or {}
    return int(metadata.get("page_start") or 0), int(metadata.get("page_end") or 0)


def auxiliary_chunk_distance(aux_chunk: Dict[str, Any], semantic_chunk: Dict[str, Any]) -> Tuple[int, int]:
    aux_metadata = aux_chunk.get("metadata") or {}
    sem_metadata = semantic_chunk.get("metadata") or {}
    aux_page = int(aux_metadata.get("page_no") or 0)
    aux_block = parse_block_no(str(aux_metadata.get("block_id") or ""))

    sem_start_page = int(sem_metadata.get("page_start") or 0)
    sem_end_page = int(sem_metadata.get("page_end") or 0)

    if sem_start_page <= aux_page <= sem_end_page:
        page_gap = 0
    else:
        page_gap = min(abs(aux_page - sem_start_page), abs(aux_page - sem_end_page))

    atomic_chunks = (semantic_chunk.get("content_for_generation") or {}).get("atomic_chunks") or []
    if not atomic_chunks:
        return page_gap, aux_block

    block_numbers = [
        parse_block_no(str(((item.get("metadata") or {}).get("block_id") or "")))
        for item in atomic_chunks
    ]
    closest_block_gap = min(abs(aux_block - block_no) for block_no in block_numbers)
    return page_gap, closest_block_gap


def semantic_chunk_text(semantic_chunk: Dict[str, Any]) -> str:
    return str(semantic_chunk.get("content_for_embedding") or "")


def is_obvious_ocr_noise(text: str) -> bool:
    cleaned = clean_text(text)
    if not cleaned:
        return True

    raw_tokens = re.findall(r"\S+", cleaned)
    if len(raw_tokens) < 6:
        return False

    alnum_tokens = re.findall(r"[A-Za-z0-9']+", cleaned.lower())
    suspicious_count = 0
    for token in alnum_tokens:
        alpha = re.sub(r"[^a-z]", "", token)
        if not alpha:
            continue
        if len(alpha) == 1 and alpha not in {"a", "i"}:
            suspicious_count += 1
            continue
        if len(alpha) >= 5 and not re.search(r"[aeiouy]", alpha):
            suspicious_count += 1
            continue
        if re.search(r"(.)\1\1", alpha):
            suspicious_count += 1

    nonspace_chars = re.sub(r"\s+", "", cleaned)
    weird_chars = re.findall(r"[^A-Za-z0-9\s,.;:!?()\[\]{}'\"/%+\-=]", cleaned)
    suspicious_ratio = suspicious_count / max(len(alnum_tokens), 1)
    weird_char_ratio = len(weird_chars) / max(len(nonspace_chars), 1)

    return suspicious_ratio >= 0.35 or weird_char_ratio >= 0.08


def is_valid_auxiliary_host(semantic_chunk: Dict[str, Any], *, min_tokens: int) -> bool:
    metadata = semantic_chunk.get("metadata") or {}
    token_count = int(metadata.get("token_count") or 0)
    if token_count < min_tokens:
        return False
    return not is_obvious_ocr_noise(semantic_chunk_text(semantic_chunk))


def attach_auxiliary_chunks(
    semantic_chunks: List[Dict[str, Any]],
    auxiliary_chunks: Sequence[Dict[str, Any]],
    *,
    aux_host_min_tokens: int,
) -> int:
    attached_count = 0
    for aux_chunk in auxiliary_chunks:
        aux_metadata = aux_chunk.get("metadata") or {}
        doc_id = str(aux_metadata.get("doc_id") or "")
        candidates = [
            chunk
            for chunk in semantic_chunks
            if str((chunk.get("metadata") or {}).get("doc_id") or "") == doc_id
            and is_valid_auxiliary_host(chunk, min_tokens=aux_host_min_tokens)
        ]
        if not candidates:
            continue
        best_chunk = min(candidates, key=lambda chunk: auxiliary_chunk_distance(aux_chunk, chunk))
        best_chunk["auxiliary_chunk_ids"].append(str(aux_chunk.get("chunk_id") or ""))
        generation = best_chunk.get("content_for_generation") or {}
        generation.setdefault("auxiliary_chunks", []).append(
            {
                "chunk_id": aux_chunk.get("chunk_id"),
                "chunk_type": aux_chunk.get("chunk_type"),
                "content_for_generation": aux_chunk.get("content_for_generation"),
                "metadata": aux_chunk.get("metadata"),
                "raw_fields": aux_chunk.get("raw_fields"),
            }
        )
        attached_count += 1
    return attached_count


def build_output_payload(
    *,
    course_id: str,
    atomic_chunks_path: Path,
    atomic_embeddings_path: Path,
    semantic_chunks: Sequence[Dict[str, Any]],
    auxiliary_chunk_count: int,
    min_tokens: int,
    max_tokens: int,
    threshold_quantile: float,
    high_similarity_quantile: float,
    aux_host_min_tokens: int,
    same_section_reject_below: float,
    different_section_reject_below: float,
    similarity_threshold: float,
    high_similarity: float,
    similarity_distribution: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "course_id": course_id,
        "source_atomic_chunks_path": str(atomic_chunks_path.as_posix()),
        "source_atomic_embeddings_path": str(atomic_embeddings_path.as_posix()),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "semantic_chunk_count": len(semantic_chunks),
        "auxiliary_chunk_count": auxiliary_chunk_count,
        "config": {
            "min_tokens": min_tokens,
            "max_tokens": max_tokens,
            "threshold_quantile": threshold_quantile,
            "high_similarity_quantile": high_similarity_quantile,
            "aux_host_min_tokens": aux_host_min_tokens,
            "same_section_reject_below": same_section_reject_below,
            "different_section_reject_below": different_section_reject_below,
            "similarity_threshold": similarity_threshold,
            "high_similarity": high_similarity,
        },
        "similarity_distribution": similarity_distribution,
        "semantic_chunks": list(semantic_chunks),
    }


def main() -> None:
    args = parse_args()
    if args.min_tokens <= 0 or args.max_tokens <= 0:
        raise ValueError("Token limits must be positive")
    if args.min_tokens > args.max_tokens:
        raise ValueError("--min-tokens must be <= --max-tokens")
    if not 0.0 <= args.threshold_quantile <= 1.0:
        raise ValueError("--threshold-quantile must be in [0, 1]")
    if not 0.0 <= args.high_similarity_quantile <= 1.0:
        raise ValueError("--high-similarity-quantile must be in [0, 1]")
    if args.aux_host_min_tokens < 0:
        raise ValueError("--aux-host-min-tokens must be non-negative")
    if not 0.0 <= args.same_section_reject_below <= 1.0:
        raise ValueError("--same-section-reject-below must be in [0, 1]")
    if not 0.0 <= args.different_section_reject_below <= 1.0:
        raise ValueError("--different-section-reject-below must be in [0, 1]")

    atomic_chunks_path = resolve_atomic_chunks_path(args.course_id, args.atomic_chunks_path)
    atomic_embeddings_path = resolve_atomic_embeddings_path(args.course_id, args.atomic_embeddings_path)
    output_path = resolve_output_path(args.course_id, args.output_dir)

    atomic_chunk_payload = load_json(atomic_chunks_path)
    atomic_embedding_payload = load_json(atomic_embeddings_path)

    atomic_records, auxiliary_chunks = build_atomic_records(
        atomic_chunk_payload=atomic_chunk_payload,
        atomic_embedding_payload=atomic_embedding_payload,
    )
    similarity_threshold, high_similarity, similarity_distribution = compute_similarity_thresholds(
        atomic_records,
        threshold_quantile=args.threshold_quantile,
        high_similarity_quantile=args.high_similarity_quantile,
    )
    semantic_chunks = build_semantic_chunks(
        atomic_records,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        threshold=similarity_threshold,
        high_similarity=high_similarity,
        same_section_reject_below=args.same_section_reject_below,
        different_section_reject_below=args.different_section_reject_below,
    )
    attached_auxiliary_count = attach_auxiliary_chunks(
        semantic_chunks,
        auxiliary_chunks,
        aux_host_min_tokens=args.aux_host_min_tokens,
    )

    output_payload = build_output_payload(
        course_id=args.course_id,
        atomic_chunks_path=atomic_chunks_path,
        atomic_embeddings_path=atomic_embeddings_path,
        semantic_chunks=semantic_chunks,
        auxiliary_chunk_count=attached_auxiliary_count,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        threshold_quantile=args.threshold_quantile,
        high_similarity_quantile=args.high_similarity_quantile,
        aux_host_min_tokens=args.aux_host_min_tokens,
        same_section_reject_below=args.same_section_reject_below,
        different_section_reject_below=args.different_section_reject_below,
        similarity_threshold=similarity_threshold,
        high_similarity=high_similarity,
        similarity_distribution=similarity_distribution,
    )
    save_json(output_path, output_payload)
    print(
        f"Wrote {len(semantic_chunks)} semantic chunks to {output_path} "
        f"with {attached_auxiliary_count} auxiliary chunks attached by position "
        f"(threshold={similarity_threshold:.4f}, high_similarity={high_similarity:.4f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
