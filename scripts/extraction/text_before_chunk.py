#!/usr/bin/env python3
"""Build cleaned text-inline-math records from processed course JSON."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None


DEFAULT_COURSE_ID = "adl"
QUALITY_VALUES = {"good", "broken"}
TITLE_TYPES = {"title"}
TEXT_TYPES = {"text"}
OUTPUT_FIELDS = (
    "block_id",
    "type",
    "page_no",
    "doc_id",
    "bbox",
    "nearby_text_before",
    "nearby_text_after",
    "section_title",
    "text_raw",
    "text_cleaned",
    "math_spans",
    "overall_quality",
)
INLINE_MATH_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "is_text_inline_math": {
            "type": "boolean",
            "description": "Whether the text block is a text_inline_math case.",
        },
        "text_cleaned": {
            "type": ["string", "null"],
            "description": "Conservatively repaired readable text.",
        },
        "math_spans": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "raw": {"type": "string"},
                    "normalized": {"type": "string"},
                    "math_name": {"type": ["string", "null"]},
                },
                "required": ["raw", "normalized", "math_name"],
                "additionalProperties": False,
            },
        },
        "overall_quality": {
            "type": ["string", "null"],
            "enum": ["good", "broken", None],
        },
    },
    "required": ["is_text_inline_math", "text_cleaned", "math_spans", "overall_quality"],
    "additionalProperties": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--course-id",
        default=DEFAULT_COURSE_ID,
        help="Course identifier, for example adl or eods.",
    )
    parser.add_argument("--input-path", help="Optional explicit path to the processed document JSON.")
    parser.add_argument("--output-dir", help="Optional output directory. Defaults to data/processed/<course_id>.")
    parser.add_argument("--doc-id", help="Optional document id filter, e.g. adl_lecture_1")
    parser.add_argument(
        "--llm-model",
        default="gpt-5.4-mini",
        help="LLM model used for inline-math detection and cleaning.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def resolve_input_path(course_id: str, explicit_path: Optional[str]) -> Path:
    if explicit_path:
        return Path(explicit_path)
    base_dir = Path("data/processed") / course_id
    processed_path = base_dir / f"{course_id}_processed.json"
    if processed_path.exists():
        return processed_path
    return base_dir / f"{course_id}_document.json"


def resolve_output_path(course_id: str, output_dir: Optional[str]) -> Path:
    base_dir = Path(output_dir) if output_dir else Path("data/processed") / course_id
    return base_dir / f"{course_id}_text_inline_math_cleaned.json"


def init_openai_client() -> Any:
    if OpenAI is None or not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OpenAI client is not available")
    return OpenAI()


def clean_text(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    text = text.replace("\u00a0", " ")
    text = text.replace("\u200b", "")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text or None


def normalize_sentence_text(text: Optional[str]) -> Optional[str]:
    text = clean_text(text)
    if not text:
        return None
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text or None


def sort_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(blocks, key=lambda block: int(block.get("reading_order", 10**9)))


def get_title_history(document: Dict[str, Any]) -> Dict[int, Optional[str]]:
    latest_title: Optional[str] = None
    history: Dict[int, Optional[str]] = {}
    for page in document.get("pages", []):
        page_no = int(page.get("page_no", 0))
        for block in sort_blocks(page.get("blocks", [])):
            if block.get("type") in TITLE_TYPES:
                candidate = normalize_sentence_text(block.get("text"))
                if candidate:
                    latest_title = candidate
        history[page_no] = latest_title
    return history


def collect_nearby_text(
    page: Dict[str, Any],
    text_block: Dict[str, Any],
    *,
    direction: str,
    limit: int = 3,
) -> Optional[str]:
    target_order = int(text_block.get("reading_order", 0))
    selected: List[str] = []

    blocks = sort_blocks(page.get("blocks", []))
    iterable = reversed(blocks) if direction == "before" else blocks

    for block in iterable:
        reading_order = int(block.get("reading_order", 0))
        if direction == "before":
            if reading_order >= target_order:
                continue
        elif reading_order <= target_order:
            continue

        if block.get("type") not in TEXT_TYPES | TITLE_TYPES:
            continue

        text = normalize_sentence_text(block.get("text"))
        if not text or len(text) < 16:
            continue
        selected.append(text)
        if len(selected) >= limit:
            break

    if direction == "before":
        selected = list(reversed(selected))
    if not selected:
        return None
    return clean_text(" ".join(selected))


def get_section_title(
    title_history: Dict[int, Optional[str]],
    page: Dict[str, Any],
    text_block: Dict[str, Any],
) -> Optional[str]:
    text_order = int(text_block.get("reading_order", 0))
    latest_title: Optional[str] = None

    for block in sort_blocks(page.get("blocks", [])):
        if int(block.get("reading_order", 0)) >= text_order:
            break
        if block.get("type") not in TITLE_TYPES:
            continue
        candidate = normalize_sentence_text(block.get("text"))
        if candidate:
            latest_title = candidate
    return latest_title or title_history.get(int(page.get("page_no", 0)))


def parse_llm_json_output(output_text: str) -> Optional[Dict[str, Any]]:
    candidate = output_text.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        candidate = re.sub(r"^```[A-Za-z0-9_-]*\n?", "", candidate)
        candidate = re.sub(r"\n?```$", "", candidate)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def llm_clean_text_inline_math(
    *,
    client: Any,
    model: str,
    text_raw: str,
    nearby_text_before: Optional[str],
    nearby_text_after: Optional[str],
    section_title: Optional[str],
) -> Dict[str, Any]:
    prompt = {
    "task": "Detect and clean text inline math",
    "instructions": [
        "Return valid JSON only.",
        "Decide whether this text block should be classified as text_inline_math.",
        "A text_inline_math block must still be mainly natural language, but contain one or more non-trivial inline math spans that are worth preserving, repairing, normalizing, or indexing.",
        "Use a high precision threshold. When uncertain, classify as false.",
        "Do not classify a block as text_inline_math merely because it contains symbols, numbers, variable letters, comparison signs, fractions, ranges, or Greek letters.",
        "Do not classify as text_inline_math if the block is adequately understandable and retrievable as ordinary text without special math handling.",
        "Only classify as text_inline_math when the inline math contributes substantive technical meaning in the local context and would benefit retrieval or understanding if normalized.",
        "A valid inline math span usually expresses a mathematical relation, operation, distribution, objective, function form, indexed quantity, vector or matrix expression, probabilistic expression, or a compact symbolic object with meaningful local semantics.",
        "Do not classify as text_inline_math if the candidate math content is only:",
        "1. standalone variable mentions such as x, y, z, f, σ, θ or simple constant euqation;",
        "2. short parenthesized variables such as (x, y);",
        "3. simple label encodings such as 1/0 or yes/no style mentions;",
        "4. grading rules, thresholds, or score cutoffs such as Score < 60 or Score ~ 82;",
        "5. time ranges, dates, counts, or administrative numeric expressions such as 1.5 hours - 2 hours;",
        "6. plain prose with technical symbols but no real damaged or non-trivial inline math to normalize.",
        "If no meaningful non-trivial inline math span exists, set is_text_inline_math to false.",
        "If is_text_inline_math is false, set text_cleaned to null, math_spans to [], and overall_quality to null.",
        "If is_text_inline_math is true, preserve the original sentence meaning.",
        "Repair only local math spans conservatively.",
        "Do not invent long formulas, derivations, or hidden conditions.",
        "text_cleaned should stay close to the original wording, except for conservative local fixes.",
        "Do not aggressively rewrite math notation in text_cleaned.",
        "math_spans should contain only meaningful inline math fragments that contribute technical semantics.",
        "Do not create math spans for isolated symbols or trivial numeric expressions.",
        "normalized should use a consistent lightweight LaTeX-style math format when the expression can be repaired confidently.",
        "math_name should be a short, context-aware semantic label for the math span.",
        "math_name should name the mathematical role of the span in the current local context, not just its generic object type.",
        "math_name may be null when the math object cannot be named confidently.",
        "overall_quality must be one of good or broken when is_text_inline_math is true.",
        "Set overall_quality to good when the cleaned text remains indexable and preserves useful meaning.",
        "Set overall_quality to broken only when the text and math are too corrupted to produce reliable indexable content.",
        "Return JSON with exactly these keys: is_text_inline_math, text_cleaned, math_spans, overall_quality."
    ],
    "inputs": {
        "text_raw": text_raw,
        "nearby_text_before": nearby_text_before,
        "nearby_text_after": nearby_text_after,
        "section_title": section_title,
    },
    } 

    response = client.responses.create(
        model=model,
        input=json.dumps(prompt, ensure_ascii=False),
        text={
            "format": {
                "type": "json_schema",
                "name": "text_inline_math_record",
                "schema": INLINE_MATH_SCHEMA,
                "strict": True,
            },
            "verbosity": "low",
        },
    )
    output_text = getattr(response, "output_text", None)
    if not output_text:
        raise RuntimeError("OpenAI response did not include output_text")

    payload = parse_llm_json_output(output_text)
    if payload is None:
        raise RuntimeError("Failed to parse LLM JSON output")
    return payload


def select_output_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    return {field: record.get(field) for field in OUTPUT_FIELDS}


def looks_like_single_symbol(text: str) -> bool:
    stripped = re.sub(r"\s+", "", text)
    stripped = stripped.strip("$")
    if not stripped:
        return False
    stripped = stripped.replace("^\\star", "").replace("\\star", "").replace("⋆", "")
    stripped = re.sub(r"_[A-Za-z0-9]+$", "", stripped)
    stripped = re.sub(r"\^[A-Za-z0-9]+$", "", stripped)
    stripped = re.sub(r"^\\(?:theta|lambda|sigma|alpha|beta|gamma|delta|mu|pi|phi|psi|omega)$", "v", stripped)
    return bool(re.fullmatch(r"[A-Za-zα-ωΑ-ΩσθλμβπγδfxyzXYZ]\d*", stripped))


def looks_like_short_parenthesized_variables(text: str) -> bool:
    stripped = re.sub(r"\s+", "", text)
    return bool(re.fullmatch(r"\([A-Za-z](?:,[A-Za-z]){0,3}\)", stripped))


def looks_like_simple_label_encoding(text: str) -> bool:
    stripped = re.sub(r"\s+", "", text.lower())
    return stripped in {"1/0", "0/1", "yes/no", "true/false"}


def looks_like_category_encoding(text: str) -> bool:
    lowered = clean_text(text.lower()) or ""
    if not lowered:
        return False
    if re.fullmatch(r"(?:['\"][^'\"]+['\"]\s*=\s*\d+\s*,?\s*){2,}", lowered):
        return True
    if lowered.count("=") >= 2 and re.search(r"['\"][^'\"]+['\"]\s*=\s*\d+", lowered):
        return True
    if "\\text{" in lowered and lowered.count("=") >= 2:
        return True
    return False


def looks_like_grade_or_threshold(text: str) -> bool:
    lowered = clean_text(text.lower()) or ""
    compact = re.sub(r"\s+", "", lowered)
    if "score" in lowered:
        return True
    if re.search(r"(grade|points|threshold|cutoff)", lowered):
        return True
    return bool(re.fullmatch(r"(?:[<>]=?|[≈~∼])\d+(?:\.\d+)?", compact))


def looks_like_administrative_numeric_expression(text: str) -> bool:
    lowered = clean_text(text.lower()) or ""
    if re.search(r"\b(hours?|days?|weeks?|minutes?|pts?|points?)\b", lowered):
        return True
    compact = re.sub(r"\s+", "", lowered)
    return bool(re.fullmatch(r"\d+(?:\.\d+)?-\d+(?:\.\d+)?", compact))


def is_trivial_math_span(raw: str, normalized: str) -> bool:
    candidates = [raw, normalized]
    return any(
        checker(candidate)
        for candidate in candidates
        for checker in (
            looks_like_single_symbol,
            looks_like_short_parenthesized_variables,
            looks_like_simple_label_encoding,
            looks_like_category_encoding,
            looks_like_grade_or_threshold,
            looks_like_administrative_numeric_expression,
        )
    )


def passes_inline_math_post_filter(
    *,
    text_raw: str,
    text_cleaned: Optional[str],
    math_spans: List[Dict[str, Optional[str]]],
) -> bool:
    if not text_cleaned:
        return False
    if not math_spans:
        return False

    substantive_spans = [
        span
        for span in math_spans
        if span.get("raw") and span.get("normalized")
        and not is_trivial_math_span(span["raw"], span["normalized"])
    ]
    if not substantive_spans:
        return False

    raw_lower = text_raw.lower()
    if (
        len(substantive_spans) == 1
        and re.search(r"\b(score|grade|points?)\b", raw_lower)
        and len(text_raw) < 120
    ):
        return False
    return True


def count_text_blocks(document: Dict[str, Any]) -> int:
    return sum(
        1
        for page in document.get("pages", [])
        for block in page.get("blocks", [])
        if block.get("type") == "text"
    )


def load_existing_text_output(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    try:
        payload = load_json(path)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def build_text_output_payload(
    *,
    payload: Dict[str, Any],
    input_path: Path,
    llm_model: str,
    all_records: List[Dict[str, Any]],
    processed_block_ids: List[str],
    per_document_counts: Dict[str, int],
) -> Dict[str, Any]:
    return {
        "course_id": payload.get("course_id"),
        "course_name": payload.get("course_name"),
        "source_type": payload.get("source_type"),
        "source_document_path": str(input_path.as_posix()),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "llm_model": llm_model,
        "record_count": len(all_records),
        "processed_text_block_count": len(processed_block_ids),
        "documents": per_document_counts,
        "processed_block_ids": processed_block_ids,
        "records": all_records,
    }


def persist_text_progress(
    *,
    output_path: Path,
    payload: Dict[str, Any],
    input_path: Path,
    llm_model: str,
    all_records: List[Dict[str, Any]],
    processed_block_ids: List[str],
    per_document_counts: Dict[str, int],
) -> None:
    save_json(
        output_path,
        build_text_output_payload(
            payload=payload,
            input_path=input_path,
            llm_model=llm_model,
            all_records=all_records,
            processed_block_ids=processed_block_ids,
            per_document_counts=per_document_counts,
        ),
    )


def build_text_record(
    *,
    doc: Dict[str, Any],
    page: Dict[str, Any],
    text_block: Dict[str, Any],
    title_history: Dict[int, Optional[str]],
    llm_client: Any,
    llm_model: str,
) -> Optional[Dict[str, Any]]:
    text_raw = clean_text(text_block.get("text"))
    if not text_raw:
        return None

    nearby_text_before = collect_nearby_text(page=page, text_block=text_block, direction="before")
    nearby_text_after = collect_nearby_text(page=page, text_block=text_block, direction="after")
    section_title = get_section_title(title_history=title_history, page=page, text_block=text_block)

    llm_result = llm_clean_text_inline_math(
        client=llm_client,
        model=llm_model,
        text_raw=text_raw,
        nearby_text_before=nearby_text_before,
        nearby_text_after=nearby_text_after,
        section_title=section_title,
    )

    if not llm_result.get("is_text_inline_math"):
        return None

    text_cleaned = clean_text(llm_result.get("text_cleaned"))
    math_spans = llm_result.get("math_spans") or []
    overall_quality = clean_text(llm_result.get("overall_quality") or "")
    if overall_quality not in QUALITY_VALUES:
        raise ValueError(
            f"Invalid overall_quality for block {text_block.get('block_id')}: {overall_quality!r}"
        )

    normalized_spans: List[Dict[str, Optional[str]]] = []
    for span in math_spans:
        if not isinstance(span, dict):
            continue
        raw = clean_text(span.get("raw"))
        normalized = clean_text(span.get("normalized"))
        math_name = clean_text(span.get("math_name"))
        if not raw or not normalized:
            continue
        normalized_spans.append(
            {
                "raw": raw,
                "normalized": normalized,
                "math_name": math_name,
            }
        )

    if not passes_inline_math_post_filter(
        text_raw=text_raw,
        text_cleaned=text_cleaned,
        math_spans=normalized_spans,
    ):
        return None

    record = {
        "block_id": text_block.get("block_id"),
        "type": "text_inline_math",
        "page_no": page.get("page_no"),
        "doc_id": doc.get("doc_id"),
        "bbox": text_block.get("bbox"),
        "nearby_text_before": nearby_text_before,
        "nearby_text_after": nearby_text_after,
        "section_title": section_title,
        "text_raw": text_raw,
        "text_cleaned": text_cleaned,
        "math_spans": normalized_spans,
        "overall_quality": overall_quality,
    }
    return select_output_fields(record)


def build_payload(
    payload: Dict[str, Any],
    *,
    doc_id: Optional[str],
    llm_client: Any,
    llm_model: str,
    input_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    documents = [
        document
        for document in payload.get("documents", [])
        if not doc_id or document.get("doc_id") == doc_id
    ]
    total_docs = len(documents)
    total_text_blocks = sum(count_text_blocks(document) for document in documents)
    scanned_blocks = 0
    existing_output = load_existing_text_output(output_path)
    existing_records = existing_output.get("records")
    if not isinstance(existing_records, list):
        existing_records = []
    text_records_by_id = {
        str(record.get("block_id")): record
        for record in existing_records
        if record.get("block_id")
    }
    raw_processed_ids = existing_output.get("processed_block_ids")
    if isinstance(raw_processed_ids, list):
        processed_block_ids = {str(item) for item in raw_processed_ids if item}
    else:
        processed_block_ids = set(text_records_by_id.keys())
    all_records: List[Dict[str, Any]] = list(text_records_by_id.values())
    per_document_counts: Dict[str, int] = {}
    for record in all_records:
        existing_doc_id = str(record.get("doc_id"))
        per_document_counts[existing_doc_id] = per_document_counts.get(existing_doc_id, 0) + 1

    for doc_index, document in enumerate(documents, start=1):
        doc_text_blocks = count_text_blocks(document)
        doc_scanned = 0
        doc_id_value = str(document.get("doc_id"))
        print(
            f"[{doc_index}/{total_docs}] Processing document {doc_id_value} "
            f"with {doc_text_blocks} text blocks",
            flush=True,
        )
        title_history = get_title_history(document)
        for page in document.get("pages", []):
            for block in sort_blocks(page.get("blocks", [])):
                if block.get("type") != "text":
                    continue
                scanned_blocks += 1
                doc_scanned += 1
                print(
                    f"  [{doc_scanned}/{doc_text_blocks}] global "
                    f"[{scanned_blocks}/{total_text_blocks}] "
                    f"page {page.get('page_no')} block {block.get('block_id')}",
                    flush=True,
                )
                block_id = str(block.get("block_id"))
                if block_id in processed_block_ids:
                    if block_id in text_records_by_id:
                        print("    resumed from existing output", flush=True)
                    else:
                        print("    skipped from previous pass", flush=True)
                    continue
                record = build_text_record(
                    doc=document,
                    page=page,
                    text_block=block,
                    title_history=title_history,
                    llm_client=llm_client,
                    llm_model=llm_model,
                )
                processed_block_ids.add(block_id)
                if record is not None:
                    text_records_by_id[block_id] = record
                    all_records[:] = list(text_records_by_id.values())
                    per_document_counts[doc_id_value] = sum(
                        1 for item in all_records if str(item.get("doc_id")) == doc_id_value
                    )
                    print("    kept as text_inline_math", flush=True)
                persist_text_progress(
                    output_path=output_path,
                    payload=payload,
                    input_path=input_path,
                    llm_model=llm_model,
                    all_records=all_records,
                    processed_block_ids=sorted(processed_block_ids),
                    per_document_counts=per_document_counts,
                )
                print("    saved progress", flush=True)
        print(
            f"[{doc_index}/{total_docs}] Finished document {doc_id_value}",
            flush=True,
        )

    return build_text_output_payload(
        payload=payload,
        input_path=input_path,
        llm_model=llm_model,
        all_records=all_records,
        processed_block_ids=sorted(processed_block_ids),
        per_document_counts=per_document_counts,
    )


def main() -> None:
    args = parse_args()
    input_path = resolve_input_path(args.course_id, args.input_path)
    output_path = resolve_output_path(args.course_id, args.output_dir)
    payload = load_json(input_path)
    llm_client = init_openai_client()
    documents = [
        document
        for document in payload.get("documents", [])
        if not args.doc_id or document.get("doc_id") == args.doc_id
    ]
    total_docs = len(documents)
    total_text_blocks = sum(count_text_blocks(document) for document in documents)
    print(f"Loading text blocks from {input_path}", flush=True)
    print(
        f"Found {total_docs} documents and {total_text_blocks} text blocks to scan",
        flush=True,
    )
    existing_output = load_existing_text_output(output_path)
    raw_processed_ids = existing_output.get("processed_block_ids")
    if isinstance(raw_processed_ids, list) and raw_processed_ids:
        print(
            f"Resuming from existing output with {len(raw_processed_ids)} processed text blocks",
            flush=True,
        )
    elif isinstance(existing_output.get("records"), list) and existing_output.get("records"):
        print(
            f"Resuming from existing output with {len(existing_output['records'])} saved text_inline_math records",
            flush=True,
        )
    output_payload = build_payload(
        payload,
        doc_id=args.doc_id,
        llm_client=llm_client,
        llm_model=args.llm_model,
        input_path=input_path,
        output_path=output_path,
    )
    save_json(output_path, output_payload)
    print(
        f"Wrote {output_payload['record_count']} text_inline_math records to {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
