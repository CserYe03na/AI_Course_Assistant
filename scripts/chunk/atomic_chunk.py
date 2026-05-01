#!/usr/bin/env python3
"""Build atomic chunks from merged block streams."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_COURSE_ID = "adl"
SKIP_BLOCK_TYPES = {"title", "table"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--course-id",
        default=DEFAULT_COURSE_ID,
        help="Course identifier, for example adl or eods.",
    )
    parser.add_argument(
        "--input-path",
        help="Optional explicit path to the merged block JSON. Defaults to data/processed/<course_id>/<course_id>_merged.json.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional output directory. Defaults to data/processed/<course_id>.",
    )
    parser.add_argument(
        "--doc-id",
        help="Optional document id filter, for example adl_lecture_1.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def clean_text(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    text = text.replace("\u00a0", " ")
    text = text.replace("\u200b", "")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text or None


def resolve_input_path(course_id: str, explicit_path: Optional[str]) -> Path:
    if explicit_path:
        return Path(explicit_path)
    return Path("data/processed") / course_id / f"{course_id}_merged.json"


def resolve_output_path(course_id: str, output_dir: Optional[str]) -> Path:
    base_dir = Path(output_dir) if output_dir else Path("data/chunk")
    return base_dir / f"{course_id}_atomic_chunks.json"


def join_embedding_parts(*parts: Any) -> Optional[str]:
    normalized: List[str] = []
    for part in parts:
        if part is None:
            continue
        if isinstance(part, list):
            for item in part:
                item_text = clean_text(str(item))
                if item_text:
                    normalized.append(item_text)
            continue
        part_text = clean_text(str(part))
        if part_text:
            normalized.append(part_text)
    if not normalized:
        return None
    return ". ".join(normalized)


def normalized_math_span_text(math_spans: List[Dict[str, Any]]) -> List[str]:
    phrases: List[str] = []
    for span in math_spans:
        if not isinstance(span, dict):
            continue
        math_name = clean_text(span.get("math_name"))
        normalized = clean_text(span.get("normalized"))
        if math_name and normalized:
            phrases.append(f"{math_name} {normalized}")
        elif math_name:
            phrases.append(math_name)
        elif normalized:
            phrases.append(normalized)
    return phrases


def build_formula_chunk(block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    overall_quality = clean_text(block.get("overall_quality"))
    if overall_quality == "broken":
        return None

    section_title = clean_text(block.get("section_title"))
    formula_focus = clean_text(block.get("formula_focus"))
    formula_explanation = clean_text(block.get("formula_explanation"))
    formula_latex = clean_text(block.get("formula_latex"))

    return {
        "chunk_id": f"{block.get('block_id')}_formula",
        "chunk_type": "formula",
        "indexable": overall_quality in {"good", "poor"},
        "content_for_embedding": join_embedding_parts(section_title, formula_focus, formula_explanation),
        "content_for_generation": {
            "section_title": section_title,
            "formula_focus": formula_focus,
            "formula_explanation": formula_explanation,
            "formula_latex": formula_latex,
        },
        "metadata": {
            "doc_id": block.get("doc_id"),
            "page_no": block.get("page_no"),
            "block_id": block.get("block_id"),
            "bbox": block.get("bbox"),
            "overall_quality": overall_quality,
        },
        "raw_fields": {
            "formula_latex": formula_latex,
            "nearby_text_before": clean_text(block.get("nearby_text_before")),
            "nearby_text_after": clean_text(block.get("nearby_text_after")),
        },
    }


def build_figure_chunk(block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    overall_quality = clean_text(block.get("overall_quality"))
    if overall_quality == "broken":
        return None

    section_title = clean_text(block.get("section_title"))
    figure_focus = clean_text(block.get("figure_focus"))
    visual_description = clean_text(block.get("visual_description"))
    keywords = [
        clean_text(str(item))
        for item in (block.get("keywords") or [])
        if clean_text(str(item))
    ]
    indexable = overall_quality == "good"

    return {
        "chunk_id": f"{block.get('block_id')}_figure",
        "chunk_type": "figure",
        "indexable": indexable,
        "content_for_embedding": (
            join_embedding_parts(section_title, figure_focus, visual_description, keywords)
            if indexable
            else None
        ),
        "content_for_generation": {
            "section_title": section_title,
            "figure_focus": figure_focus,
            "visual_description": visual_description,
            "keywords": keywords,
        },
        "metadata": {
            "doc_id": block.get("doc_id"),
            "page_no": block.get("page_no"),
            "block_id": block.get("block_id"),
            "bbox": block.get("bbox"),
            "image_path": block.get("image_path"),
            "overall_quality": overall_quality,
        },
        "raw_fields": {
            "nearby_text_before": clean_text(block.get("nearby_text_before")),
            "nearby_text_after": clean_text(block.get("nearby_text_after")),
            "visual_description": visual_description,
            "figure_focus": figure_focus,
            "keywords": keywords,
        },
    }


def build_text_inline_math_chunk(block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    overall_quality = clean_text(block.get("overall_quality"))
    if overall_quality == "broken":
        return None

    section_title = clean_text(block.get("section_title"))
    text_cleaned = clean_text(block.get("text_cleaned"))
    math_spans = block.get("math_spans") or []

    return {
        "chunk_id": f"{block.get('block_id')}_text_inline_math",
        "chunk_type": "text_inline_math",
        "indexable": True,
        "content_for_embedding": join_embedding_parts(
            section_title,
            text_cleaned,
            normalized_math_span_text(math_spans),
        ),
        "content_for_generation": {
            "section_title": section_title,
            "text_cleaned": text_cleaned,
            "math_spans": math_spans,
        },
        "metadata": {
            "doc_id": block.get("doc_id"),
            "page_no": block.get("page_no"),
            "block_id": block.get("block_id"),
            "bbox": block.get("bbox"),
            "overall_quality": overall_quality,
        },
        "raw_fields": {
            "nearby_text_before": clean_text(block.get("nearby_text_before")),
            "nearby_text_after": clean_text(block.get("nearby_text_after")),
            "text_raw": clean_text(block.get("text_raw")),
            "text_cleaned": text_cleaned,
            "math_spans": math_spans,
        },
    }


def build_text_chunk(block: Dict[str, Any]) -> Dict[str, Any]:
    section_title = clean_text(block.get("section_title"))
    text = clean_text(block.get("text"))
    original_type = clean_text(block.get("type")) or "text"

    return {
        "chunk_id": f"{block.get('block_id')}_text",
        "chunk_type": "text",
        "indexable": True,
        "content_for_embedding": join_embedding_parts(section_title, text),
        "content_for_generation": {
            "section_title": section_title,
            "text": text,
        },
        "metadata": {
            "doc_id": block.get("doc_id"),
            "page_no": block.get("page_no"),
            "block_id": block.get("block_id"),
            "bbox": block.get("bbox"),
            "reading_order": block.get("reading_order"),
            "original_type": original_type,
            "indexable": True,
        },
        "raw_fields": {
            "text": text,
            "nearby_text_before": clean_text(block.get("nearby_text_before")),
            "nearby_text_after": clean_text(block.get("nearby_text_after")),
        },
    }


def build_atomic_chunk(block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    block_type = clean_text(block.get("type"))
    if block_type in SKIP_BLOCK_TYPES:
        return None
    if block_type == "formula":
        return build_formula_chunk(block)
    if block_type == "figure":
        return build_figure_chunk(block)
    if block_type == "text_inline_math":
        return build_text_inline_math_chunk(block)
    if block_type == "text":
        return build_text_chunk(block)
    return None


def build_output_payload(
    *,
    merged_payload: Dict[str, Any],
    input_path: Path,
    chunks: List[Dict[str, Any]],
    skipped_counts: Dict[str, int],
    doc_id: Optional[str],
) -> Dict[str, Any]:
    return {
        "course_id": merged_payload.get("course_id"),
        "course_name": merged_payload.get("course_name"),
        "source_type": merged_payload.get("source_type"),
        "source_document_path": str(input_path.as_posix()),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "doc_id_filter": doc_id,
        "chunk_count": len(chunks),
        "indexable_chunk_count": sum(1 for chunk in chunks if chunk.get("indexable")),
        "skipped_counts": skipped_counts,
        "chunks": chunks,
    }


def should_keep_block(block: Dict[str, Any], doc_id: Optional[str]) -> bool:
    if doc_id and block.get("doc_id") != doc_id:
        return False
    return True


def main() -> None:
    args = parse_args()
    input_path = resolve_input_path(args.course_id, args.input_path)
    output_path = resolve_output_path(args.course_id, args.output_dir)
    payload = load_json(input_path)

    chunks: List[Dict[str, Any]] = []
    skipped_counts = {
        "title_or_table": 0,
        "broken_or_discarded": 0,
        "unknown_type": 0,
    }

    for block in payload.get("blocks", []):
        if not should_keep_block(block, args.doc_id):
            continue

        block_type = clean_text(block.get("type"))
        if block_type in SKIP_BLOCK_TYPES:
            skipped_counts["title_or_table"] += 1
            continue

        chunk = build_atomic_chunk(block)
        if chunk is None:
            if block_type in {"formula", "figure", "text_inline_math"}:
                skipped_counts["broken_or_discarded"] += 1
            else:
                skipped_counts["unknown_type"] += 1
            continue
        chunks.append(chunk)

    output_payload = build_output_payload(
        merged_payload=payload,
        input_path=input_path,
        chunks=chunks,
        skipped_counts=skipped_counts,
        doc_id=args.doc_id,
    )
    save_json(output_path, output_payload)
    print(
        f"Wrote {len(chunks)} atomic chunks to {output_path} "
        f"({output_payload['indexable_chunk_count']} indexable)",
        flush=True,
    )


if __name__ == "__main__":
    main()
