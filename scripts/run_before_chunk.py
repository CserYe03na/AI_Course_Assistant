#!/usr/bin/env python3
"""Run pre-chunk cleaning scripts and merge their outputs into a unified block stream."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_COURSE_ID = "adl"
SCRIPT_DIR = Path(__file__).resolve().parent
EXTRACTION_DIR = SCRIPT_DIR / "extraction"
TITLE_TYPES = {"title"}
TEXT_TYPES = {"text", "title"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--course-id",
        default=DEFAULT_COURSE_ID,
        help="Course identifier, for example adl or eods.",
    )
    parser.add_argument(
        "--input-path",
        help="Optional explicit path to the processed document JSON passed to all three scripts.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional output directory passed to all three scripts and used for the merged output.",
    )
    parser.add_argument(
        "--doc-id",
        help="Optional document id filter, for example adl_lecture_1.",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-5.4-mini",
        help="LLM model passed to figure, formula, and text cleaning scripts.",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path.cwd(),
        help="Workspace root passed to extraction/formula_before_chunk.py.",
    )
    parser.add_argument(
        "--skip-figure",
        action="store_true",
        help="Skip extraction/figure_before_chunk.py.",
    )
    parser.add_argument(
        "--skip-formula",
        action="store_true",
        help="Skip extraction/formula_before_chunk.py.",
    )
    parser.add_argument(
        "--skip-text",
        action="store_true",
        help="Skip extraction/text_before_chunk.py.",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip the final merge step.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def strip_control_chars(text: str) -> str:
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def sanitize_nested_strings(value: Any) -> Any:
    if isinstance(value, str):
        return strip_control_chars(value)
    if isinstance(value, list):
        return [sanitize_nested_strings(item) for item in value]
    if isinstance(value, dict):
        return {key: sanitize_nested_strings(item) for key, item in value.items()}
    return value


def resolve_input_path(course_id: str, explicit_path: str | None) -> Path:
    if explicit_path:
        return Path(explicit_path)
    base_dir = Path("data/processed") / course_id
    processed_path = base_dir / f"{course_id}_processed.json"
    if processed_path.exists():
        return processed_path
    return base_dir / f"{course_id}_document.json"


def resolve_output_dir(course_id: str, explicit_output_dir: str | None) -> Path:
    return Path(explicit_output_dir) if explicit_output_dir else Path("data/processed") / course_id


def resolve_cleaned_paths(course_id: str, output_dir: Path) -> Dict[str, Path]:
    return {
        "figure": output_dir / f"{course_id}_figures_cleaned.json",
        "formula": output_dir / f"{course_id}_formula_cleaned.json",
        "text": output_dir / f"{course_id}_text_inline_math_cleaned.json",
        "merged": output_dir / f"{course_id}_merged.json",
    }


def build_shared_args(args: argparse.Namespace) -> List[str]:
    shared_args = ["--course-id", args.course_id, "--llm-model", args.llm_model]
    if args.input_path:
        shared_args.extend(["--input-path", args.input_path])
    if args.output_dir:
        shared_args.extend(["--output-dir", args.output_dir])
    if args.doc_id:
        shared_args.extend(["--doc-id", args.doc_id])
    return shared_args


def run_step(step_name: str, command: List[str]) -> None:
    print(f"Running {step_name}...", flush=True)
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)
    print(f"Finished {step_name}", flush=True)


def index_records(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        str(record.get("block_id")): record
        for record in records
        if isinstance(record, dict) and record.get("block_id")
    }


def clean_text(text: str | None) -> str | None:
    if text is None:
        return None
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def sort_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(blocks, key=lambda block: int(block.get("reading_order", 10**9)))


def get_document_title_block_id(document: Dict[str, Any]) -> str | None:
    for page in document.get("pages", []):
        for block in sort_blocks(page.get("blocks", [])):
            if block.get("type") not in TITLE_TYPES:
                continue
            candidate = clean_text(block.get("text"))
            if candidate:
                return str(block.get("block_id"))
    return None


def get_title_history(document: Dict[str, Any], *, document_title_block_id: str | None) -> Dict[int, str | None]:
    latest_title: str | None = None
    history: Dict[int, str | None] = {}
    for page in document.get("pages", []):
        page_no = int(page.get("page_no", 0))
        history[page_no] = latest_title
        for block in sort_blocks(page.get("blocks", [])):
            if block.get("type") not in TITLE_TYPES:
                continue
            if str(block.get("block_id")) == document_title_block_id:
                continue
            candidate = clean_text(block.get("text"))
            if candidate:
                latest_title = candidate
    return history


def get_section_title(
    title_history: Dict[int, str | None],
    page: Dict[str, Any],
    block: Dict[str, Any],
    *,
    document_title_block_id: str | None,
) -> str | None:
    block_order = int(block.get("reading_order", 0))
    latest_title: str | None = None

    for candidate_block in sort_blocks(page.get("blocks", [])):
        if int(candidate_block.get("reading_order", 0)) >= block_order:
            break
        if candidate_block.get("type") not in TITLE_TYPES:
            continue
        if str(candidate_block.get("block_id")) == document_title_block_id:
            continue
        candidate = clean_text(candidate_block.get("text"))
        if candidate:
            latest_title = candidate

    return latest_title or title_history.get(int(page.get("page_no", 0)))


def collect_nearby_text(
    page: Dict[str, Any],
    text_block: Dict[str, Any],
    *,
    direction: str,
    limit: int = 3,
) -> str | None:
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

        if block.get("type") not in TEXT_TYPES:
            continue

        text = clean_text(block.get("text"))
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


def load_cleaned_record_maps(paths: Dict[str, Path]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    figure_payload = load_json(paths["figure"]) if paths["figure"].exists() else {}
    formula_payload = load_json(paths["formula"]) if paths["formula"].exists() else {}
    text_payload = load_json(paths["text"]) if paths["text"].exists() else {}

    return {
        "figure": index_records(figure_payload.get("figures") or []),
        "formula": index_records(formula_payload.get("records") or []),
        "text": index_records(text_payload.get("records") or []),
    }


def merge_block(
    original_block: Dict[str, Any],
    *,
    doc_id: str,
    page_no: int,
    page: Dict[str, Any],
    title_history: Dict[int, str | None],
    document_title_block_id: str | None,
    figure_records: Dict[str, Dict[str, Any]],
    formula_records: Dict[str, Dict[str, Any]],
    text_records: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Any], str]:
    block_id = str(original_block.get("block_id"))
    block_type = original_block.get("type")

    merged_block = dict(original_block)
    merged_block["doc_id"] = doc_id
    merged_block["page_no"] = page_no
    if block_type == "text" and "section_title" not in merged_block:
        merged_block["section_title"] = get_section_title(
            title_history,
            page,
            original_block,
            document_title_block_id=document_title_block_id,
        )
    if block_type == "text" and "nearby_text_before" not in merged_block:
        merged_block["nearby_text_before"] = collect_nearby_text(
            page,
            original_block,
            direction="before",
        )
    if block_type == "text" and "nearby_text_after" not in merged_block:
        merged_block["nearby_text_after"] = collect_nearby_text(
            page,
            original_block,
            direction="after",
        )
    source = "original"

    if block_type == "figure" and block_id in figure_records:
        merged_block.update(figure_records[block_id])
        source = "figure_cleaned"
    elif block_type == "formula" and block_id in formula_records:
        merged_block.update(formula_records[block_id])
        source = "formula_cleaned"
    elif block_type == "text" and block_id in text_records:
        merged_block.update(text_records[block_id])
        source = "text_inline_math_cleaned"

    return sanitize_nested_strings(merged_block), source


def merge_cleaned_outputs(
    *,
    original_payload: Dict[str, Any],
    cleaned_maps: Dict[str, Dict[str, Dict[str, Any]]],
    input_path: Path,
    merged_output_path: Path,
    doc_id: str | None,
) -> Dict[str, Any]:
    merged_blocks: List[Dict[str, Any]] = []
    merged_counts = {
        "figure_cleaned": 0,
        "formula_cleaned": 0,
        "text_inline_math_cleaned": 0,
        "original": 0,
    }
    total_blocks = 0

    for document in original_payload.get("documents", []):
        if doc_id and document.get("doc_id") != doc_id:
            continue

        doc_id_value = str(document.get("doc_id"))
        document_title_block_id = get_document_title_block_id(document)
        title_history = get_title_history(
            document,
            document_title_block_id=document_title_block_id,
        )
        for page in document.get("pages", []):
            page_no = int(page.get("page_no", 0))
            for block in page.get("blocks", []):
                if block.get("type") == "table":
                    continue
                merged_block, source = merge_block(
                    block,
                    doc_id=doc_id_value,
                    page_no=page_no,
                    page=page,
                    title_history=title_history,
                    document_title_block_id=document_title_block_id,
                    figure_records=cleaned_maps["figure"],
                    formula_records=cleaned_maps["formula"],
                    text_records=cleaned_maps["text"],
                )
                merged_blocks.append(merged_block)
                merged_counts[source] += 1
                total_blocks += 1

    merged_payload = {
        "course_id": original_payload.get("course_id"),
        "course_name": original_payload.get("course_name"),
        "source_type": original_payload.get("source_type"),
        "source_document_path": str(input_path.as_posix()),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "document_count": sum(
            1
            for document in original_payload.get("documents", [])
            if not doc_id or document.get("doc_id") == doc_id
        ),
        "block_count": total_blocks,
        "merge_summary": merged_counts,
        "blocks": merged_blocks,
    }
    merged_payload = sanitize_nested_strings(merged_payload)
    save_json(merged_output_path, merged_payload)
    return merged_payload


def main() -> None:
    args = parse_args()
    shared_args = build_shared_args(args)
    output_dir = resolve_output_dir(args.course_id, args.output_dir)
    input_path = resolve_input_path(args.course_id, args.input_path)
    cleaned_paths = resolve_cleaned_paths(args.course_id, output_dir)

    steps: List[tuple[str, List[str]]] = []
    if not args.skip_figure:
        steps.append(
            (
                "figure_before_chunk.py",
                [sys.executable, str(EXTRACTION_DIR / "figure_before_chunk.py"), *shared_args],
            )
        )
    if not args.skip_formula:
        steps.append(
            (
                "formula_before_chunk.py",
                [
                    sys.executable,
                    str(EXTRACTION_DIR / "formula_before_chunk.py"),
                    *shared_args,
                    "--workspace-root",
                    str(args.workspace_root.resolve()),
                ],
            )
        )
    if not args.skip_text:
        steps.append(
            (
                "text_before_chunk.py",
                [sys.executable, str(EXTRACTION_DIR / "text_before_chunk.py"), *shared_args],
            )
        )

    print(f"Using Python: {sys.executable}", flush=True)
    print(f"Scripts directory: {SCRIPT_DIR}", flush=True)
    print(f"Extraction directory: {EXTRACTION_DIR}", flush=True)
    print(f"Input path: {input_path}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)

    if steps:
        print(f"Selected steps: {', '.join(name for name, _ in steps)}", flush=True)
        for step_name, command in steps:
            run_step(step_name, command)
    else:
        print("No cleaning steps selected. Reusing existing cleaned JSON files.", flush=True)

    if args.skip_merge:
        print("Skipping merge step.", flush=True)
        return

    print("Merging cleaned outputs back into the original document structure...", flush=True)
    cleaned_maps = load_cleaned_record_maps(cleaned_paths)
    original_payload = load_json(input_path)
    merged_payload = merge_cleaned_outputs(
        original_payload=original_payload,
        cleaned_maps=cleaned_maps,
        input_path=input_path,
        merged_output_path=cleaned_paths["merged"],
        doc_id=args.doc_id,
    )
    print(
        f"Wrote merged block stream to {cleaned_paths['merged']} "
        f"with {merged_payload['block_count']} blocks",
        flush=True,
    )
    print(f"Merge summary: {merged_payload['merge_summary']}", flush=True)


if __name__ == "__main__":
    main()
