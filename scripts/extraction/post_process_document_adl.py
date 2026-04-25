#!/usr/bin/env python3
"""Post-process the extracted ADL course document JSON.

Tasks:
1. Clean formula text strings.
2. Remove text blocks that fall inside a figure on the same page.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List


INPUT_PATH = Path("data/processed/adl/adl_document.json")
OUTPUT_PATH = Path("data/processed/adl/adl_processed.json")

def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def clean_formula_text(text: str) -> str:
    text = text.replace("▶", " ")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # Conservatively merge sequences of single-letter tokens such as
    # "L e a k y" -> "Leaky" without relying on a formula-specific dictionary.
    text = re.sub(
        r"(?<![A-Za-z\\])(?:[A-Za-z]\s+){2,}[A-Za-z](?![A-Za-z])",
        lambda match: "".join(match.group(0).split()),
        text,
    )

    # Structural cleanup only: normalize spacing around LaTeX-style syntax
    # without hard-coding specific formula words.
    text = re.sub(r"\s+([,.;:)\]\}])", r"\1", text)
    text = re.sub(r"([(\[\{])\s+", r"\1", text)
    text = re.sub(r"\s*_\s*", "_", text)
    text = re.sub(r"\s*\^\s*", "^", text)
    text = re.sub(r"\s*\\\s+([A-Za-z]+)", r" \\\1", text)
    text = re.sub(r"\{\s+", "{", text)
    text = re.sub(r"\s+\}", "}", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\[\s+", "[", text)
    text = re.sub(r"\s+\]", "]", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def clean_general_text(text: str) -> str:
    text = text.replace("▶", " ")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def classify_formula_quality(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "truncated"

    brace_diff = abs(stripped.count("{") - stripped.count("}"))
    paren_diff = abs(stripped.count("(") - stripped.count(")"))

    if brace_diff >= 2 or paren_diff >= 2:
        return "truncated"

    if stripped.endswith(("=", "+", "-", "\\", "{", "(")):
        return "truncated"

    if stripped.startswith(("0)", "1)", "2)", "3)", "\\\\", "&")):
        return "truncated"

    if "\\text {" in stripped and "\\\\" in stripped and len(stripped) > 120:
        return "truncated"

    if re.search(r"(?:[A-Za-z]\s+){3,}[A-Za-z](?![A-Za-z])", stripped):
        return "noisy"

    if "\\ " in stripped or re.search(r"\\\\\s+[A-Za-z]", stripped):
        return "noisy"

    if re.search(r"[A-Za-z]+\\max", stripped):
        return "noisy"

    if re.search(r"\\[A-Za-z]+\s+[A-Za-z]+", stripped):
        return "noisy"

    if "\\mathring" in stripped or "\\intercal" in stripped:
        return "noisy"

    if brace_diff == 1 or paren_diff == 1:
        return "noisy"

    return "good"


def bbox_inside(inner: List[float], outer: List[float], tolerance: float = 2.0) -> bool:
    return (
        inner[0] >= outer[0] - tolerance
        and inner[1] >= outer[1] - tolerance
        and inner[2] <= outer[2] + tolerance
        and inner[3] <= outer[3] + tolerance
    )


def bbox_intersects(a: List[float], b: List[float], tolerance: float = 0.0) -> bool:
    return not (
        a[2] < b[0] - tolerance
        or a[0] > b[2] + tolerance
        or a[3] < b[1] - tolerance
        or a[1] > b[3] + tolerance
    )


def bbox_area(bbox: List[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def bbox_intersection_area(a: List[float], b: List[float]) -> float:
    overlap_left = max(a[0], b[0])
    overlap_top = max(a[1], b[1])
    overlap_right = min(a[2], b[2])
    overlap_bottom = min(a[3], b[3])

    if overlap_right <= overlap_left or overlap_bottom <= overlap_top:
        return 0.0

    return (overlap_right - overlap_left) * (overlap_bottom - overlap_top)


def bbox_overlap_ratio(inner: List[float], outer: List[float]) -> float:
    inner_area = bbox_area(inner)
    if inner_area <= 0.0:
        return 0.0
    return bbox_intersection_area(inner, outer) / inner_area


def bbox_center(bbox: List[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def point_inside_bbox(point: tuple[float, float], bbox: List[float], tolerance: float = 0.0) -> bool:
    x, y = point
    return (
        bbox[0] - tolerance <= x <= bbox[2] + tolerance
        and bbox[1] - tolerance <= y <= bbox[3] + tolerance
    )


def bbox_vertical_gap(a: List[float], b: List[float]) -> float:
    if a[3] < b[1]:
        return b[1] - a[3]
    if b[3] < a[1]:
        return a[1] - b[3]
    return 0.0


def bbox_width(bbox: List[float]) -> float:
    return max(0.0, bbox[2] - bbox[0])


def bbox_horizontal_overlap_ratio(a: List[float], b: List[float]) -> float:
    overlap = min(a[2], b[2]) - max(a[0], b[0])
    if overlap <= 0:
        return 0.0
    base = min(bbox_width(a), bbox_width(b))
    if base <= 0:
        return 0.0
    return overlap / base


def bbox_height(bbox: List[float]) -> float:
    return max(0.0, bbox[3] - bbox[1])


def bbox_vertical_overlap_ratio(a: List[float], b: List[float]) -> float:
    overlap = min(a[3], b[3]) - max(a[1], b[1])
    if overlap <= 0:
        return 0.0
    base = min(bbox_height(a), bbox_height(b))
    if base <= 0:
        return 0.0
    return overlap / base


def is_large_figure(page: Dict[str, Any], block: Dict[str, Any]) -> bool:
    page_width = float(page.get("width") or 0.0)
    page_height = float(page.get("height") or 0.0)
    page_area = max(1.0, page_width * page_height)
    figure_area = bbox_area(block.get("bbox") or [0.0, 0.0, 0.0, 0.0])
    return figure_area / page_area >= 0.12


def is_caption_or_table_note(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    if re.match(r"^(figure|fig|table)\b", stripped, flags=re.IGNORECASE):
        return True

    if re.search(r"\b(?:figure|fig|table)\s+[A-Za-z]?\d", stripped, flags=re.IGNORECASE):
        return True

    return False


def is_noisy_text_block(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    numeric_tokens = len(re.findall(r"\d+(?:\.\d+)?", stripped))
    short_tokens = len(re.findall(r"\b\w{1,3}\b", stripped))
    has_table_pipe = "|" in stripped
    has_ocr_artifacts = bool(re.search(r"[÷‹›•·₫∆≈]", stripped))
    non_ascii = bool(re.search(r"[^\x00-\x7F]", stripped))
    dimension_like = bool(re.search(r"\b\d+\s*[x×]\s*\d+(?:\s*[x×]\s*\d+)?\b", stripped, flags=re.IGNORECASE))
    metric_like = bool(
        re.search(r"\b(?:BLEU|F1|Acc|Accuracy|SST-2|MRPC|MNLI|QQP|STS-B|SQuAD|SuperGLUE)\b", stripped)
    )

    if has_table_pipe and numeric_tokens >= 2:
        return True
    if numeric_tokens >= 6:
        return True
    if metric_like and numeric_tokens >= 2:
        return True
    if has_ocr_artifacts:
        return True
    if dimension_like and short_tokens >= 2:
        return True
    if non_ascii and numeric_tokens >= 1:
        return True

    return False


def is_table_caption_or_summary(block: Dict[str, Any]) -> bool:
    if block.get("type") != "text":
        return False

    text = (block.get("text") or "").strip()
    if not text:
        return False

    if re.match(r"^table\b", text, flags=re.IGNORECASE):
        return True

    return len(text) >= 70 or len(text.split()) >= 12


def is_numeric_heavy_sentence(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    words = stripped.split()
    if len(words) < 10:
        return False

    numeric_tokens = re.findall(r"\d+(?:\.\d+)?", stripped)
    slash_numeric_spans = re.findall(r"\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?){1,3}", stripped)

    return (
        len(numeric_tokens) >= 6
        or len(slash_numeric_spans) >= 2
        or (len(numeric_tokens) >= 4 and len(words) <= 28)
    )


def is_obvious_noise_text_block(block: Dict[str, Any]) -> bool:
    if block.get("type") != "text":
        return False

    text = (block.get("text") or "").strip()
    if not text:
        return False

    words = text.split()
    word_count = len(words)
    char_count = len(text)
    numeric_spans = re.findall(r"\d+(?:\.\d+)?", text)
    slash_numeric = re.fullmatch(r"\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?){1,3}", text)
    formulaish_short_phrase = bool(
        re.match(r"^(?:Full|Partial)\s+Scoring:\s*P\s*\(", text, flags=re.IGNORECASE)
    )
    pure_numeric = bool(
        re.fullmatch(r"\d+(?:\.\d+)?(?:[MK%]|M)?", text)
        or re.fullmatch(r"\d+(?:,\d{3})+(?:\.\d+)?[MK%]?", text)
        or slash_numeric
    )
    high_numeric_density = (
        len(numeric_spans) >= 3
        or (char_count > 0 and sum(ch.isdigit() for ch in text) / char_count >= 0.35)
    )
    ocr_noise = is_noisy_text_block(text)
    isolated_token = is_table_content_fragment_text(block)
    looks_like_sentence = word_count >= 5 and bool(re.search(r"[a-z]{3,}", text))

    if word_count <= 2:
        return True
    if char_count <= 12:
        return True
    if formulaish_short_phrase:
        return False
    if pure_numeric:
        return True
    if high_numeric_density and not looks_like_sentence:
        return True
    if ocr_noise:
        return True
    if isolated_token and not looks_like_sentence:
        return True
    return False


def is_near_figure_fragment_text(block: Dict[str, Any]) -> bool:
    if block.get("type") != "text":
        return False

    text = (block.get("text") or "").strip()
    if not text:
        return False

    words = text.split()
    word_count = len(words)
    char_count = len(text)
    normalized = re.sub(r"\s+", " ", text)
    looks_like_sentence = (
        word_count >= 6
        or normalized.endswith((".", "?", "!"))
        or ";" in normalized
        or ":" in normalized and word_count >= 5
    )
    formulaish_short_phrase = bool(
        re.match(r"^(?:Full|Partial)\s+Scoring:\s*P\s*\(", normalized, flags=re.IGNORECASE)
    )

    short_label = (
        word_count <= 4
        and char_count <= 42
        and not normalized.endswith((".", "?", "!"))
    )
    qa_item = bool(
        re.match(r"^(?:who|what|when|where|why|how|which|state|name)\b", normalized, flags=re.IGNORECASE)
        or "question answering" in normalized.lower()
        or normalized.lower().startswith("allow ")
    ) and word_count <= 10
    scoring_or_output = bool(
        re.search(r"\b(?:zero shot|one-shot|few-shot|json output|maximize|output)\b", normalized, flags=re.IGNORECASE)
        or normalized.startswith("<>")
    ) and not looks_like_sentence
    legend_like = bool(
        re.search(r"\b(?:example|context|completion|parameters in lm|small model completion|gpt-2 completion)\b", normalized, flags=re.IGNORECASE)
        or "|" in normalized
    ) and word_count <= 8
    demo_local_text = bool(
        re.search(r"\b(?:hugging face|billing|account access|website|mobile|winograd)\b", normalized, flags=re.IGNORECASE)
    ) and word_count <= 8

    if formulaish_short_phrase:
        return False

    return short_label or qa_item or scoring_or_output or legend_like or demo_local_text


def is_table_content_fragment_text(block: Dict[str, Any]) -> bool:
    if block.get("type") != "text":
        return False

    text = (block.get("text") or "").strip()
    if not text:
        return False

    words = text.split()
    if len(words) > 4 or len(text) > 64:
        return False

    normalized = re.sub(r"\s+", " ", text)
    numeric_like = bool(
        re.fullmatch(r"\d+(?:\.\d+)?(?:[MK%]|M)?", normalized)
        or re.fullmatch(r"\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?){1,3}", normalized)
        or re.fullmatch(r"\d+(?:,\d{3})+(?:\.\d+)?[MK]?", normalized)
    )
    metric_like = bool(
        re.fullmatch(
            r"(?:Acc\.?\s*\(%\)|Accuracy|BLEU|ROUGE(?:-\d)?|R1/R2/RL|F1|EM|PPL|MNLI(?:-m|-mm)?|"
            r"MRPC|SST-2|QQP|QNLI|RTE|STS-B|WiC|BoolQ|WSC|COPA|MultiRC|ReCoRD|"
            r"SQuAD(?:\s*v?\d\.\d)?|WikiSQL|SAMSum)",
            normalized,
            flags=re.IGNORECASE,
        )
    )
    model_like = bool(
        re.fullmatch(
            r"(?:GPT-?\d(?:\s*\([^)]*\))?|BERT(?:-base|-large)?|RoBERTa(?:-base|-large)?|"
            r"LoRA|Adapter['\"]?|BitFit|PreEmbed|PreLayer|FT)",
            normalized,
            flags=re.IGNORECASE,
        )
    )
    header_like = bool(
        "|" in normalized
        or normalized.lower() in {"parameter", "parameters", "num trainable parameters / task"}
        or bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9._/%()-]*", normalized))
    )

    return numeric_like or metric_like or model_like or header_like


def is_embedded_figure_text_block(block: Dict[str, Any], figure_bbox: List[float]) -> bool:
    if block.get("type") != "text":
        return False

    bbox = block.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    block_area = bbox_area(bbox)
    figure_area = bbox_area(figure_bbox)
    if block_area <= 0.0 or figure_area <= 0.0:
        return False

    overlap_area = bbox_intersection_area(bbox, figure_bbox)
    if overlap_area <= 0.0:
        return False

    overlap_ratio = overlap_area / block_area
    figure_coverage_ratio = overlap_area / figure_area
    center_inside = point_inside_bbox(bbox_center(bbox), figure_bbox, tolerance=2.0)

    return (
        overlap_ratio >= 0.5
        or center_inside
        or figure_coverage_ratio >= 0.12
    )


def collect_figure_embedded_text_cluster_ids(
    page: Dict[str, Any],
    figure_blocks: List[Dict[str, Any]],
) -> set[str]:
    cluster_ids: set[str] = set()
    blocks = page.get("blocks", [])

    for figure in figure_blocks:
        if not is_large_figure(page, figure):
            continue

        figure_bbox = figure.get("bbox") or [0.0, 0.0, 0.0, 0.0]
        expanded_bbox = [
            figure_bbox[0] - 56.0,
            figure_bbox[1] - 28.0,
            figure_bbox[2] + 56.0,
            figure_bbox[3] + 28.0,
        ]

        candidates: List[Dict[str, Any]] = []
        for block in blocks:
            if block.get("type") != "text":
                continue

            text = (block.get("text") or "").strip()
            if len(text.split()) < 12:
                continue

            bbox = block.get("bbox") or [0.0, 0.0, 0.0, 0.0]
            vertical_overlap = bbox_vertical_overlap_ratio(bbox, figure_bbox)
            horizontal_overlap = bbox_horizontal_overlap_ratio(bbox, figure_bbox)
            _, center_y = bbox_center(bbox)
            center_in_vertical_band = figure_bbox[1] - 18.0 <= center_y <= figure_bbox[3] + 18.0

            if not bbox_intersects(bbox, expanded_bbox) and not center_in_vertical_band:
                continue

            if vertical_overlap >= 0.55 or (center_in_vertical_band and horizontal_overlap >= 0.2):
                candidates.append(block)

        if len(candidates) < 4:
            continue

        x_centers = sorted(
            (
                bbox_center(block.get("bbox") or [0.0, 0.0, 0.0, 0.0])[0],
                block.get("block_id", ""),
                block,
            )
            for block in candidates
        )
        column_groups: List[List[Dict[str, Any]]] = []
        current_group: List[Dict[str, Any]] = []
        previous_x = None
        for x_center, _, block in x_centers:
            if previous_x is None or abs(x_center - previous_x) <= 90.0:
                current_group.append(block)
            else:
                column_groups.append(current_group)
                current_group = [block]
            previous_x = x_center
        if current_group:
            column_groups.append(current_group)

        strong_groups = [group for group in column_groups if len(group) >= 2]
        if not strong_groups or len(strong_groups) > 2:
            continue

        for group in strong_groups:
            for block in group:
                block_id = block.get("block_id")
                if block_id:
                    cluster_ids.add(block_id)

    return cluster_ids


def is_near_region(
    bbox: List[float],
    region_bbox: List[float],
    *,
    expanded_bbox: List[float],
    vertical_gap_max: float,
    horizontal_overlap_min: float,
) -> bool:
    vertical_gap = bbox_vertical_gap(bbox, region_bbox)
    horizontal_overlap = bbox_horizontal_overlap_ratio(bbox, region_bbox)
    return (
        bbox_intersects(bbox, expanded_bbox)
        or (vertical_gap <= vertical_gap_max and horizontal_overlap >= horizontal_overlap_min)
    )


def should_remove_figure_fragment(
    block: Dict[str, Any],
    figure_blocks: List[Dict[str, Any]],
) -> bool:
    if block.get("type") not in {"text", "title", "formula"}:
        return False

    bbox = block.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    text = (block.get("text") or "").strip()
    caption_or_note = is_caption_or_table_note(text)
    obvious_noise = is_obvious_noise_text_block(block)
    near_figure_fragment = is_near_figure_fragment_text(block)

    for figure in figure_blocks:
        figure_bbox = figure.get("bbox") or [0.0, 0.0, 0.0, 0.0]
        overlap_ratio = bbox_overlap_ratio(bbox, figure_bbox)
        near_figure = is_near_region(
            bbox,
            figure_bbox,
            expanded_bbox=[
                figure_bbox[0] - 24.0,
                figure_bbox[1] - 24.0,
                figure_bbox[2] + 24.0,
                figure_bbox[3] + 24.0,
            ],
            vertical_gap_max=20.0,
            horizontal_overlap_min=0.35,
        )

        if bbox_inside(bbox, figure_bbox):
            return True

        if is_embedded_figure_text_block(block, figure_bbox):
            return True

        if overlap_ratio >= 0.5:
            return True

        if block.get("type") == "formula" and overlap_ratio > 0.0:
            return True

        if block.get("type") == "text" and near_figure and (
            caption_or_note or obvious_noise or near_figure_fragment
        ):
            return True

    return False


def should_remove_table_fragment(
    block: Dict[str, Any],
    table_blocks: List[Dict[str, Any]],
) -> bool:
    if block.get("type") != "text":
        return False

    bbox = block.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    text = (block.get("text") or "").strip()
    if not text:
        return False

    is_caption_or_summary = is_table_caption_or_summary(block)
    is_content_fragment = is_table_content_fragment_text(block)
    obvious_noise = is_obvious_noise_text_block(block)
    numeric_heavy_sentence = is_numeric_heavy_sentence(text)

    for table in table_blocks:
        table_bbox = table.get("bbox") or [0.0, 0.0, 0.0, 0.0]
        overlap_area = bbox_intersection_area(bbox, table_bbox)
        overlap_ratio = bbox_overlap_ratio(bbox, table_bbox)
        expanded_table_bbox = [
            table_bbox[0] - 40.0,
            table_bbox[1] - 120.0,
            table_bbox[2] + 40.0,
            table_bbox[3] + 56.0,
        ]
        near_table = is_near_region(
            bbox,
            table_bbox,
            expanded_bbox=expanded_table_bbox,
            vertical_gap_max=28.0,
            horizontal_overlap_min=0.3,
        )

        if bbox_inside(bbox, table_bbox):
            return True

        if overlap_ratio >= 0.35:
            return True

        if is_caption_or_summary and not numeric_heavy_sentence:
            continue

        if near_table and numeric_heavy_sentence:
            return True

        if (is_content_fragment or obvious_noise) and (overlap_area > 0.0 or near_table):
            return True

    return False


def post_process_page(page: Dict[str, Any]) -> None:
    blocks = page.get("blocks", [])

    for block in blocks:
        if isinstance(block.get("text"), str):
            if block.get("type") == "formula":
                block["text"] = clean_formula_text(block["text"])
            else:
                block["text"] = clean_general_text(block["text"])

    title_blocks = [block for block in blocks if block.get("type") == "title"]
    first_title_order = None
    if title_blocks:
        first_title_order = min(block.get("reading_order", 10**9) for block in title_blocks)

    figure_blocks = [block for block in blocks if block.get("type") == "figure"]
    table_blocks = [block for block in blocks if block.get("type") == "table"]
    has_table = bool(table_blocks)
    has_table_or_figure = bool(figure_blocks)
    figure_embedded_cluster_ids = collect_figure_embedded_text_cluster_ids(page, figure_blocks)

    processed_blocks: List[Dict[str, Any]] = []
    for block in blocks:
        block_type = block.get("type")

        if (
            first_title_order is not None
            and first_title_order > 1
            and block.get("reading_order", 10**9) < first_title_order
        ):
            continue

        if block.get("block_id") in figure_embedded_cluster_ids:
            continue

        if block_type == "table":
            block["text"] = None

        if block_type == "formula" and block.get("text"):
            block["formula_quality"] = classify_formula_quality(block["text"])
        elif block_type == "formula":
            block["formula_quality"] = "truncated"

        if has_table_or_figure and is_obvious_noise_text_block(block):
            continue

        if has_table and block_type == "text":
            text = (block.get("text") or "").strip()
            if text:
                is_caption_or_summary = is_table_caption_or_summary(block)
                if (
                    (is_table_content_fragment_text(block) or is_obvious_noise_text_block(block))
                    and not is_caption_or_summary
                ):
                    continue
                if is_numeric_heavy_sentence(text) and not text.lower().startswith("table"):
                    continue

        if figure_blocks and should_remove_figure_fragment(
            block,
            figure_blocks=figure_blocks,
        ):
            continue

        if table_blocks and should_remove_table_fragment(
            block,
            table_blocks=table_blocks,
        ):
            continue

        processed_blocks.append(block)

    for reading_order, block in enumerate(processed_blocks, start=1):
        block["reading_order"] = reading_order

    page["blocks"] = processed_blocks


def main() -> None:
    payload = load_json(INPUT_PATH)
    for document in payload.get("documents", []):
        for page in document.get("pages", []):
            post_process_page(page)

    save_json(OUTPUT_PATH, payload)
    print(f"Wrote post-processed document JSON to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
