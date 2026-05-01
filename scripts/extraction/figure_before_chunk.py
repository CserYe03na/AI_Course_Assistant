#!/usr/bin/env python3
"""Clean and enrich extracted figure blocks before chunking."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None


TITLE_TYPES = {"title"}
TEXT_TYPES = {"text", "title", "formula"}
CAPTION_TYPES = {"text", "title"}
QUALITY_VALUES = {"good", "poor", "broken"}
DEFAULT_COURSE_ID = "adl"
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


@dataclass
class FigureRecord:
    block_id: str
    type: str
    page_no: int
    doc_id: str
    bbox: List[float]
    image_path: Optional[str]
    nearby_text_before: Optional[str]
    nearby_text_after: Optional[str]
    section_title: Optional[str]
    visual_description: Optional[str]
    figure_focus: Optional[str]
    keywords: List[str]
    overall_quality: str
    indexable: Optional[bool] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean and enrich extracted figure blocks before chunking."
    )
    parser.add_argument(
        "--course-id",
        default=DEFAULT_COURSE_ID,
        help="Course identifier, for example adl or eods.",
    )
    parser.add_argument("--input-path", help="Optional explicit path to the processed document JSON.")
    parser.add_argument(
        "--output-dir",
        help="Optional output directory. Defaults to data/processed/<course_id>.",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-5.4-mini",
        help="VLM model used for figure semantic enhancement.",
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


def decode_unicode_escapes(text: Optional[str]) -> Optional[str]:
    if text is None or "\\u" not in text:
        return text
    try:
        return text.encode("utf-8").decode("unicode_escape")
    except UnicodeDecodeError:
        return text


def clean_text(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    text = decode_unicode_escapes(text)
    text = text.replace("\u00a0", " ")
    text = text.replace("\u200b", "")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text or None


def normalize_sentence_text(text: Optional[str]) -> Optional[str]:
    text = clean_text(text)
    if not text:
        return None
    text = re.sub(r"\b(?:Figure|Fig)\.?\s*\d+[:.]?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text or None


def word_count(text: Optional[str]) -> int:
    if not text:
        return 0
    return len(re.findall(r"\b\w+\b", text))


def is_probably_header_or_footer(block: Dict[str, Any], page: Dict[str, Any]) -> bool:
    bbox = block.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    page_height = max(1.0, float(page.get("height") or 0.0))
    return bbox[1] <= page_height * 0.08 or bbox[3] >= page_height * 0.92


def bbox_width(bbox: List[float]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0]))


def bbox_center_x(bbox: List[float]) -> float:
    return (float(bbox[0]) + float(bbox[2])) / 2.0


def bbox_vertical_gap(a: List[float], b: List[float]) -> float:
    if a[3] < b[1]:
        return float(b[1]) - float(a[3])
    if b[3] < a[1]:
        return float(a[1]) - float(b[3])
    return 0.0


def bbox_horizontal_overlap_ratio(a: List[float], b: List[float]) -> float:
    overlap = min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0]))
    if overlap <= 0:
        return 0.0
    base = min(bbox_width(a), bbox_width(b))
    if base <= 0:
        return 0.0
    return overlap / base


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


def is_caption_like(block: Dict[str, Any], figure_block: Dict[str, Any], page: Dict[str, Any]) -> bool:
    if block.get("type") not in CAPTION_TYPES:
        return False

    text = normalize_sentence_text(block.get("text"))
    if not text:
        return False

    bbox = block.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    figure_bbox = figure_block.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    gap = bbox_vertical_gap(bbox, figure_bbox)
    overlap_ratio = bbox_horizontal_overlap_ratio(bbox, figure_bbox)
    near_figure = gap <= max(40.0, float(page.get("height", 0.0)) * 0.05)

    if re.match(r"^(figure|fig)\b", text, flags=re.IGNORECASE):
        return near_figure

    short_caption = word_count(text) <= 24 and len(text) <= 180
    similar_alignment = overlap_ratio >= 0.35 or abs(bbox_center_x(bbox) - bbox_center_x(figure_bbox)) <= 60
    just_above_or_below = bbox[3] <= figure_bbox[1] + 5 or bbox[1] >= figure_bbox[3] - 5
    return near_figure and short_caption and similar_alignment and just_above_or_below


def is_good_nearby_text(block: Dict[str, Any], page: Dict[str, Any], figure_block: Dict[str, Any]) -> bool:
    if block.get("type") not in TEXT_TYPES:
        return False
    if is_probably_header_or_footer(block, page):
        return False
    if is_caption_like(block, figure_block=figure_block, page=page):
        return False

    text = normalize_sentence_text(block.get("text"))
    if not text:
        return False
    if word_count(text) < 4:
        return False
    if len(text) < 20:
        return False
    return True


def collect_nearby_text(
    page: Dict[str, Any],
    figure_block: Dict[str, Any],
    *,
    direction: str,
    limit: int = 3,
) -> Optional[str]:
    figure_order = int(figure_block.get("reading_order", 0))
    selected: List[str] = []

    blocks = sort_blocks(page.get("blocks", []))
    iterable = reversed(blocks) if direction == "before" else blocks

    for block in iterable:
        reading_order = int(block.get("reading_order", 0))
        if direction == "before":
            if reading_order >= figure_order:
                continue
        elif reading_order <= figure_order:
            continue

        if not is_good_nearby_text(block, page=page, figure_block=figure_block):
            continue

        text = normalize_sentence_text(block.get("text"))
        if not text:
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
    figure_block: Dict[str, Any],
) -> Optional[str]:
    figure_order = int(figure_block.get("reading_order", 0))
    latest_title: Optional[str] = None

    for block in sort_blocks(page.get("blocks", [])):
        if int(block.get("reading_order", 0)) >= figure_order:
            break
        if block.get("type") not in TITLE_TYPES:
            continue
        candidate = normalize_sentence_text(block.get("text"))
        if candidate:
            latest_title = candidate

    return latest_title or title_history.get(int(page.get("page_no", 0)))


def normalize_keyword(phrase: Optional[str]) -> Optional[str]:
    phrase = clean_text(phrase)
    if not phrase:
        return None

    phrase = phrase.strip(" ,.;:()[]{}")
    if not phrase:
        return None

    lowered = phrase.lower()
    lowered = re.sub(r"\s+", " ", lowered).strip()
    if not lowered:
        return None

    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_\-]*", lowered)
    if not tokens:
        return None
    if len(tokens) > 5:
        return None
    if all(token in STOPWORDS for token in tokens):
        return None
    return phrase


def sanitize_keywords(keywords: List[str], *, min_keywords: int = 5, max_keywords: int = 10) -> List[str]:
    normalized: List[str] = []
    seen: set[str] = set()

    for item in keywords:
        candidate = normalize_keyword(item)
        if not candidate:
            continue
        lowered = candidate.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(candidate)

    if len(normalized) < min_keywords:
        return normalized
    return normalized[:max_keywords]


def extract_keywords(*texts: Optional[str], max_keywords: int = 10) -> List[str]:
    phrases: List[str] = []
    seen: set[str] = set()
    banned_substrings = {
        "this figure",
        "figure appears",
        "nearby text",
        "section",
        "context",
        "about",
        "lecture content",
        "surrounding",
    }

    for text in texts:
        if not text:
            continue
        for raw_phrase in re.findall(r"\b[A-Za-z][A-Za-z0-9_\-]{1,}(?:\s+[A-Za-z0-9_\-]{1,}){0,4}\b", text):
            phrase = clean_text(raw_phrase)
            if not phrase:
                continue
            lowered = phrase.lower()
            tokens = re.findall(r"[A-Za-z][A-Za-z0-9_\-]*", lowered)
            if not tokens:
                continue
            if any(part in lowered for part in banned_substrings):
                continue
            if len(tokens) == 1 and tokens[0] in STOPWORDS:
                continue
            if all(token in STOPWORDS for token in tokens):
                continue
            if len(tokens) >= 2 and sum(token not in STOPWORDS for token in tokens) < 2:
                continue
            if lowered in seen:
                continue
            seen.add(lowered)
            phrases.append(phrase)

    return phrases[:max_keywords]


def build_visual_description(
    *,
    nearby_text_before: Optional[str],
    nearby_text_after: Optional[str],
    section_title: Optional[str],
) -> Optional[str]:
    parts: List[str] = []
    if section_title:
        parts.append(f"This figure appears in the {section_title} section.")
    else:
        parts.append("This figure supports the surrounding lecture content.")

    nearby = clean_text(" ".join(part for part in [nearby_text_before, nearby_text_after] if part))
    if nearby:
        parts.append(f"Nearby text indicates that {nearby}.")

    description = clean_text(" ".join(parts))
    if not description:
        return None
    sentences = re.split(r"(?<=[.!?])\s+", description)
    return clean_text(" ".join(sentences[:4]))


def build_figure_focus(*, visual_description: Optional[str], section_title: Optional[str]) -> Optional[str]:
    for candidate in [section_title, visual_description]:
        if not candidate:
            continue
        cleaned = normalize_sentence_text(candidate)
        if not cleaned:
            continue
        phrase = " ".join(cleaned.split()[:15]).rstrip(".")
        if word_count(phrase) >= 2:
            return phrase
    return None


def init_openai_client() -> Any:
    if OpenAI is None or not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OpenAI client is not available")
    return OpenAI()


def image_to_data_url(image_path: str) -> Optional[str]:
    path = Path(image_path)
    if not path.exists():
        return None
    try:
        image_bytes = path.read_bytes()
    except OSError:
        return None
    mime_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{image_b64}"


def vlm_enrich_figure(
    *,
    client: Optional[Any],
    model: str,
    image_path: Optional[str],
    nearby_text_before: Optional[str],
    nearby_text_after: Optional[str],
    section_title: Optional[str],
) -> Optional[Dict[str, Any]]:
    if client is None:
        return None

    prompt = {
    "task": "VLM-first figure cleaning for knowledge indexing",
    "instructions": [
        "Return valid JSON only.",
        "Use the image as the primary source of truth.",
        "Generate visual_description, figure_focus, keywords, and overall_quality.",

        "This task is for deciding whether a figure should become a standalone knowledge-index chunk.",
        "Only assign good when the image has clear indexing value: it explicitly contains a meaningful knowledge point, definition, concept explanation, model structure, algorithmic process, mathematical/statistical relationship, comparison, taxonomy, mechanism, or reusable learning content.",
        "A good figure should help a learner retrieve or understand a specific concept even without reading the full surrounding page.",
        "Do not assign good merely because the image is visually complete, clean, or related to the section title.",

        "visual_description should be 2 to 3 sentences. Describe only the semantic content of the image: concepts, entities, processes, relations, comparisons, examples, trends, definitions, formulas, or mechanisms. Start directly with the content, not with a wrapper phrase. Avoid mentioning layout unless layout itself is semantically important. Avoid low-information words such as 'diagram', 'figure', 'image', or 'panel' unless necessary for disambiguation.",
        
        "figure_focus should be a short phrase of about 5 to 15 words that captures the central indexable knowledge point. If the figure does not contain an indexable knowledge point, describe its limited role, such as 'decorative illustration', 'example-only visual', 'context-dependent screenshot', or 'unclear damaged figure'.",

        "keywords should contain up to 4 terms or short phrases. Use fewer keywords when the image supports only a small number of clearly grounded terms. Prefer precise technical terms, definitions, model names, variables, formulas, axis names, module names, or concept labels that are directly visible or strongly grounded in the image. Do not add broad section-level labels, nearby prose fragments, decorative labels, or speculative terms just to fill the list.",

        "overall_quality must be one of good, poor, broken.",

        "Use good ONLY when all of the following are true:",
        "1. The image is complete enough and visually readable.",
        "2. The main semantic meaning can be recovered reliably from the image itself.",
        "3. The image contains explicit learning or indexing value, such as a definition, knowledge point, conceptual mechanism, model architecture, algorithm flow, mathematical relation, statistical idea, data trend, comparison, or taxonomy.",
        "4. The figure is broadly consistent with the section_title.",
        "5. The image could reasonably be retrieved as a standalone figure chunk for learning, review, or question answering.",

        "Use poor when the image is readable but should not enter the main knowledge index as a standalone figure chunk. This includes images that are mainly examples, exercises, auxiliary illustrations, decorative visuals, screenshots without generalizable knowledge, motivational images, icons, photos, UI fragments, isolated tables or plots with weak learning value, or figures whose meaning depends heavily on surrounding text.",
        "Use poor for example-only content even if it is clear and relevant to the section title.",
        "Use poor for supporting material that illustrates a concept but does not itself state or encode a reusable concept, definition, mechanism, relationship, or conclusion.",
        "Use poor for figures with weak or partial standalone semantic value, ambiguity, mild to heavy noise, weak relevance to the section_title, or insufficiently stable semantics for indexing.",

        "Use broken for blank, severely miscropped, badly damaged, semantically unrecoverable, unreadable, or clearly irrelevant/unusable figures.",
        "Use broken when the main semantic content cannot be identified reliably from the image itself.",
        "Use broken when the image appears corrupted, mostly empty, cut off in a way that destroys meaning, or when any interpretation would require guessing.",

        "Do not use good for decorative images, icons, stock photos, example-only visuals, screenshots used only as auxiliary material, or figures that provide little recoverable learning content.",
        "Do not guess unreadable small text.",
        "Use nearby_text_before and nearby_text_after only as weak auxiliary context. They may help disambiguate, but they cannot turn an example-only, decorative, or semantically weak image into good.",
        "Return JSON with keys: visual_description, figure_focus, keywords, overall_quality.",
    ],
    "inputs": {
        "image_path": image_path,
        "nearby_text_before": nearby_text_before,
        "nearby_text_after": nearby_text_after,
        "section_title": section_title,
    },
    }

    try:
        content: List[Dict[str, Any]] = [{"type": "input_text", "text": json.dumps(prompt, ensure_ascii=False)}]
        if image_path:
            data_url = image_to_data_url(image_path)
            if data_url:
                content.append({"type": "input_image", "image_url": data_url})

        response = client.responses.create(
            model=model,
            input=[{"role": "user", "content": content}],
        )
    except Exception:
        return None

    output_text = getattr(response, "output_text", None)
    if not output_text:
        return None

    try:
        payload = json.loads(output_text)
    except json.JSONDecodeError:
        return None

    keywords = payload.get("keywords") or []
    if not isinstance(keywords, list):
        keywords = []

    overall_quality = clean_text(payload.get("overall_quality"))
    if overall_quality not in QUALITY_VALUES:
        overall_quality = None

    return {
        "visual_description": clean_text(payload.get("visual_description")),
        "figure_focus": clean_text(payload.get("figure_focus")),
        "keywords": keywords,
        "overall_quality": overall_quality,
    }


def count_document_figures(document: Dict[str, Any]) -> int:
    return sum(
        1
        for page in document.get("pages", [])
        for block in page.get("blocks", [])
        if block.get("type") == "figure"
    )


def load_existing_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    try:
        payload = load_json(path)
    except json.JSONDecodeError:
        return []

    figures = payload.get("figures")
    if not isinstance(figures, list):
        return []
    return figures


def build_output_payload(
    *,
    payload: Dict[str, Any],
    input_path: Path,
    course_id: str,
    all_figures: List[Dict[str, Any]],
    per_document_counts: Dict[str, int],
) -> Dict[str, Any]:
    return {
        "course_id": payload.get("course_id", course_id),
        "course_name": payload.get("course_name"),
        "source_document_path": str(input_path.as_posix()),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "figure_count": len(all_figures),
        "indexable_figure_count": sum(1 for item in all_figures if item.get("indexable")),
        "documents": per_document_counts,
        "figures": all_figures,
    }


def persist_progress(
    *,
    output_json: Path,
    payload: Dict[str, Any],
    input_path: Path,
    course_id: str,
    all_figures: List[Dict[str, Any]],
    per_document_counts: Dict[str, int],
) -> None:
    output_payload = build_output_payload(
        payload=payload,
        input_path=input_path,
        course_id=course_id,
        all_figures=all_figures,
        per_document_counts=per_document_counts,
    )
    save_json(output_json, output_payload)


def enrich_figure_block(
    *,
    doc_id: str,
    page: Dict[str, Any],
    figure_block: Dict[str, Any],
    title_history: Dict[int, Optional[str]],
    llm_client: Optional[Any],
    llm_model: str,
) -> FigureRecord:
    nearby_text_before = collect_nearby_text(page=page, figure_block=figure_block, direction="before")
    nearby_text_after = collect_nearby_text(page=page, figure_block=figure_block, direction="after")
    section_title = get_section_title(title_history=title_history, page=page, figure_block=figure_block)

    vlm_result = vlm_enrich_figure(
        image_path=figure_block.get("image_path"),
        client=llm_client,
        model=llm_model,
        nearby_text_before=nearby_text_before,
        nearby_text_after=nearby_text_after,
        section_title=section_title,
    )

    visual_description = (vlm_result or {}).get("visual_description") or build_visual_description(
        nearby_text_before=nearby_text_before,
        nearby_text_after=nearby_text_after,
        section_title=section_title,
    )
    figure_focus = (vlm_result or {}).get("figure_focus") or build_figure_focus(
        visual_description=visual_description,
        section_title=section_title,
    )
    keywords = (vlm_result or {}).get("keywords") or extract_keywords(
        figure_focus,
        visual_description,
        section_title,
    )
    overall_quality = (vlm_result or {}).get("overall_quality")

    return FigureRecord(
        block_id=str(figure_block.get("block_id")),
        type="figure",
        page_no=int(page.get("page_no", 0)),
        doc_id=doc_id,
        bbox=list(figure_block.get("bbox") or [0.0, 0.0, 0.0, 0.0]),
        image_path=figure_block.get("image_path"),
        nearby_text_before=nearby_text_before,
        nearby_text_after=nearby_text_after,
        section_title=section_title,
        visual_description=visual_description,
        figure_focus=figure_focus,
        keywords=keywords,
        overall_quality=overall_quality,
        # indexable=overall_quality == "good",
    )


def enrich_document_figures(
    document: Dict[str, Any],
    llm_client: Optional[Any],
    llm_model: str,
    *,
    doc_index: int,
    total_docs: int,
    completed_block_ids: set[str],
    figure_records_by_id: Dict[str, Dict[str, Any]],
    payload: Dict[str, Any],
    input_path: Path,
    course_id: str,
    output_json: Path,
    all_figures: List[Dict[str, Any]],
    per_document_counts: Dict[str, int],
) -> List[FigureRecord]:
    title_history = get_title_history(document)
    figures: List[FigureRecord] = []
    total_figures = count_document_figures(document)
    processed_figures = 0
    reused_figures = 0
    doc_id = str(document.get("doc_id"))

    print(
        f"[{doc_index}/{total_docs}] Processing document {doc_id} "
        f"with {total_figures} figures",
        flush=True,
    )

    for page in document.get("pages", []):
        for block in sort_blocks(page.get("blocks", [])):
            if block.get("type") != "figure":
                continue
            processed_figures += 1
            block_id = str(block.get("block_id"))
            print(
                f"  [{processed_figures}/{total_figures}] "
                f"page {page.get('page_no')} block {block_id}",
                flush=True,
            )
            if block_id in completed_block_ids:
                reused_figures += 1
                existing_record = figure_records_by_id.get(block_id)
                if existing_record is not None:
                    figures.append(FigureRecord(**existing_record))
                print("    resumed from existing output", flush=True)
                continue

            figures.append(
                enrich_figure_block(
                    doc_id=doc_id,
                    page=page,
                    figure_block=block,
                    title_history=title_history,
                    llm_client=llm_client,
                    llm_model=llm_model,
                )
            )
            record_dict = figures[-1].__dict__
            figure_records_by_id[block_id] = record_dict
            completed_block_ids.add(block_id)
            all_figures[:] = list(figure_records_by_id.values())
            per_document_counts[doc_id] = len(figures)
            persist_progress(
                output_json=output_json,
                payload=payload,
                input_path=input_path,
                course_id=course_id,
                all_figures=all_figures,
                per_document_counts=per_document_counts,
            )
            print("    saved progress", flush=True)

    print(
        f"[{doc_index}/{total_docs}] Finished document {doc_id} "
        f"(reused {reused_figures}, processed {total_figures - reused_figures})",
        flush=True,
    )
    return figures


def main() -> None:
    args = parse_args()

    input_path = resolve_input_path(args.course_id, args.input_path)
    output_dir = Path(args.output_dir) if args.output_dir else Path("data/processed") / args.course_id
    output_json = output_dir / f"{args.course_id}_figures_cleaned.json"
    payload = load_json(input_path)
    llm_client = init_openai_client()
    existing_figures = load_existing_records(output_json)
    figure_records_by_id = {
        str(record.get("block_id")): record
        for record in existing_figures
        if record.get("block_id")
    }
    completed_block_ids = set(figure_records_by_id.keys())
    documents = payload.get("documents", [])
    total_docs = len(documents)
    total_figures = sum(count_document_figures(document) for document in documents)

    print(f"Loading figures from {input_path}", flush=True)
    print(f"Found {total_docs} documents and {total_figures} figures to process", flush=True)
    if completed_block_ids:
        print(
            f"Resuming from existing output with {len(completed_block_ids)} completed figures",
            flush=True,
        )

    all_figures: List[Dict[str, Any]] = list(figure_records_by_id.values())
    per_document_counts: Dict[str, int] = {}
    for record in all_figures:
        doc_id = str(record.get("doc_id"))
        per_document_counts[doc_id] = per_document_counts.get(doc_id, 0) + 1

    for doc_index, document in enumerate(documents, start=1):
        doc_id = str(document.get("doc_id"))
        enriched_figures = enrich_document_figures(
            document=document,
            llm_client=llm_client,
            llm_model=args.llm_model,
            doc_index=doc_index,
            total_docs=total_docs,
            completed_block_ids=completed_block_ids,
            figure_records_by_id=figure_records_by_id,
            payload=payload,
            input_path=input_path,
            course_id=args.course_id,
            output_json=output_json,
            all_figures=all_figures,
            per_document_counts=per_document_counts,
        )
        per_document_counts[doc_id] = len(enriched_figures)
        all_figures[:] = list(figure_records_by_id.values())

    persist_progress(
        output_json=output_json,
        payload=payload,
        input_path=input_path,
        course_id=args.course_id,
        all_figures=all_figures,
        per_document_counts=per_document_counts,
    )

    print(
        f"Completed figure cleaning: {len(all_figures)} records, "
        f"{sum(1 for item in all_figures if item.get('indexable'))} indexable",
        flush=True,
    )
    print(f"Wrote {len(all_figures)} figure records to {output_json}", flush=True)


if __name__ == "__main__":
    main()
