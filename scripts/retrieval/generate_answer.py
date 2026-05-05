#!/usr/bin/env python3
"""Generate a source-grounded answer from retrieved course chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

import retrieve_faiss_bm25 as retrieval


DEFAULT_GENERATION_MODEL = "gpt-5.4-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def clean_text(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    normalized = " ".join(str(text).split())
    return normalized or None


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def location(result: Dict[str, Any]) -> str:
    doc_id = result.get("doc_id") or "unknown_doc"
    if result.get("level") == "semantic":
        page_start = result.get("page_start")
        page_end = result.get("page_end")
        if page_start and page_end and page_start != page_end:
            return f"{doc_id}, pages {page_start}-{page_end}"
        if page_start:
            return f"{doc_id}, page {page_start}"
    page_no = result.get("page_no")
    return f"{doc_id}, page {page_no}" if page_no else doc_id


def render_math_spans(math_spans: Any) -> List[str]:
    rendered: List[str] = []
    if not isinstance(math_spans, list):
        return rendered
    for span in math_spans:
        if not isinstance(span, dict):
            continue
        math_name = clean_text(span.get("math_name"))
        normalized = clean_text(span.get("normalized"))
        raw = clean_text(span.get("raw"))
        if math_name and normalized:
            rendered.append(f"{math_name}: {normalized}")
        elif normalized:
            rendered.append(normalized)
        elif raw:
            rendered.append(raw)
    return rendered


def render_generation_payload(payload: Any) -> List[str]:
    if not isinstance(payload, dict):
        return []

    parts: List[str] = []
    section_title = clean_text(payload.get("section_title"))
    if section_title:
        parts.append(f"Section: {section_title}")

    for key in ("text_cleaned", "text", "formula_focus", "formula_explanation", "formula_latex", "figure_focus", "visual_description"):
        value = clean_text(payload.get(key))
        if value:
            parts.append(value)

    keywords = payload.get("keywords")
    if isinstance(keywords, list):
        keyword_text = ", ".join(filter(None, (clean_text(str(item)) for item in keywords)))
        if keyword_text:
            parts.append(f"Keywords: {keyword_text}")

    math_span_text = render_math_spans(payload.get("math_spans"))
    if math_span_text:
        parts.append("Math: " + "; ".join(math_span_text))

    return parts


def result_context_text(result: Dict[str, Any]) -> str:
    content = result.get("content_for_generation")
    parts: List[str] = []

    if isinstance(content, dict) and isinstance(content.get("atomic_chunks"), list):
        for atomic in content.get("atomic_chunks") or []:
            if not isinstance(atomic, dict):
                continue
            atomic_type = clean_text(atomic.get("chunk_type"))
            rendered = render_generation_payload(atomic.get("content_for_generation"))
            if rendered:
                prefix = f"{atomic_type}: " if atomic_type else ""
                parts.append(prefix + " ".join(rendered))
        for auxiliary in content.get("auxiliary_chunks") or []:
            if not isinstance(auxiliary, dict):
                continue
            rendered = render_generation_payload(auxiliary.get("content_for_generation"))
            if rendered:
                parts.append("auxiliary: " + " ".join(rendered))
    else:
        parts.extend(render_generation_payload(content))

    if not parts:
        text = clean_text(result.get("text"))
        if text:
            parts.append(text)

    return clean_text(" ".join(parts)) or ""


def is_low_value_result(result: Dict[str, Any]) -> bool:
    text = clean_text(result_context_text(result)) or ""
    token_count = len(text.split())
    if result.get("level") == "semantic":
        return token_count < 5
    return token_count < 6 and result.get("chunk_type") not in {"formula", "text_inline_math"}


def source_key(result: Dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    return (
        result.get("doc_id"),
        result.get("page_no") or result.get("page_start"),
        result.get("page_end"),
        result.get("text"),
    )


def select_context_results(results: List[Dict[str, Any]], max_sources: int) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    seen: set[tuple[Any, Any, Any, Any]] = set()

    semantic_first = sorted(
        results,
        key=lambda item: (
            0 if item.get("level") == "semantic" else 1,
            -(item.get("combined_score") or 0.0),
        ),
    )

    for result in semantic_first:
        key = source_key(result)
        if key in seen:
            continue
        if is_low_value_result(result) and len(selected) >= max(2, max_sources // 2):
            continue
        selected.append(result)
        seen.add(key)
        if len(selected) >= max_sources:
            break

    if not selected:
        return results[:max_sources]
    return selected


def retrieve_results(
    *,
    course_id: str,
    index_dir: Path,
    query: str,
    target: str,
    top_k: int,
    candidate_k: int,
    embedding_model: str,
    rrf_k: int,
    faiss_weight: float,
    bm25_weight: float,
) -> List[Dict[str, Any]]:
    query_vector = retrieval.embed_query(query, embedding_model)
    levels = ["atomic", "semantic"] if target == "both" else [target]

    all_faiss_results: List[Dict[str, Any]] = []
    all_bm25_results: List[Dict[str, Any]] = []

    for level in levels:
        index, metadata = retrieval.load_index(index_dir, course_id, level)
        all_faiss_results.extend(
            retrieval.faiss_search(
                query_vector=query_vector,
                index=index,
                metadata=metadata,
                level=level,
                top_k=candidate_k,
            )
        )
        all_bm25_results.extend(
            retrieval.bm25_search(
                query=query,
                metadata=metadata,
                level=level,
                top_k=candidate_k,
            )
        )

    combined = retrieval.combine_results(
        faiss_results=all_faiss_results,
        bm25_results=all_bm25_results,
        top_k=top_k,
        rrf_k=rrf_k,
        faiss_weight=faiss_weight,
        bm25_weight=bm25_weight,
    )
    return [retrieval.format_output_result(result) for result in combined]


def build_context_block(results: List[Dict[str, Any]]) -> str:
    blocks: List[str] = []
    for index, result in enumerate(results, start=1):
        source_id = f"S{index}"
        context_text = result_context_text(result)
        blocks.append(
            "\n".join(
                [
                    f"[{source_id}] {location(result)}",
                    f"retrieval_level: {result.get('level')}",
                    f"chunk_type: {result.get('chunk_type')}",
                    f"content: {context_text}",
                ]
            )
        )
    return "\n\n".join(blocks)


def build_prompt(query: str, context_block: str) -> str:
    return f"""Answer the student question using only the course context below.

Rules:
- If the context is not enough, say what is missing instead of guessing.
- Cite the source labels inline, for example [S1] or [S2].
- End with a short "Sources" list containing the cited labels and page locations.
- Prefer clear, student-friendly explanations over copying slide text.
- Ignore obvious OCR fragments when better context is available.

Student question:
{query}

Course context:
{context_block}
"""


def generate_answer(*, client: OpenAI, model: str, query: str, context_results: List[Dict[str, Any]]) -> str:
    context_block = build_context_block(context_results)
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "You are an AI course assistant. You answer from retrieved lecture "
                    "chunks and cite the source pages. Do not invent course content."
                ),
            },
            {
                "role": "user",
                "content": build_prompt(query, context_block),
            },
        ],
    )
    output_text = getattr(response, "output_text", None)
    if not output_text:
        raise RuntimeError("OpenAI response did not include output_text")
    return output_text.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course-id", default="eods")
    parser.add_argument("--index-dir", default="data/retrieval")
    parser.add_argument("--query", required=True)
    parser.add_argument("--target", choices=["atomic", "semantic", "both"], default="both")
    parser.add_argument("--top-k", type=int, default=8, help="Retrieved results to consider before context filtering.")
    parser.add_argument("--context-k", type=int, default=6, help="Sources to pass to the generation model.")
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--generation-model", default=DEFAULT_GENERATION_MODEL)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--faiss-weight", type=float, default=1.0)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--output-json", help="Optional path to save the answer, sources, and raw retrieval results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.context_k <= 0:
        raise ValueError("--context-k must be positive")
    if args.candidate_k < args.top_k:
        raise ValueError("--candidate-k must be >= --top-k")

    retrieved_results = retrieve_results(
        course_id=args.course_id,
        index_dir=Path(args.index_dir),
        query=args.query,
        target=args.target,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        embedding_model=args.embedding_model,
        rrf_k=args.rrf_k,
        faiss_weight=args.faiss_weight,
        bm25_weight=args.bm25_weight,
    )
    context_results = select_context_results(retrieved_results, args.context_k)
    answer = generate_answer(
        client=OpenAI(),
        model=args.generation_model,
        query=args.query,
        context_results=context_results,
    )

    print(f"Question: {args.query}\n")
    print(answer)
    print("\nRetrieved sources used for generation:")
    for index, result in enumerate(context_results, start=1):
        print(f"[S{index}] {location(result)} id={result.get('id')} level={result.get('level')}")

    if args.output_json:
        save_json(
            Path(args.output_json),
            {
                "course_id": args.course_id,
                "query": args.query,
                "target": args.target,
                "top_k": args.top_k,
                "context_k": args.context_k,
                "candidate_k": args.candidate_k,
                "embedding_model": args.embedding_model,
                "generation_model": args.generation_model,
                "answer": answer,
                "context_results": context_results,
                "retrieved_results": retrieved_results,
            },
        )
        print(f"\nWrote generation output to {args.output_json}")


if __name__ == "__main__":
    main()
