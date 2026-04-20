#!/usr/bin/env python3
"""Build cleaned formula records from processed course JSON."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


WORKSPACE_ROOT = Path.cwd()
LOCAL_FORMULA_VENV_NAME = ".venv_formula"
SUPPORTED_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
DEFAULT_COURSE_ID = "adl"
QUALITY_VALUES = {"good", "poor", "broken"}
_PIX2TEX_MODEL: Any = None
_PIX2TEX_INIT_ERROR: Optional[str] = None
FORMULA_LLM_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "formula_latex": {
            "type": ["string", "null"],
            "description": "A single LaTeX expression or null when the formula structure is not recoverable.",
        },
        "formula_focus": {
            "type": "string",
            "description": "A short 3 to 10 word phrase naming the formula or its immediate mathematical role.",
        },
        "formula_explanation": {
            "type": "string",
            "description": "A 1 to 3 sentence explanation of what the visible formula means or does.",
        },
        "overall_quality": {
            "type": "string",
            "enum": ["good", "poor", "broken"],
            "description": "Quality label for downstream use.",
        },
    },
    "required": [
        "formula_latex",
        "formula_focus",
        "formula_explanation",
        "overall_quality",
    ],
    "additionalProperties": False,
}


def bootstrap_local_formula_env(workspace_root: Path) -> None:
    venv_root = workspace_root / LOCAL_FORMULA_VENV_NAME
    if not venv_root.exists():
        return

    lib_dir = venv_root / "lib"
    if not lib_dir.exists():
        return

    for site_packages in lib_dir.glob("python*/site-packages"):
        site_packages_str = str(site_packages)
        if site_packages_str not in sys.path:
            sys.path.insert(0, site_packages_str)


bootstrap_local_formula_env(WORKSPACE_ROOT)


try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None


try:
    from post_process_document import classify_formula_quality, clean_formula_text
except ImportError:
    def clean_formula_text(text: str) -> str:
        text = text.replace("▶", " ")
        text = text.replace("\u00a0", " ")
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(
            r"(?<![A-Za-z\\])(?:[A-Za-z]\s+){2,}[A-Za-z](?![A-Za-z])",
            lambda match: "".join(match.group(0).split()),
            text,
        )
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
        if re.search(r"(?:[A-Za-z]\s+){3,}[A-Za-z](?![A-Za-z])", stripped):
            return "noisy"
        if "\\ " in stripped or re.search(r"\\\\\s+[A-Za-z]", stripped):
            return "noisy"
        if brace_diff == 1 or paren_diff == 1:
            return "noisy"
        return "good"


CORE_RECORD_FIELDS = (
    "block_id",
    "type",
    "page_no",
    "doc_id",
    "bbox",
    "formula_path",
    "nearby_text_before",
    "nearby_text_after",
    "section_title",
    "formula_latex",
    "formula_focus",
    "formula_explanation",
    "overall_quality",
)

LOW_SIGNAL_CONTEXT = {
    "or",
    "for example",
    "for example,",
    "e.g.",
    "example",
    "thus",
    "then",
    "so that",
}

SYMBOL_LATEX_REPLACEMENTS = [
    ("⊺", r"^\top"),
    ("×", r" \times "),
    ("·", r" \cdot "),
    ("∑", r"\sum "),
    ("∏", r"\prod "),
    ("√", r"\sqrt "),
    ("∈", r" \in "),
    ("∼", r" \sim "),
    ("∞", r"\infty"),
    ("α", r"\alpha"),
    ("β", r"\beta"),
    ("γ", r"\gamma"),
    ("δ", r"\delta"),
    ("θ", r"\theta"),
    ("λ", r"\lambda"),
    ("μ", r"\mu"),
    ("σ", r"\sigma"),
    ("π", r"\pi"),
    ("ℓ", r"\ell"),
]

KNOWN_LATEX_COMMANDS = {
    "Big",
    "Delta",
    "Gamma",
    "Lambda",
    "Pi",
    "Psi",
    "Sigma",
    "alpha",
    "atop",
    "bar",
    "begin",
    "beta",
    "big",
    "bigl",
    "bigr",
    "cdot",
    "delta",
    "displaystyle",
    "ell",
    "end",
    "epsilon",
    "equiv",
    "exp",
    "frac",
    "gamma",
    "hat",
    "in",
    "infty",
    "int",
    "lambda",
    "ldots",
    "left",
    "log",
    "mathbb",
    "mathbf",
    "mathcal",
    "mathrm",
    "mathsf",
    "max",
    "mid",
    "min",
    "mu",
    "operatorname",
    "partial",
    "pi",
    "prod",
    "qquad",
    "quad",
    "right",
    "rightarrow",
    "sigma",
    "sim",
    "sqrt",
    "sum",
    "text",
    "theta",
    "times",
    "top",
}

SUSPICIOUS_LATEX_PATTERNS = [
    r"\\begin\{array\}",
    r"\\end\{array\}",
    r"\{\{",
    r"&\{\{",
    r"\\rho\(",
    r"\\mathbb\{X\}\\operatorname",
    r"\\mathrm\{\\tiny",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--course-id",
        default=DEFAULT_COURSE_ID,
        help="Course identifier, for example adl or eods.",
    )
    parser.add_argument(
        "--input",
        "--input-path",
        dest="input_path",
        help="Optional explicit path to the processed document JSON.",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        help="Optional explicit output path. Overrides --output-dir when set.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional output directory. Defaults to data/processed/<course_id>.",
    )
    parser.add_argument("--doc-id", help="Optional document id filter, e.g. adl_lecture_10")
    parser.add_argument(
        "--llm-model",
        default="gpt-5.4-mini",
        help="LLM model used for formula semantic enhancement.",
    )
    parser.add_argument(
        "--disable-llm",
        action="store_true",
        help="Disable LLM semantic enhancement even if API access is configured.",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=WORKSPACE_ROOT,
        help="Workspace root used to resolve local formula image paths.",
    )
    parser.add_argument(
        "--disable-pix2tex",
        action="store_true",
        help="Disable pix2tex and use the OCR text fallback only.",
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


def resolve_output_path(
    course_id: str,
    *,
    explicit_output_path: Optional[str],
    output_dir: Optional[str],
) -> Path:
    if explicit_output_path:
        return Path(explicit_output_path)

    base_dir = Path(output_dir) if output_dir else Path("data/processed") / course_id
    return base_dir / f"{course_id}_formula_cleaned.json"


def init_openai_client(disable_llm: bool) -> Optional[Any]:
    if disable_llm or OpenAI is None or not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        return OpenAI()
    except Exception:
        return None


def clean_line(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\u00a0", " ")).strip()


def strip_markdown_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        stripped = re.sub(r"^```[A-Za-z0-9_-]*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped)
    return stripped.strip()


def parse_llm_json_output(output_text: str) -> Optional[Dict[str, Any]]:
    candidate = strip_markdown_code_fence(output_text)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        sanitized = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", candidate)
        try:
            payload = json.loads(sanitized)
        except json.JSONDecodeError:
            return None
    return payload if isinstance(payload, dict) else None


def image_to_data_url(image_path: Path) -> Optional[str]:
    if not image_path.exists():
        return None
    try:
        image_bytes = image_path.read_bytes()
    except OSError:
        return None
    mime_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{image_b64}"


def format_exception(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def sort_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(blocks, key=lambda block: int(block.get("reading_order", 10**9)))


def is_low_signal_context(text: str) -> bool:
    stripped = clean_line(text)
    if not stripped:
        return True
    lowered = stripped.lower()
    if lowered in LOW_SIGNAL_CONTEXT:
        return True
    if len(stripped) <= 3:
        return True
    words = re.findall(r"[A-Za-z]+", stripped)
    if len(words) <= 2 and len(stripped) <= 14:
        return True
    return False


def clean_context_sentence(text: str) -> str:
    return clean_line(text).rstrip(" .;:,")


def pick_context(blocks: List[Dict[str, Any]], index: int, direction: int) -> Optional[str]:
    snippets: List[str] = []
    char_budget = 280
    current = index + direction

    while 0 <= current < len(blocks) and len(snippets) < 3 and char_budget > 0:
        candidate = blocks[current]
        if candidate.get("type") in {"text", "title"}:
            text = clean_context_sentence(candidate.get("text") or "")
            if text and not is_low_signal_context(text):
                if direction < 0:
                    snippets.insert(0, text)
                else:
                    snippets.append(text)
                char_budget -= len(text)
        current += direction

    if not snippets:
        return None
    return " ".join(snippets)


def find_section_title(
    pages: List[Dict[str, Any]],
    page_index: int,
    reading_order: int,
) -> Optional[str]:
    current_page = pages[page_index]
    current_titles = [
        clean_line(block.get("text") or "")
        for block in sort_blocks(current_page.get("blocks", []))
        if block.get("type") == "title" and int(block.get("reading_order", 10**9)) < reading_order
    ]
    if current_titles:
        return current_titles[-1]

    for previous_page_index in range(page_index, -1, -1):
        titles = [
            clean_line(block.get("text") or "")
            for block in sort_blocks(pages[previous_page_index].get("blocks", []))
            if block.get("type") == "title"
        ]
        if titles:
            return titles[-1]
    return None


def build_image_index(workspace_root: Path) -> Dict[str, Path]:
    image_index: Dict[str, Path] = {}
    for suffix in SUPPORTED_IMAGE_SUFFIXES:
        for path in workspace_root.rglob(f"*{suffix}"):
            image_index.setdefault(path.name, path)
    return image_index


def resolve_formula_image_path(
    block: Dict[str, Any],
    workspace_root: Path,
    image_index: Dict[str, Path],
) -> Optional[Path]:
    image_path = block.get("image_path")
    block_id = block.get("block_id")
    candidates: List[Path] = []

    if image_path:
        raw_path = Path(image_path)
        candidates.append(raw_path)
        if not raw_path.is_absolute():
            candidates.append(workspace_root / raw_path)
        candidates.append(workspace_root / raw_path.name)

    if block_id:
        for suffix in SUPPORTED_IMAGE_SUFFIXES:
            candidates.append(workspace_root / f"{block_id}{suffix}")

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    if image_path:
        indexed = image_index.get(Path(image_path).name)
        if indexed:
            return indexed.resolve()

    if block_id:
        for suffix in SUPPORTED_IMAGE_SUFFIXES:
            indexed = image_index.get(f"{block_id}{suffix}")
            if indexed:
                return indexed.resolve()
    return None


def strip_latex_wrappers(text: str) -> str:
    stripped = text.strip()
    wrapper_pairs = [
        ("$$", "$$"),
        ("$", "$"),
        (r"\[", r"\]"),
        (r"\(", r"\)"),
    ]
    for start, end in wrapper_pairs:
        if stripped.startswith(start) and stripped.endswith(end):
            stripped = stripped[len(start) : len(stripped) - len(end)].strip()
            break
    return stripped


def normalize_latex_candidate(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    stripped = strip_latex_wrappers(clean_line(text))
    if not stripped:
        return None
    return stripped


def has_render_blocking_latex_errors(formula_latex: Optional[str]) -> bool:
    if not formula_latex:
        return True
    if formula_latex.count("{") != formula_latex.count("}"):
        return True
    if formula_latex.count(r"\left") != formula_latex.count(r"\right"):
        return True
    if formula_latex.count(r"\begin") != formula_latex.count(r"\end"):
        return True

    commands = set(re.findall(r"\\([A-Za-z]+)", formula_latex))
    unknown_commands = {command for command in commands if command not in KNOWN_LATEX_COMMANDS}
    return bool(unknown_commands)


def has_suspicious_latex_output(text: Optional[str]) -> bool:
    if not text:
        return False
    return any(re.search(pattern, text) for pattern in SUSPICIOUS_LATEX_PATTERNS)


def apply_generic_latex_repairs(formula: str) -> str:
    repaired = clean_formula_text(formula)
    for source, target in SYMBOL_LATEX_REPLACEMENTS:
        repaired = repaired.replace(source, target)

    repaired = repaired.replace("...", r" \ldots ")
    repaired = re.sub(r"\s+/+\s*", " / ", repaired)
    repaired = re.sub(r"\s*\|\s*", " | ", repaired)
    repaired = re.sub(r"([A-Za-z])\s+\(", r"\1(", repaired)
    repaired = re.sub(r"\(\s+", "(", repaired)
    repaired = re.sub(r"\s+\)", ")", repaired)
    repaired = re.sub(r"\s+([=+\-<>])\s+", r" \1 ", repaired)
    repaired = re.sub(r"\s*,\s*", ", ", repaired)
    repaired = re.sub(r"\s*;\s*", "; ", repaired)
    repaired = re.sub(r"([A-Za-z\\]+)\s+\^\\top", r"\1^\\top", repaired)
    repaired = re.sub(r"\\(sum|prod)\s+([A-Za-z])\s*=\s*([0-9A-Za-z]+)", r"\\\1_{\2=\3}", repaired)
    repaired = re.sub(r"([A-Za-z])\s+\((i|j|k|n|N|\d+)\)", r"\1^{(\2)}", repaired)
    repaired = re.sub(r"\(\s*1 -\s*", "(1 - ", repaired)
    repaired = re.sub(r"\s{2,}", " ", repaired).strip()
    return repaired


def normalize_formula_for_latex(
    text: str,
    context_before: Optional[str],
    context_after: Optional[str],
    section_title: Optional[str],
) -> Optional[str]:
    del context_before, context_after, section_title

    stripped = clean_formula_text(text)
    if not stripped:
        return None

    formula = apply_generic_latex_repairs(stripped)
    if not re.search(r"[=<>\\]|[\^_(){}\[\]]", formula) and len(formula) < 6:
        return None
    return formula


def get_pix2tex_model() -> Any:
    global _PIX2TEX_MODEL, _PIX2TEX_INIT_ERROR

    if _PIX2TEX_MODEL is not None:
        return _PIX2TEX_MODEL
    if _PIX2TEX_INIT_ERROR is not None:
        raise RuntimeError(_PIX2TEX_INIT_ERROR)

    try:
        from PIL import Image  # noqa: F401
        from pix2tex.cli import LatexOCR
    except Exception as exc:
        _PIX2TEX_INIT_ERROR = f"{type(exc).__name__}: {exc}"
        raise RuntimeError(_PIX2TEX_INIT_ERROR) from exc

    try:
        _PIX2TEX_MODEL = LatexOCR()
    except Exception as exc:
        _PIX2TEX_INIT_ERROR = f"{type(exc).__name__}: {exc}"
        raise RuntimeError(_PIX2TEX_INIT_ERROR) from exc

    return _PIX2TEX_MODEL


def extract_latex_with_pix2tex(image_path: Path) -> tuple[Optional[str], Optional[str]]:
    try:
        from PIL import Image
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"

    try:
        model = get_pix2tex_model()
        with Image.open(image_path) as image:
            raw_output = model(image)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"

    normalized = normalize_latex_candidate(str(raw_output))
    if not normalized:
        return None, "pix2tex returned empty LaTeX"
    return normalized, None


def derive_formula_focus(
    text: str,
    context_before: Optional[str],
    context_after: Optional[str],
    section_title: Optional[str],
) -> Optional[str]:
    for candidate in [section_title, context_before, context_after]:
        cleaned = clean_line(candidate or "")
        if cleaned:
            words = cleaned.split()
            return " ".join(words[:8])

    cleaned_text = clean_formula_text(text)
    if cleaned_text:
        return " ".join(cleaned_text.split()[:8])
    return None


def derive_formula_normalized(
    text: str,
    context_before: Optional[str],
    context_after: Optional[str],
    section_title: Optional[str],
) -> Optional[str]:
    return derive_formula_focus(text, context_before, context_after, section_title)


def derive_formula_explanation(
    focus: Optional[str],
    text: str,
    nearby_text_before: Optional[str],
    nearby_text_after: Optional[str],
    section_title: Optional[str],
) -> Optional[str]:
    if focus:
        return f"This formula is about {focus}."
    if nearby_text_before:
        return f"This formula is introduced by the nearby text: {nearby_text_before}."
    if nearby_text_after:
        return f"This formula is followed by the explanation: {nearby_text_after}."
    if section_title:
        return f"This formula appears in the {section_title} section."
    if text:
        return f"This record stores the formula text: {text}."
    return None


def llm_enrich_formula(
    *,
    client: Optional[Any],
    model: str,
    image_path: Optional[Path],
    raw_formula_text: str,
    fallback_formula_latex: Optional[str],
    pix2tex_formula_latex: Optional[str],
    context_before: Optional[str],
    context_after: Optional[str],
    section_title: Optional[str],
) -> Optional[Dict[str, Any]]:
    if client is None:
        raise RuntimeError("OpenAI client is not available")

    prompt = {
        "task": "Formula cleaning and semantic enrichment",
        "instructions": [
            "Return valid JSON only.",
            "Use the formula image as the primary source of truth.",
            "formula_latex must be based mainly on pix2tex_formula_latex and fallback_formula_latex, checked against the image.",
            "The LLM may only make light local fixes to candidate LaTeX and must not reconstruct the formula mainly from scratch.",
            "Use raw_formula_text, section_title, nearby_text_before, and nearby_text_after only as auxiliary context.",
            "Do not guess unseen symbols, bounds, exponents, subscripts, superscripts, or hidden terms unless they are clearly supported by the image.",
            "Do not rewrite the whole formula to make it look more standard.",
            "Allowed light fixes: brace balancing, \\left/\\right balancing, \\begin/\\end balancing, removing obvious garbage commands, fixing high-confidence local OCR errors.",
            "If the main structure is not recoverable with confidence, set formula_latex to null.",
            "If formula_latex is null or unreliable, still try to generate formula_focus and formula_explanation from the image itself.",
            "When formula_latex is unreliable, base formula_focus and formula_explanation on the visible mathematical content in the image, not on the section title alone.",
            "formula_latex should be a single LaTeX expression without code fences or dollar delimiters.",
            "formula_focus should be a short phrase of 3 to 10 words naming the formula or its immediate mathematical role.",
            "formula_explanation should be 1 to 3 sentences explaining what the visible formula means or does.",
            "formula_explanation must describe the visible formula itself first, not the broader lecture topic.",
            "Do not turn formula_explanation into a general lecture explanation.",
            "Do not let nearby context override the visible semantic content of the formula.",
            "If only a local fragment is visible, keep the explanation local and do not upgrade it into a full formula interpretation.",
            "overall_quality must be one of good, poor, broken.",
            "Use good only when formula_latex has clear main structure, is mostly renderable, and formula_focus plus formula_explanation stably capture the core visible meaning with little or no uncertainty.",
            "Use poor when formula_latex is incomplete or noisy, but the visible formula still provides enough stable semantic content for retrieval, and formula_explanation remains mostly local, concrete, and non-speculative.",
            "Use broken when the visible content is only a fragment, local symbol group, or cropped sub-expression, and the full formula meaning is not stably recoverable.",
            "Use broken when formula_focus or formula_explanation depends mainly on context-based guessing rather than the visible formula itself.",
            "Use broken when the explanation indicates cropping, partial visibility, local recoverability only, or uncertainty about the overall formula meaning.",
            "Use broken when formula_latex captures only a local piece but not enough structure to support a reliable formula-level interpretation.",
            "Do not use poor for cases where only a partial fragment is visible and the broader formula role is inferred mainly from context.",
            "If uncertain between poor and broken, choose broken when the visible expression is only a fragment rather than a stable formula unit.",
            "Return JSON with exactly these keys: formula_latex, formula_focus, formula_explanation, overall_quality."
        ],
        "inputs": {
            "raw_formula_text": raw_formula_text,
            "fallback_formula_latex": fallback_formula_latex,
            "pix2tex_formula_latex": pix2tex_formula_latex,
            "context_before": context_before,
            "context_after": context_after,
            "section_title": section_title,
            "image_path": str(image_path) if image_path else None,
        },
    }

    content: List[Dict[str, Any]] = [{"type": "input_text", "text": json.dumps(prompt, ensure_ascii=False)}]
    if image_path:
        data_url = image_to_data_url(image_path)
        if data_url:
            content.append({"type": "input_image", "image_url": data_url})

    response = None
    last_error: Optional[str] = None
    for _ in range(2):
        try:
            response = client.responses.create(
                model=model,
                input=[{"role": "user", "content": content}],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "formula_semantic_record",
                        "schema": FORMULA_LLM_SCHEMA,
                        "strict": True,
                    },
                    "verbosity": "low",
                },
            )
            break
        except Exception as exc:
            last_error = format_exception(exc)

    if response is None:
        raise RuntimeError(last_error or "Unknown LLM request failure")

    output_text = getattr(response, "output_text", None)
    if not output_text:
        raise RuntimeError("OpenAI response did not include output_text")

    payload = parse_llm_json_output(output_text)
    if payload is None:
        raise RuntimeError("Failed to parse LLM JSON output")

    formula_latex = normalize_latex_candidate(payload.get("formula_latex"))
    formula_focus = clean_line(payload.get("formula_focus") or payload.get("formula_normalized") or "") or None
    formula_explanation = clean_line(payload.get("formula_explanation") or "") or None
    overall_quality = clean_line(payload.get("overall_quality") or "") or None

    quality_aliases = {
        "high": "good",
        "clean": "good",
        "usable": "poor",
        "noisy": "poor",
    }
    overall_quality = quality_aliases.get(overall_quality, overall_quality)
    if overall_quality not in QUALITY_VALUES:
        overall_quality = None

    return {
        "formula_latex": formula_latex,
        "formula_focus": formula_focus,
        "formula_explanation": formula_explanation,
        "overall_quality": overall_quality,
    }


def has_usable_formula_semantics(
    focus: Optional[str],
    explanation: Optional[str],
    *,
    section_title: Optional[str],
) -> bool:
    cleaned_focus = clean_line(focus or "")
    cleaned_explanation = clean_line(explanation or "")
    if not cleaned_focus or not cleaned_explanation:
        return False

    if len(cleaned_focus) < 4:
        return False
    if len(cleaned_explanation) < 32:
        return False

    generic_starts = (
        "this formula is about",
        "this formula appears in",
        "this record stores",
        "this formula is introduced by",
        "this formula is followed by",
    )
    lowered_explanation = cleaned_explanation.lower()
    if lowered_explanation.startswith(generic_starts):
        return False

    if section_title:
        normalized_title = clean_line(section_title).lower()
        if cleaned_focus.lower() == normalized_title:
            return False
        if normalized_title and normalized_title in lowered_explanation and len(cleaned_explanation.split()) <= 10:
            return False

    return True


def derive_overall_quality(
    *,
    llm_overall_quality: str,
    formula_quality: str,
    text: str,
    resolved_image_path: Optional[Path],
    formula_latex: Optional[str],
    formula_latex_failed_renderability_check: bool,
    formula_focus: Optional[str],
    formula_explanation: Optional[str],
    section_title: Optional[str],
) -> str:
    if not text and resolved_image_path is None:
        return "broken"

    semantics_usable = has_usable_formula_semantics(
        formula_focus,
        formula_explanation,
        section_title=section_title,
    )
    latex_usable = bool(
        formula_latex
        and not formula_latex_failed_renderability_check
        and not has_suspicious_latex_output(formula_latex)
    )

    if formula_quality == "truncated" and not semantics_usable:
        return "broken"

    if not latex_usable:
        if not semantics_usable:
            return "broken"
        return "poor"

    if llm_overall_quality == "broken":
        return "broken"
    if llm_overall_quality == "poor":
        return "poor"
    return "good"


def derive_chunk_strategy(overall_quality: str) -> str:
    if overall_quality == "good":
        return "context_explanation_latex_section_title"
    if overall_quality == "poor":
        return "context_explanation_section_title"
    return "skip"


def build_retrieval_text(
    overall_quality: str,
    formula_latex: Optional[str],
    focus: Optional[str],
    explanation: Optional[str],
    nearby_text_before: Optional[str],
    nearby_text_after: Optional[str],
    section_title: Optional[str],
) -> str:
    if overall_quality == "broken":
        return ""

    parts: List[str] = []
    if section_title:
        parts.append(f"Section: {section_title}.")
    if focus:
        parts.append(f"Formula focus: {focus}.")
    if explanation:
        parts.append(explanation.rstrip(".") + ".")
    if nearby_text_before:
        parts.append(f"Before the formula: {nearby_text_before}.")
    if nearby_text_after:
        parts.append(f"After the formula: {nearby_text_after}.")
    if overall_quality == "good" and formula_latex:
        parts.append(f"Formula: {formula_latex}.")
    return " ".join(parts)


def build_chunk_text(
    *,
    overall_quality: str,
    section_title: Optional[str],
    focus: Optional[str],
    explanation: Optional[str],
    nearby_text_before: Optional[str],
    nearby_text_after: Optional[str],
    formula_latex: Optional[str],
) -> Optional[str]:
    if overall_quality == "broken":
        return None

    parts: List[str] = []
    if section_title:
        parts.append(f"Section: {section_title}.")
    if focus:
        parts.append(f"Formula focus: {focus}.")
    if explanation:
        parts.append(explanation.rstrip(".") + ".")
    if nearby_text_before:
        parts.append(f"Before the formula: {nearby_text_before}.")
    if nearby_text_after:
        parts.append(f"After the formula: {nearby_text_after}.")
    if overall_quality == "good" and formula_latex:
        parts.append(f"Formula: {formula_latex}.")
    return " ".join(parts) if parts else None


def choose_formula_latex(
    *,
    llm_formula_latex: Optional[str],
    pix2tex_formula_latex: Optional[str],
    fallback_formula_latex: Optional[str],
) -> tuple[Optional[str], str]:
    candidates = [
        ("llm", llm_formula_latex),
        ("pix2tex", pix2tex_formula_latex),
        ("rules", fallback_formula_latex),
    ]

    for source, candidate in candidates:
        if not candidate:
            continue
        if has_suspicious_latex_output(candidate):
            continue
        if has_render_blocking_latex_errors(candidate):
            continue
        return candidate, source

    for source, candidate in candidates:
        if candidate:
            return candidate, source
    return None, "none"


def select_formula_output_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    return {field: record.get(field) for field in CORE_RECORD_FIELDS}


def build_formula_record(
    doc: Dict[str, Any],
    pages: List[Dict[str, Any]],
    page_index: int,
    block_index: int,
    workspace_root: Path,
    image_index: Dict[str, Path],
    pix2tex_enabled: bool,
    llm_client: Optional[Any],
    llm_model: str,
) -> Dict[str, Any]:
    page = pages[page_index]
    block = page["blocks"][block_index]

    text = clean_formula_text(block.get("text") or "")
    formula_quality = block.get("formula_quality") or classify_formula_quality(text)
    nearby_text_before = pick_context(page["blocks"], block_index, direction=-1)
    nearby_text_after = pick_context(page["blocks"], block_index, direction=1)
    section_title = find_section_title(pages, page_index, int(block.get("reading_order", 10**9)))
    resolved_image_path = resolve_formula_image_path(block, workspace_root, image_index)
    formula_path = block.get("image_path") or (str(resolved_image_path) if resolved_image_path else None)

    fallback_formula_latex = normalize_formula_for_latex(
        text,
        nearby_text_before,
        nearby_text_after,
        section_title,
    )

    pix2tex_formula_latex: Optional[str] = None
    if pix2tex_enabled and resolved_image_path is not None:
        pix2tex_formula_latex, _ = extract_latex_with_pix2tex(resolved_image_path)

    llm_result = llm_enrich_formula(
        client=llm_client,
        model=llm_model,
        image_path=resolved_image_path,
        raw_formula_text=text,
        fallback_formula_latex=fallback_formula_latex,
        pix2tex_formula_latex=pix2tex_formula_latex,
        context_before=nearby_text_before,
        context_after=nearby_text_after,
        section_title=section_title,
    )
    formula_latex, formula_latex_source = choose_formula_latex(
        llm_formula_latex=(llm_result or {}).get("formula_latex"),
        pix2tex_formula_latex=pix2tex_formula_latex,
        fallback_formula_latex=fallback_formula_latex,
    )
    formula_latex_failed_renderability_check = has_render_blocking_latex_errors(formula_latex)

    formula_focus = (llm_result or {}).get("formula_focus")
    formula_explanation = (llm_result or {}).get("formula_explanation")
    llm_overall_quality = (llm_result or {}).get("overall_quality")
    if not formula_focus or not formula_explanation or llm_overall_quality not in QUALITY_VALUES:
        raise ValueError(
            "LLM enrichment returned incomplete formula metadata for "
            f"block {block.get('block_id')}: "
            f"formula_focus={formula_focus!r}, "
            f"formula_explanation={formula_explanation!r}, "
            f"overall_quality={llm_overall_quality!r}"
        )
    overall_quality = derive_overall_quality(
        llm_overall_quality=llm_overall_quality,
        formula_quality=formula_quality,
        text=text,
        resolved_image_path=resolved_image_path,
        formula_latex=formula_latex,
        formula_latex_failed_renderability_check=formula_latex_failed_renderability_check,
        formula_focus=formula_focus,
        formula_explanation=formula_explanation,
        section_title=section_title,
    )

    record = {
        "block_id": block.get("block_id"),
        "type": "formula",
        "page_no": page.get("page_no"),
        "doc_id": doc.get("doc_id"),
        "bbox": block.get("bbox"),
        "formula_path": formula_path,
        "nearby_text_before": nearby_text_before,
        "nearby_text_after": nearby_text_after,
        "section_title": section_title,
        "formula_latex": formula_latex if overall_quality != "broken" else None,
        "formula_focus": formula_focus,
        "formula_explanation": formula_explanation,
        "overall_quality": overall_quality,
    }
    return select_formula_output_fields(record)


def build_formula_payload(
    payload: Dict[str, Any],
    doc_id: Optional[str],
    workspace_root: Path,
    pix2tex_enabled: bool,
    llm_client: Optional[Any],
    llm_model: str,
    input_path: Path,
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    image_index = build_image_index(workspace_root)

    for document in payload.get("documents", []):
        if doc_id and document.get("doc_id") != doc_id:
            continue

        pages = document.get("pages", [])
        for page_index, page in enumerate(pages):
            for block_index, block in enumerate(page.get("blocks", [])):
                if block.get("type") != "formula":
                    continue
                records.append(
                    build_formula_record(
                        document,
                        pages,
                        page_index,
                        block_index,
                        workspace_root=workspace_root,
                        image_index=image_index,
                        pix2tex_enabled=pix2tex_enabled,
                        llm_client=llm_client,
                        llm_model=llm_model,
                    )
                )

    return {
        "course_id": payload.get("course_id"),
        "course_name": payload.get("course_name"),
        "source_type": payload.get("source_type"),
        "source_document_path": str(input_path.as_posix()),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "workspace_root": str(workspace_root),
        "pix2tex_enabled": pix2tex_enabled,
        "llm_enabled": llm_client is not None,
        "llm_model": llm_model if llm_client is not None else None,
        "formula_count": len(records),
        "records": records,
    }


def main() -> None:
    args = parse_args()
    workspace_root = args.workspace_root.resolve()
    input_path = resolve_input_path(args.course_id, args.input_path)
    output_path = resolve_output_path(
        args.course_id,
        explicit_output_path=args.output_path,
        output_dir=args.output_dir,
    )

    payload = load_json(input_path)
    llm_client = init_openai_client(args.disable_llm)
    formula_payload = build_formula_payload(
        payload,
        args.doc_id,
        workspace_root=workspace_root,
        pix2tex_enabled=not args.disable_pix2tex,
        llm_client=llm_client,
        llm_model=args.llm_model,
        input_path=input_path,
    )
    save_json(output_path, formula_payload)
    print(f"Loading formulas from {input_path}")
    print(
        f"Wrote {formula_payload['formula_count']} formula records to {output_path} "
        f"(llm_enabled={formula_payload['llm_enabled']}, pix2tex_enabled={formula_payload['pix2tex_enabled']})"
    )


if __name__ == "__main__":
    main()
