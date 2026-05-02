#!/usr/bin/env python3
"""Post-process EODS document JSON.

EODS slides contain many notebook screenshots. Docling often extracts those
screenshots twice: once as a figure crop and once as text inside the figure
bbox. For downstream retrieval we keep code-like notebook text as ``code``
instead of dropping all figure-embedded text.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

INPUT_PATH = Path("data/processed/eods/eods_document.json")
OUTPUT_PATH = Path("data/processed/eods/eods_processed.json")


def clean_text(text: str) -> str:
    text = text.replace("▶", " ")
    text = text.replace("\u00a0", " ")
    text = text.replace("\u200b", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


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


def has_jupyter_prompt(text: str) -> bool:
    return bool(re.search(r"\b(?:in|out)\s*\[\d+\]\s*:?", text, flags=re.IGNORECASE))


def clean_jupyter_prompt_text(text: str) -> str:
    text = re.sub(r"\bIn\s*\[\d+\]\s*:?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bOut\s*\[\d+\]\s*:?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bCell\s+In\s*\[\d+\]", "Cell", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def looks_like_python_or_notebook_code(text: str) -> bool:
    if not text:
        return False

    code_patterns = [
        r"\bimport\s+[A-Za-z_][\w.]*",
        r"\bfrom\s+[A-Za-z_][\w.]*\s+import\b",
        r"\bdef\s+[A-Za-z_]\w*\s*\(",
        r"\bclass\s+[A-Za-z_]\w*\s*[:(]",
        r"\bfor\s+[A-Za-z_]\w*\s+in\b.*:",
        r"\b(?:while|if|elif)\s+[^:]+:",
        r"\b(?:else|try)\s*:",
        r"\bexcept(?:\s+[A-Za-z_][\w.]*)?(?:\s+as\s+[A-Za-z_]\w*)?\s*:",
        r"\bwith\s+[^:]+:",
        r"\breturn\b",
        r"\bassert\b",
        r"\bprint\s*\(",
        r"\b(?:pd|np|plt|sns)\.",
        r"\bdf(?:_[A-Za-z0-9_]+)?\.",
        r"\b(?:fig|ax)\s*,?\s*(?:ax|axes)?\s*=",
        r"\b(?:fig|ax|axes)\.",
        r"\.[A-Za-z_]\w*\s*\(",
        r"[A-Za-z_]\w*\s*(?:=|\+=|-=|\*=|/=)\s*[^=]",
        r"\[[^\]]+\]",
        r"\{[^}]+:[^}]+\}",
        r"\b(?:Traceback|Error|Exception|KeyError|ValueError|TypeError|AssertionError)\b",
        r"\bdtype:\s*[A-Za-z_]",
        r"\bName:\s*[A-Za-z_].*,\s*dtype:",
        r"\bRangeIndex:\s*\d+\s+entries",
        r"<class\s+'[^']+'>",
    ]
    return any(re.search(pattern, text) for pattern in code_patterns)


def looks_like_shell_code(text: str) -> bool:
    shell_patterns = [
        r"\$\s*(?:conda|python|pip|jupyter|ipython|ls|cd|pwd|which)\b",
        r"\b(?:conda|pip)\s+(?:create|install|activate|list|env)\b",
        r"\bpython\s+[-/\w.]+",
        r"%%?\s*(?:timeit|bash|capture|run)\b",
        r"!\s*(?:echo|python|pip|ls|cat|pwd|which)\b",
    ]
    return any(re.search(pattern, text) for pattern in shell_patterns)


def looks_like_explanatory_sentence(text: str) -> bool:
    words = text.split()
    if len(words) < 5:
        return False
    starts_like_prose = bool(
        re.match(
            r"^(?:to|for|example|often|missing|this|these|we|you|the|a|an|also)\b",
            text,
            flags=re.IGNORECASE,
        )
    )
    sentence_punctuation = text.endswith((".", ":", "?"))
    prose_verbs = bool(
        re.search(
            r"\b(?:is|are|was|were|use|uses|used|means|compares|checks|called|represented)\b",
            text,
            flags=re.IGNORECASE,
        )
    )
    return starts_like_prose and (sentence_punctuation or prose_verbs or len(words) >= 6)


def is_code(text: str) -> bool:
    stripped = clean_text(text)
    if not stripped:
        return False

    if has_jupyter_prompt(stripped):
        return True
    if looks_like_shell_code(stripped):
        return True
    if re.search(r"\b(?:Traceback|Error|Exception)\b", stripped):
        return True

    codeish = looks_like_python_or_notebook_code(stripped)
    if not codeish:
        return False

    # Avoid turning prose with a single inline assignment example into code.
    if looks_like_explanatory_sentence(stripped):
        strong_signals = [
            r"\b(?:pd|np|plt|sns)\.\w+\s*\(",
            r"\b(?:import|def|class|return|assert|print)\b",
            r"\b(?:Traceback|Error|Exception)\b",
            r"\$\s*\w+",
        ]
        return any(re.search(pattern, stripped) for pattern in strong_signals)

    return True



def clean_code_noise(text: str) -> str:
    text = clean_jupyter_prompt_text(text)

    # Drop one specific chart label only when it was mixed into code blocks.
    text = re.sub(r"fare amount \(dollars\)", "", text, flags=re.IGNORECASE)

    # Remove OCR'd notebook line-number tails, but preserve comments and code.
    text = re.sub(r"(?:\s+\d+){4,}\s*$", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = normalize_code_spacing(text)
    return text


def normalize_code_spacing(text: str) -> str:
    text = re.sub(
        r"!\s+(echo|python|pip|ls|cat|pwd|which)\b",
        r"!\1",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"%%\s+(timeit|bash|capture)\b",
        r"%%\1",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"%\s+(run|timeit|matplotlib|load_ext|capture|bash)\b",
        r"%\1",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"/\s+(tmp|Users|home|var|private)\s*/\s*", r"/\1/", text)
    text = re.sub(r"\s+([,.;:)\]\}])", r"\1", text)
    text = re.sub(r"([(\[\{])\s+", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


CODE_SEGMENT_MARKER_RE = re.compile(
    r"(?P<comment>#\s+)"
    r"|(?P<code>"
    r"!\s*(?:echo|python|pip|ls|cat|pwd|which)\b"
    r"|%%?\s*(?:run|timeit|bash|capture|matplotlib|load_ext)\b"
    r"|\bimport\s+[A-Za-z_][\w.]*"
    r"|\bfrom\s+[A-Za-z_][\w.]*\s+import\b"
    r"|\bdef\s+[A-Za-z_]\w*\s*\("
    r"|\bclass\s+[A-Za-z_]\w*\s*[:(]"
    r"|\bfor\s+[A-Za-z_]\w*\s+in\b"
    r"|\b(?:while|if|elif|else|try|except|with)\b"
    r"|\b(?:return|assert)\b"
    r"|\bprint\s*\("
    r"|\b(?:pd|np|plt|sns)\."
    r"|\bdf(?:_[A-Za-z0-9_]+)?\."
    r"|\b(?:fig|ax|axes)\b\s*(?:,|=|\.)"
    r"|[A-Za-z_]\w*\s*(?:=|\+=|-=|\*=|/=)\s*[^=]"
    r")",
    flags=re.IGNORECASE,
)


def classify_code_segment(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "output"
    if stripped.startswith("#"):
        return "comment"
    if looks_like_shell_code(stripped) or looks_like_python_or_notebook_code(stripped):
        return "code"
    return "output"


def split_code_segments(text: str) -> List[Dict[str, str]]:
    matches = list(CODE_SEGMENT_MARKER_RE.finditer(text))
    if not matches:
        return [{"kind": classify_code_segment(text), "text": text}]

    segments: List[Dict[str, str]] = []

    def append_segment(kind: str, value: str) -> None:
        value = clean_text(value)
        if not value:
            return
        if segments and segments[-1]["kind"] == kind:
            previous = segments[-1]["text"]
            joiner = ""
            if (
                previous
                and value
                and not previous.endswith((" ", "'", '"', "(", "[", "{", "/", "."))
                and not value.startswith((".", ",", ";", ":", ")", "]", "}"))
            ):
                joiner = " "
            segments[-1]["text"] = clean_text(f"{previous}{joiner}{value}")
            return
        segments.append({"kind": kind, "text": value})

    if matches[0].start() > 0:
        append_segment("output", text[: matches[0].start()])

    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        segment_text = text[start:end]
        kind = "comment" if match.lastgroup == "comment" else "code"
        append_segment(kind, segment_text)

    return segments


def bbox_area(bbox: List[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def bbox_inside(inner: List[float], outer: List[float], tolerance: float = 2.0) -> bool:
    return (
        inner[0] >= outer[0] - tolerance
        and inner[1] >= outer[1] - tolerance
        and inner[2] <= outer[2] + tolerance
        and inner[3] <= outer[3] + tolerance
    )


def is_inside_any_figure(block: Dict[str, Any], figures: List[Dict[str, Any]]) -> bool:
    bbox = block.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    return any(bbox_inside(bbox, figure.get("bbox") or [0.0, 0.0, 0.0, 0.0]) for figure in figures)


def is_axis_noise(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    axis_words = {"frequency", "count", "density", "probability"}
    if stripped.lower() in axis_words:
        return True
    if re.fullmatch(r"-+", stripped):
        return True

    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?\s*-?", stripped):
        return True
    if re.fullmatch(r"-\s*[-+]?\d+(?:\.\d+)?", stripped):
        return True
    if re.fullmatch(r"(?:[-+]?\d+(?:\.\d+)?\s*){2,}", stripped):
        return True
    if re.fullmatch(r"(?:[-+]?\d+(?:\.\d+)?\s*-?\s*){2,}", stripped):
        return True

    words = text.split()
    if len(words) <= 5:
        digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
        if digit_ratio > 0.4:
            return True

    if re.search(r"(\d+\s+){4,}", text):
        return True
    return False


def is_table_fragment(text: str) -> bool:
    return (
        bool(re.fullmatch(r"\d+(\.\d+)?", text))
        or ("|" in text and text.count("|") >= 2)
    )


def is_short_noise(text: str) -> bool:
    return len(text) < 5


def is_figure_ocr_noise(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if is_axis_noise(stripped) or is_table_fragment(stripped) or is_short_noise(stripped):
        return True
    if len(stripped.split()) <= 3 and not is_code(stripped):
        return True

    letters = sum(ch.isalpha() for ch in stripped)
    digits = sum(ch.isdigit() for ch in stripped)
    punctuation = sum(not ch.isalnum() and not ch.isspace() for ch in stripped)
    if letters < 4 and digits + punctuation > letters:
        return True
    return False


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def process() -> None:
    data = load_json(INPUT_PATH)
    summary = {
        "kept_code": 0,
        "kept_figure_code": 0,
        "removed_figure_noise": 0,
        "removed_text_noise": 0,
        "removed_empty_code": 0,
    }

    for doc in data["documents"]:
        for page in doc["pages"]:
            blocks = page["blocks"]
            figures = [block for block in blocks if block.get("type") == "figure"]
            new_blocks = []

            for block in blocks:
                text = block.get("text") or ""
                block_type = block.get("type")

                if block_type == "text":
                    inside_figure = is_inside_any_figure(block, figures)
                    text = clean_text(text)

                    if is_code(text):
                        text = clean_code_noise(text)
                        if not text:
                            summary["removed_empty_code"] += 1
                            continue

                        block["type"] = "code"
                        block["contains_code"] = True
                        block["text"] = text
                        block["code_segments"] = split_code_segments(text)
                        if inside_figure:
                            block["extracted_from_figure"] = True
                            summary["kept_figure_code"] += 1
                        summary["kept_code"] += 1
                        new_blocks.append(block)
                        continue

                    if inside_figure and is_figure_ocr_noise(text):
                        summary["removed_figure_noise"] += 1
                        continue

                    if is_table_fragment(text) and len(text.split()) <= 5:
                        summary["removed_text_noise"] += 1
                        continue

                    if is_axis_noise(text):
                        summary["removed_text_noise"] += 1
                        continue

                    if is_short_noise(text):
                        summary["removed_text_noise"] += 1
                        continue

                    block["text"] = text

                if block_type == "formula" and text:
                    text = clean_formula_text(text)
                    block["formula_quality"] = classify_formula_quality(text)
                    block["text"] = text

                new_blocks.append(block)

            for i, b in enumerate(new_blocks):
                b["reading_order"] = i + 1

            page["blocks"] = new_blocks

    save_json(OUTPUT_PATH, data)
    print(f"Saved to {OUTPUT_PATH}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    process()
