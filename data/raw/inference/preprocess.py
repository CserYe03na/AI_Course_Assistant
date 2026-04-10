"""
Week 1: Course Material Preprocessing Pipeline
================================================
Converts PDF course materials (slides, notes, homework) into
a structured knowledge base (JSONL) with metadata.

Usage:
    python preprocess.py --config config.json
    python preprocess.py --input_dir ./data/raw --output ./data/processed/knowledge_base.jsonl

Dependencies:
    pip install pdfplumber pypdf
"""
import sys
import re
import json
import argparse
import logging
from pathlib import Path

import pdfplumber
from pypdf import PdfReader

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. PDF Text Extraction
# ─────────────────────────────────────────────

def extract_pages(pdf_path: str) -> list[dict]:
    """
    Extract text from each page of a PDF.
    Returns a list of {page_number, raw_text} dicts.
    Falls back from pdfplumber to pypdf if needed.
    """
    pages = []
    path = Path(pdf_path)

    if not path.exists():
        log.error(f"File not found: {pdf_path}")
        return pages

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                pages.append({
                    "page_number": i + 1,
                    "raw_text": text,
                })
        log.info(f"  Extracted {len(pages)} pages from {path.name} (pdfplumber)")
    except Exception as e:
        log.warning(f"  pdfplumber failed ({e}), falling back to pypdf...")
        try:
            reader = PdfReader(pdf_path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                pages.append({
                    "page_number": i + 1,
                    "raw_text": text,
                })
            log.info(f"  Extracted {len(pages)} pages from {path.name} (pypdf)")
        except Exception as e2:
            log.error(f"  Both extractors failed for {path.name}: {e2}")

    return pages


# ─────────────────────────────────────────────
# 2. Text Cleaning
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean extracted PDF text:
    - Remove standalone page numbers
    - Collapse excessive whitespace and blank lines
    - Strip leading/trailing whitespace
    """
    # Remove lines that are only digits (page numbers)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    # Collapse 3+ consecutive newlines into 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Collapse multiple spaces/tabs into one space
    text = re.sub(r'[ \t]{2,}', ' ', text)
    # Strip
    text = text.strip()
    return text


def is_empty_chunk(text: str, min_words: int = 5) -> bool:
    """Return True if the text has fewer than min_words meaningful words."""
    return len(text.split()) < min_words


# ─────────────────────────────────────────────
# 3. Chunking Strategies
# ─────────────────────────────────────────────

def chunk_by_page(pages: list[dict], config: dict) -> list[dict]:
    """
    Slide strategy: one chunk per page.
    Short pages are merged with the next page.
    """
    min_words = config.get("slide_min_words_per_page", 10)
    chunks = []
    buffer_text = ""
    buffer_pages = []

    for page in pages:
        cleaned = clean_text(page["raw_text"])

        if not cleaned:
            continue

        word_count = len(cleaned.split())

        if word_count < min_words:
            # Accumulate short pages into buffer
            buffer_text += (" " + cleaned).strip()
            buffer_pages.append(page["page_number"])
        else:
            # Flush buffer if any
            if buffer_text:
                full_text = (buffer_text + " " + cleaned).strip()
                buffer_pages.append(page["page_number"])
                chunks.append({
                    "text": full_text,
                    "page_numbers": buffer_pages[:],
                })
                buffer_text = ""
                buffer_pages = []
            else:
                chunks.append({
                    "text": cleaned,
                    "page_numbers": [page["page_number"]],
                })

    # Flush remaining buffer
    if buffer_text:
        chunks.append({
            "text": buffer_text,
            "page_numbers": buffer_pages,
        })

    return chunks


def chunk_by_words(pages: list[dict], config: dict) -> list[dict]:
    """
    Notes/HW strategy: sliding window over all text, chunked by word count.
    """
    chunk_size  = config.get("chunk_size_words", 300)
    overlap     = config.get("chunk_overlap_words", 50)

    # Concatenate all pages into one stream, keeping page boundary markers
    full_text = ""
    for page in pages:
        cleaned = clean_text(page["raw_text"])
        if cleaned:
            full_text += cleaned + " "

    words = full_text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])
        if not is_empty_chunk(chunk_text):
            chunks.append({
                "text": chunk_text,
                "page_numbers": [],   # word-level chunking loses page info
            })
        start += chunk_size - overlap

    return chunks


# ─────────────────────────────────────────────
# 4. Build Knowledge Base
# ─────────────────────────────────────────────

SOURCE_TYPE_STRATEGY = {
    "slide":    chunk_by_page,
    "note":     chunk_by_words,
    "homework": chunk_by_words,
}

def process_document(doc_config: dict, global_config: dict, chunk_id_start: int) -> list[dict]:
    """
    Process a single PDF document into knowledge-base chunks.

    doc_config keys:
        path          (str)  : path to PDF file
        title         (str)  : human-readable lecture/document title
        source_type   (str)  : "slide" | "note" | "homework"
    """
    path        = doc_config["path"]
    title       = doc_config.get("title", Path(path).stem)
    source_type = doc_config.get("source_type", "note")

    log.info(f"Processing [{source_type}] {title} ...")

    pages = extract_pages(path)
    if not pages:
        log.warning(f"  No pages extracted — skipping {path}")
        return []

    strategy = SOURCE_TYPE_STRATEGY.get(source_type, chunk_by_words)
    raw_chunks = strategy(pages, global_config)

    kb_chunks = []
    for i, raw in enumerate(raw_chunks):
        if is_empty_chunk(raw["text"]):
            continue
        kb_chunks.append({
            "chunk_id": f"chunk_{chunk_id_start + i:05d}",
            "text": raw["text"],
            "metadata": {
                "source":       path,
                "source_type":  source_type,
                "lecture_title": title,
                "page_numbers": raw.get("page_numbers", []),
            }
        })

    log.info(f"  → {len(kb_chunks)} chunks created")
    return kb_chunks


def build_knowledge_base(doc_configs: list[dict], global_config: dict) -> list[dict]:
    """Process all documents and return a flat list of chunks."""
    all_chunks = []
    for doc in doc_configs:
        chunks = process_document(doc, global_config, chunk_id_start=len(all_chunks))
        all_chunks.extend(chunks)

    log.info(f"\nTotal chunks in knowledge base: {len(all_chunks)}")
    return all_chunks


# ─────────────────────────────────────────────
# 5. Save / Load Knowledge Base
# ─────────────────────────────────────────────

def save_jsonl(chunks: list[dict], output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    log.info(f"Saved {len(chunks)} chunks → {output_path}")


def load_jsonl(path: str) -> list[dict]:
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


# ─────────────────────────────────────────────
# 6. Quick Statistics
# ─────────────────────────────────────────────

def print_stats(chunks: list[dict]):
    from collections import Counter
    type_counter = Counter(c["metadata"]["source_type"] for c in chunks)
    word_counts  = [len(c["text"].split()) for c in chunks]

    print("\n── Knowledge Base Statistics ──────────────────")
    print(f"  Total chunks : {len(chunks)}")
    print(f"  By type      : {dict(type_counter)}")
    print(f"  Words/chunk  : min={min(word_counts)}, "
          f"avg={sum(word_counts)//len(word_counts)}, max={max(word_counts)}")
    print("────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────
# 7. Auto-discover PDFs from a directory
# ─────────────────────────────────────────────

def discover_pdfs(input_dir: str) -> list[dict]:
    """
    Scan a directory for PDFs and guess source_type from filename keywords.
    Naming conventions expected:
        lecture*.pdf / lec*.pdf / slide*.pdf  → slide
        hw*.pdf / homework*.pdf               → homework
        note*.pdf / everything else           → note
    """
    configs = []
    for pdf_path in sorted(Path(input_dir).glob("**/*.pdf")):
        name_lower = pdf_path.stem.lower()
        if any(kw in name_lower for kw in ["lecture", "lec", "slide"]):
            source_type = "slide"
        elif any(kw in name_lower for kw in ["hw", "homework", "assignment"]):
            source_type = "homework"
        else:
            source_type = "note"

        configs.append({
            "path":        str(pdf_path),
            "title":       pdf_path.stem.replace("_", " ").replace("-", " ").title(),
            "source_type": source_type,
        })
        log.info(f"  Discovered [{source_type}] {pdf_path.name}")
    return configs


# ─────────────────────────────────────────────
# 8. CLI Entry Point
# ─────────────────────────────────────────────

DEFAULT_CONFIG = {
    "chunk_size_words":         300,   # words per chunk (notes/hw)
    "chunk_overlap_words":       50,   # overlap between consecutive chunks
    "slide_min_words_per_page":  10,   # pages below this are merged with next
}

def main():
    parser = argparse.ArgumentParser(description="Course PDF → Knowledge Base pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config",    type=str, help="Path to JSON config file")
    group.add_argument("--input_dir", type=str, help="Directory to auto-discover PDFs from")
    parser.add_argument("--output", type=str, default="data/processed/knowledge_base.jsonl",
                        help="Output JSONL path (default: data/processed/knowledge_base.jsonl)")
    args = parser.parse_args(
    args=None if len(sys.argv) > 1 else [
        "--config", "config.json",
        "--output", "data/processed/knowledge_base.jsonl"
    ]
)

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        doc_configs    = cfg.get("documents", [])
        global_config  = {**DEFAULT_CONFIG, **cfg.get("settings", {})}
    else:
        log.info(f"Auto-discovering PDFs in: {args.input_dir}")
        doc_configs   = discover_pdfs(args.input_dir)
        global_config = DEFAULT_CONFIG

    if not doc_configs:
        log.error("No documents found. Check your config or input_dir.")
        return

    chunks = build_knowledge_base(doc_configs, global_config)

    if chunks:
        print_stats(chunks)
        save_jsonl(chunks, args.output)
    else:
        log.error("No chunks were generated — check your PDF files.")


if __name__ == "__main__":
    main()
