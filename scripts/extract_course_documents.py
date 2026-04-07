#!/usr/bin/env python3
"""Extract structured course documents from lecture PDFs with Docling."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

try:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.doc import CoordOrigin, PictureItem, TableItem
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "Docling is required. Install it with: pip install docling pillow"
    ) from exc


@dataclass
class Block:
    block_id: str
    type: str
    text: Optional[str]
    bbox: List[float]
    reading_order: int
    caption: Optional[str] = None
    image_path: Optional[str] = None


@dataclass
class PageData:
    page_no: int
    width: float
    height: float
    blocks: List[Block] = field(default_factory=list)


@dataclass
class DocumentData:
    doc_id: str
    title: str
    source_file: str
    page_count: int
    pages: List[PageData] = field(default_factory=list)


def document_from_dict(payload: dict) -> DocumentData:
    return DocumentData(
        doc_id=payload["doc_id"],
        title=payload["title"],
        source_file=payload["source_file"],
        page_count=payload["page_count"],
        pages=[
            PageData(
                page_no=page["page_no"],
                width=page["width"],
                height=page["height"],
                blocks=[Block(**block) for block in page.get("blocks", [])],
            )
            for page in payload.get("pages", [])
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract structured document JSON and cropped figures from lecture PDFs."
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory for extracted JSON and figure crops.",
    )
    parser.add_argument(
        "--course-id",
        default="adl",
        help="Course identifier stored in the output JSON.",
    )
    parser.add_argument(
        "--course-name",
        default="Applied Deep Learning",
        help="Course name stored in the output JSON.",
    )
    parser.add_argument(
        "--render-scale",
        type=float,
        default=2.0,
        help="Scale factor used when rendering page and figure images.",
    )
    parser.add_argument(
        "--save-docling-json",
        action="store_true",
        default=True,
        help="Also save Docling's native JSON beside the normalized output.",
    )
    parser.add_argument(
        "--enrich-formula",
        action="store_true",
        default=True,
        help="Enable Docling formula enrichment to improve formula text extraction.",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()
    return normalized or "document"


def clean_text(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text or None


def label_to_string(label: object) -> str:
    return getattr(label, "value", str(label)).lower()


def map_docling_label(item: object) -> str:
    if isinstance(item, PictureItem):
        return "figure"
    if isinstance(item, TableItem):
        return "table"

    label = label_to_string(getattr(item, "label", "text"))
    if label in {"title", "document_index", "section_header"}:
        return "title"
    if label == "formula":
        return "formula"
    return "text"


def normalize_bbox_from_prov(prov: object, page_height: float) -> List[float]:
    bbox = getattr(prov, "bbox", None)
    if bbox is None:
        return [0.0, 0.0, 0.0, 0.0]

    bbox_origin = getattr(bbox, "coord_origin", CoordOrigin.TOPLEFT)
    if bbox_origin != CoordOrigin.TOPLEFT:
        bbox = bbox.to_top_left_origin(page_height)

    return [
        round(float(bbox.l), 2),
        round(float(bbox.t), 2),
        round(float(bbox.r), 2),
        round(float(bbox.b), 2),
    ]


def bbox_width_height(bbox: List[float]) -> tuple[float, float]:
    return max(0.0, bbox[2] - bbox[0]), max(0.0, bbox[3] - bbox[1])


def should_keep_figure(bbox: List[float]) -> bool:
    width, height = bbox_width_height(bbox)
    area = width * height

    # Filter out slide-template decorations such as footer lines and tiny icons.
    if width < 24 or height < 24:
        return False
    if area < 2000:
        return False

    return True


def item_bbox(item: object, page_height: float) -> List[float]:
    prov = getattr(item, "prov", None)
    if not prov:
        return [0.0, 0.0, 0.0, 0.0]
    return normalize_bbox_from_prov(prov[0], page_height=page_height)


def item_text(item: object, document: object) -> Optional[str]:
    if isinstance(item, PictureItem):
        if hasattr(item, "caption_text"):
            try:
                return clean_text(item.caption_text(document))
            except TypeError:
                return clean_text(item.caption_text())
        return None

    if isinstance(item, TableItem) and hasattr(item, "export_to_markdown"):
        return clean_text(item.export_to_markdown())

    text = getattr(item, "text", None)
    if text is not None:
        return clean_text(text)

    if hasattr(item, "export_to_markdown"):
        return clean_text(item.export_to_markdown())

    return None


def formula_text_candidates(item: object) -> List[str]:
    candidates: List[str] = []

    for attr_name in ("latex", "orig", "text"):
        value = clean_text(getattr(item, attr_name, None))
        if value:
            candidates.append(value)

    for method_name in ("export_to_markdown", "export_to_html", "export_to_text"):
        if not hasattr(item, method_name):
            continue
        method = getattr(item, method_name)
        try:
            value = clean_text(method())
        except TypeError:
            continue
        if value:
            candidates.append(value)

    return candidates


def extract_formula_text(item: object) -> Optional[str]:
    candidates = formula_text_candidates(item)
    return candidates[0] if candidates else None


def save_item_image(
    item: object,
    document: object,
    output_path: Path,
) -> Optional[str]:
    if not hasattr(item, "get_image"):
        return None

    image = item.get_image(document)
    if image is None:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, "PNG")
    return str(output_path.as_posix())


def build_converter(render_scale: float, enrich_formula: bool) -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = render_scale
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    pipeline_options.do_formula_enrichment = enrich_formula

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def iter_page_items(document: object, page_no: int) -> Iterable[object]:
    for item, _level in document.iterate_items(
        page_no=page_no,
        with_groups=False,
        traverse_pictures=True,
    ):
        yield item


def extract_document(
    pdf_path: Path,
    output_dir: Path,
    course_id: str,
    converter: DocumentConverter,
    save_docling_json: bool,
) -> DocumentData:
    doc_slug = slugify(pdf_path.stem)
    doc_id = f"{course_id}_{doc_slug}"
    doc_output_dir = output_dir / doc_id

    print(f"Start extracting {pdf_path.name}")
    print(f"{pdf_path.name}: starting docling conversion")
    conv_res = converter.convert(pdf_path)
    print(f"{pdf_path.name}: docling conversion finished")
    document = conv_res.document


    conv_res = converter.convert(pdf_path)
    document = conv_res.document

    if save_docling_json:
        doc_output_dir.mkdir(parents=True, exist_ok=True)
        document.save_as_json(doc_output_dir / f"{doc_id}.docling.json")

    pages: List[PageData] = []
    total_pages = len(document.pages)

    for page_no, page in document.pages.items():
        # print(f"{pdf_path.name}: processing page {page_no}/{total_pages}")
        page_width = round(float(page.size.width), 2)
        page_height = round(float(page.size.height), 2)
        blocks: List[Block] = []

        for reading_order, item in enumerate(iter_page_items(document, page_no), start=1):
            block_type = map_docling_label(item)
            text = item_text(item, document=document)
            bbox = item_bbox(item, page_height=page_height)

            if block_type == "figure" and not should_keep_figure(bbox):
                continue

            caption = text if block_type == "figure" else None
            image_path = None

            if block_type == "figure":
                image_name = f"{doc_id}_p{page_no}_b{reading_order}.png"
                image_path = save_item_image(
                    item,
                    document=document,
                    output_path=doc_output_dir / "figures" / image_name,
                )
            elif block_type == "formula":
                # Keep formula crops even when enrichment is enabled so we always
                # retain a visual fallback for downstream retrieval and citations.
                image_name = f"{doc_id}_p{page_no}_b{reading_order}.png"
                image_path = save_item_image(
                    item,
                    document=document,
                    output_path=doc_output_dir / "formulas" / image_name,
                )
                extracted_formula_text = extract_formula_text(item)
                if extracted_formula_text and not text:
                    text = extracted_formula_text

            blocks.append(
                Block(
                    block_id=f"{doc_id}_p{page_no}_b{reading_order}",
                    type=block_type,
                    text=(None if block_type == "figure" else text),
                    caption=caption,
                    image_path=image_path,
                    bbox=bbox,
                    reading_order=reading_order,
                )
            )

        pages.append(
            PageData(
                page_no=page_no,
                width=page_width,
                height=page_height,
                blocks=blocks,
            )
        )

    return DocumentData(
        doc_id=doc_id,
        title=pdf_path.stem,
        source_file=str(pdf_path.as_posix()),
        page_count=len(document.pages),
        pages=pages,
    )


def write_course_json(
    *,
    output_dir: Path,
    course_id: str,
    course_name: str,
    documents: List[DocumentData],
) -> Path:
    payload = {
        "course_id": course_id,
        "course_name": course_name,
        "source_type": "lecture_pdf",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "documents": [asdict(document) for document in documents],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "document.json"
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return output_path


def load_existing_documents(output_dir: Path) -> List[DocumentData]:
    output_path = output_dir / "document.json"
    if not output_path.exists():
        return []

    payload = json.loads(output_path.read_text())
    return [document_from_dict(document) for document in payload.get("documents", [])]


def resolve_input_pdfs() -> List[Path]:
    return sorted(Path("data/raw/adl").glob("*.pdf"))


def should_skip_pdf(pdf_path: Path) -> bool:
    return False


def should_enrich_formula(pdf_path: Path, default_enrich_formula: bool) -> bool:
    if pdf_path.name == "Lecture 3.pdf":
        return False
    return default_enrich_formula


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    pdf_paths = resolve_input_pdfs()
    if not pdf_paths:
        raise SystemExit("No PDF files found for extraction.")

    documents = load_existing_documents(output_dir)
    completed_doc_ids = {document.doc_id for document in documents}
    pending_pdf_paths = [
        pdf_path
        for pdf_path in pdf_paths
        if f"{args.course_id}_{slugify(pdf_path.stem)}" not in completed_doc_ids
        and not should_skip_pdf(pdf_path)
    ]

    if not pending_pdf_paths:
        print("All PDFs have already been extracted.")
        return

    for pdf_path in pending_pdf_paths:
        enrich_formula = should_enrich_formula(
            pdf_path=pdf_path,
            default_enrich_formula=args.enrich_formula,
        )
        converter = build_converter(
            render_scale=args.render_scale,
            enrich_formula=enrich_formula,
        )
        if not enrich_formula:
            print(f"{pdf_path.name}: formula enrichment disabled")
        document = extract_document(
            pdf_path=pdf_path,
            output_dir=output_dir,
            course_id=args.course_id,
            converter=converter,
            save_docling_json=args.save_docling_json,
        )
        documents.append(document)
        output_path = write_course_json(
            output_dir=output_dir,
            course_id=args.course_id,
            course_name=args.course_name,
            documents=documents,
        )
        print(f"{pdf_path.name} extract successfully")
    print(f"Wrote extracted course JSON to {output_path}")


if __name__ == "__main__":
    main()
