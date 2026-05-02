"""Post-process STAT 5703 extracted course JSON."""

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List
DEFAULT_COURSE_ID = '5703'
COURSE_TEMPLATE_NOISE_PATTERNS = [re.compile('^stat\\s*5703$', re.IGNORECASE), re.compile('^(?:do\\s*)?columbia\\s*university$', re.IGNORECASE), re.compile('^in\\s+the\\s+city\\s+of\\s+new\\s+york$', re.IGNORECASE), re.compile('^dobrin\\s+marchev$', re.IGNORECASE)]
SINGLE_CHAR_OCR_GARBAGE = {'ҧ', 'Ƹ', '…', '-', '–', '—', '÷', '·', '•'}
STATS_TERMS_PATTERN = re.compile('\\b(?:p[-\\s]?value|R[-\\s]?squared|R\\^2|AIC|BIC|MSE|RMSE|MLE|MOM|OLS|GLS|WLS|std\\.?\\s*err(?:or)?|SE|CI|Normal|Gaussian|Binomial|Bernoulli|Poisson|Beta|Gamma|Exponential|Chi[-\\s]?squared?|t[-\\s]?dist(?:ribution)?|F[-\\s]?dist(?:ribution)?|Pareto|Uniform|Cauchy|Weibull|linear\\s+regression|logistic|GLM|hierarchical|Cox|survival|Markov)\\b', re.IGNORECASE)
AXIS_TICK_PATTERN = re.compile('^\\s*[\\-+]?\\d+(?:[.,]\\d+)?\\s*[\\-–—]?\\s*$')

def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())

def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

def clean_formula_text(text: str) -> str:
    text = text.replace('▶', ' ')
    text = text.replace('\xa0', ' ')
    text = re.sub('\\s+', ' ', text).strip()
    text = re.sub('(?<![A-Za-z\\\\])(?:[A-Za-z]\\s+){2,}[A-Za-z](?![A-Za-z])', lambda match: ''.join(match.group(0).split()), text)
    text = re.sub('\\s+([,.;:)\\]\\}])', '\\1', text)
    text = re.sub('([(\\[\\{])\\s+', '\\1', text)
    text = re.sub('\\s*_\\s*', '_', text)
    text = re.sub('\\s*\\^\\s*', '^', text)
    text = re.sub('\\s*\\\\\\s+([A-Za-z]+)', ' \\\\\\1', text)
    text = re.sub('\\{\\s+', '{', text)
    text = re.sub('\\s+\\}', '}', text)
    text = re.sub('\\(\\s+', '(', text)
    text = re.sub('\\s+\\)', ')', text)
    text = re.sub('\\[\\s+', '[', text)
    text = re.sub('\\s+\\]', ']', text)
    text = re.sub('\\s{2,}', ' ', text).strip()
    return text

def clean_general_text(text: str) -> str:
    text = text.replace('▶', ' ')
    text = text.replace('\xa0', ' ')
    text = re.sub('\\s+', ' ', text).strip()
    return text

def classify_formula_quality(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return 'truncated'
    brace_diff = abs(stripped.count('{') - stripped.count('}'))
    paren_diff = abs(stripped.count('(') - stripped.count(')'))
    if brace_diff >= 2 or paren_diff >= 2:
        return 'truncated'
    if stripped.endswith(('=', '+', '-', '\\', '{', '(')):
        return 'truncated'
    if stripped.startswith(('0)', '1)', '2)', '3)', '\\\\', '&')):
        return 'truncated'
    if '\\text {' in stripped and '\\\\' in stripped and (len(stripped) > 120):
        return 'truncated'
    if re.search('(?:[A-Za-z]\\s+){3,}[A-Za-z](?![A-Za-z])', stripped):
        return 'noisy'
    if '\\ ' in stripped or re.search('\\\\\\\\\\s+[A-Za-z]', stripped):
        return 'noisy'
    if '\\mathring' in stripped or '\\intercal' in stripped:
        return 'noisy'
    if brace_diff == 1 or paren_diff == 1:
        return 'noisy'
    return 'good'

def bbox_inside(inner: List[float], outer: List[float], tolerance: float=2.0) -> bool:
    return inner[0] >= outer[0] - tolerance and inner[1] >= outer[1] - tolerance and (inner[2] <= outer[2] + tolerance) and (inner[3] <= outer[3] + tolerance)

def bbox_intersects(a: List[float], b: List[float], tolerance: float=0.0) -> bool:
    return not (a[2] < b[0] - tolerance or a[0] > b[2] + tolerance or a[3] < b[1] - tolerance or (a[1] > b[3] + tolerance))

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

def point_inside_bbox(point: tuple[float, float], bbox: List[float], tolerance: float=0.0) -> bool:
    x, y = point
    return bbox[0] - tolerance <= x <= bbox[2] + tolerance and bbox[1] - tolerance <= y <= bbox[3] + tolerance

def bbox_vertical_gap(a: List[float], b: List[float]) -> float:
    if a[3] < b[1]:
        return b[1] - a[3]
    if b[3] < a[1]:
        return a[1] - b[3]
    return 0.0

def bbox_width(bbox: List[float]) -> float:
    return max(0.0, bbox[2] - bbox[0])

def bbox_height(bbox: List[float]) -> float:
    return max(0.0, bbox[3] - bbox[1])

def bbox_horizontal_overlap_ratio(a: List[float], b: List[float]) -> float:
    overlap = min(a[2], b[2]) - max(a[0], b[0])
    if overlap <= 0:
        return 0.0
    base = min(bbox_width(a), bbox_width(b))
    if base <= 0:
        return 0.0
    return overlap / base

def is_course_template_noise(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return any((p.match(stripped) for p in COURSE_TEMPLATE_NOISE_PATTERNS))

def is_single_char_ocr_garbage(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if len(stripped) <= 2 and stripped in SINGLE_CHAR_OCR_GARBAGE:
        return True
    if len(stripped) == 1 and (not stripped.isalnum()):
        return True
    return False

def is_axis_tick_label(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if AXIS_TICK_PATTERN.match(stripped):
        return True
    return False

def is_short_axis_or_legend_fragment(text: str, bbox: List[float], page_width: float, page_height: float) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    width = bbox_width(bbox)
    height = bbox_height(bbox)
    is_tiny_bbox = width <= 40 and height <= 14
    if is_axis_tick_label(stripped):
        return True
    if is_single_char_ocr_garbage(stripped):
        return True
    if is_tiny_bbox and len(stripped) <= 2 and (not re.search('[A-Za-z0-9]', stripped)):
        return True
    return False

def is_obvious_noise_text_block(block: Dict[str, Any]) -> bool:
    if block.get('type') != 'text':
        return False
    text = (block.get('text') or '').strip()
    if not text:
        return False
    if is_course_template_noise(text):
        return True
    if is_single_char_ocr_garbage(text):
        return True
    if is_axis_tick_label(text):
        return True
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    numeric_spans = re.findall('\\d+(?:\\.\\d+)?', text)
    pure_numeric = bool(re.fullmatch('\\d+(?:\\.\\d+)?(?:[MK%]|M)?', text))
    high_numeric_density = len(numeric_spans) >= 4 or (char_count > 0 and sum((ch.isdigit() for ch in text)) / char_count >= 0.45)
    looks_like_sentence = word_count >= 5 and bool(re.search('[a-z]{3,}', text))
    if word_count <= 1 and char_count <= 4:
        return True
    if pure_numeric:
        return True
    if high_numeric_density and (not looks_like_sentence):
        return True
    return False

def is_caption_or_table_note(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if re.match('^(figure|fig|table)\\b', stripped, flags=re.IGNORECASE):
        return True
    if re.search('\\b(?:figure|fig|table)\\s+[A-Za-z]?\\d', stripped, flags=re.IGNORECASE):
        return True
    return False

def is_table_caption_or_summary(block: Dict[str, Any]) -> bool:
    if block.get('type') != 'text':
        return False
    text = (block.get('text') or '').strip()
    if not text:
        return False
    if re.match('^table\\b', text, flags=re.IGNORECASE):
        return True
    return len(text) >= 70 or len(text.split()) >= 12

def is_table_content_fragment_text(block: Dict[str, Any]) -> bool:
    if block.get('type') != 'text':
        return False
    text = (block.get('text') or '').strip()
    if not text:
        return False
    words = text.split()
    if len(words) > 4 or len(text) > 64:
        return False
    normalized = re.sub('\\s+', ' ', text)
    numeric_like = bool(re.fullmatch('-?\\d+(?:\\.\\d+)?(?:[%MK])?', normalized) or re.fullmatch('-?\\d+(?:,\\d{3})+(?:\\.\\d+)?[MK%]?', normalized))
    stats_term_like = bool(STATS_TERMS_PATTERN.fullmatch(normalized) or (STATS_TERMS_PATTERN.search(normalized) and len(words) <= 3))
    header_like = bool('|' in normalized or normalized.lower() in {'n', 'mean', 'median', 'sd', 'std', 'var', 'min', 'max', 'count', 'gender', 'decision', 'promoted', 'not promoted', 'freq', 'frequency', 'estimate', 'value'})
    return numeric_like or stats_term_like or header_like

def is_numeric_heavy_sentence(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    words = stripped.split()
    if len(words) < 10:
        return False
    numeric_tokens = re.findall('\\d+(?:\\.\\d+)?', stripped)
    return len(numeric_tokens) >= 6 or (len(numeric_tokens) >= 4 and len(words) <= 28)

def is_meaningful_course_content(block: Dict[str, Any]) -> bool:
    if block.get('type') not in {'text', 'title', 'formula', 'table'}:
        return False
    text = (block.get('text') or '').strip()
    if not text:
        return False
    normalized = re.sub('\\s+', ' ', text)
    if is_course_template_noise(normalized):
        return False
    if is_axis_tick_label(normalized):
        return False
    if is_single_char_ocr_garbage(normalized):
        return False
    if len(normalized) >= 25 and re.search('[A-Za-z]{3,}', normalized):
        return True
    words = normalized.split()
    if len(words) >= 3 and re.search('[A-Za-z]{3,}', normalized):
        return True
    important_patterns = ['\\bp[-\\s]?value\\b', '\\bR[-\\s]?squared\\b', '\\bEstimate\\b', '\\bStd\\.?\\s*Error\\b', '\\bt value\\b', '\\bPr\\(>\\|t\\|\\)', '\\bt\\.test\\b', '\\blm\\(', '\\bpnorm\\b', '\\bqnorm\\b', '\\bpchisq\\b', '\\bmean squared\\b', '\\bprediction error\\b', '\\bhypothesis\\b', '\\bconfidence interval\\b', '\\bMLE\\b', '\\bestimator\\b', '\\bregression\\b', '\\btransition probabilities\\b', '\\btest set error\\b', '\\btraining data\\b', '\\bprediction errors?\\b', '\\bpolynomial\\b', '\\bpromotion\\b', '\\bchi[-\\s]?square\\b', '\\bindependent\\b', '\\bdependent\\b', '\\bassumption\\b', '\\bsample mean\\b', '\\bvariance\\b', '\\bbias\\b']
    if any((re.search(pattern, normalized, re.IGNORECASE) for pattern in important_patterns)):
        return True
    return False

def is_near_figure_fragment_text(block: Dict[str, Any]) -> bool:
    if block.get('type') != 'text':
        return False
    text = (block.get('text') or '').strip()
    if not text:
        return False
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    normalized = re.sub('\\s+', ' ', text)
    looks_like_sentence = word_count >= 6 or normalized.endswith(('.', '?', '!')) or ';' in normalized or (':' in normalized and word_count >= 5)
    short_label = word_count <= 4 and char_count <= 42 and (not normalized.endswith(('.', '?', '!')))
    plot_legend_like = bool(re.search('\\b(?:density|frequency|count|cdf|pdf|pmf|empirical|theoretical|observed|expected|fitted|residual[s]?|quantile|high\\s+bias|low\\s+bias|high\\s+variance|low\\s+variance|true\\s+value|null|alternative|reject|do\\s+not\\s+reject|convergence(?:\\s+in)?|almost\\s+sure|in\\s+probability|in\\s+distribution|self|cross|sample|population|darwin\\.maize)\\b', normalized, flags=re.IGNORECASE)) and word_count <= 6
    axis_label_like = bool(re.search('\\b(?:x[-\\s]?axis|y[-\\s]?axis|time|index|step|iteration|epoch|theta|lambda|mu|sigma|alpha|beta)\\b', normalized, flags=re.IGNORECASE)) and word_count <= 4
    return short_label and (plot_legend_like or axis_label_like)

def is_embedded_figure_text_block(block: Dict[str, Any], figure_bbox: List[float]) -> bool:
    if block.get('type') != 'text':
        return False
    bbox = block.get('bbox') or [0.0, 0.0, 0.0, 0.0]
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
    return overlap_ratio >= 0.5 or center_inside or figure_coverage_ratio >= 0.12

def is_near_region(bbox: List[float], region_bbox: List[float], *, expanded_bbox: List[float], vertical_gap_max: float, horizontal_overlap_min: float) -> bool:
    vertical_gap = bbox_vertical_gap(bbox, region_bbox)
    horizontal_overlap = bbox_horizontal_overlap_ratio(bbox, region_bbox)
    return bbox_intersects(bbox, expanded_bbox) or (vertical_gap <= vertical_gap_max and horizontal_overlap >= horizontal_overlap_min)

def should_remove_figure_fragment(block: Dict[str, Any], figure_blocks: List[Dict[str, Any]]) -> bool:
    if block.get('type') not in {'text', 'title', 'formula'}:
        return False
    bbox = block.get('bbox') or [0.0, 0.0, 0.0, 0.0]
    text = (block.get('text') or '').strip()
    if is_meaningful_course_content(block):
        return False
    caption_or_note = is_caption_or_table_note(text)
    obvious_noise = is_obvious_noise_text_block(block)
    near_figure_fragment = is_near_figure_fragment_text(block)
    for figure in figure_blocks:
        figure_bbox = figure.get('bbox') or [0.0, 0.0, 0.0, 0.0]
        overlap_ratio = bbox_overlap_ratio(bbox, figure_bbox)
        near_figure = is_near_region(bbox, figure_bbox, expanded_bbox=[figure_bbox[0] - 24.0, figure_bbox[1] - 24.0, figure_bbox[2] + 24.0, figure_bbox[3] + 24.0], vertical_gap_max=20.0, horizontal_overlap_min=0.35)
        if bbox_inside(bbox, figure_bbox):
            return True
        if is_embedded_figure_text_block(block, figure_bbox):
            return True
        if overlap_ratio >= 0.5:
            return True
        if block.get('type') == 'text' and near_figure and (caption_or_note or obvious_noise or near_figure_fragment):
            return True
    return False

def should_remove_table_fragment(block: Dict[str, Any], table_blocks: List[Dict[str, Any]]) -> bool:
    if block.get('type') != 'text':
        return False
    bbox = block.get('bbox') or [0.0, 0.0, 0.0, 0.0]
    text = (block.get('text') or '').strip()
    if not text:
        return False
    if is_meaningful_course_content(block):
        return False
    is_caption_or_summary = is_table_caption_or_summary(block)
    is_content_fragment = is_table_content_fragment_text(block)
    obvious_noise = is_obvious_noise_text_block(block)
    numeric_heavy_sentence = is_numeric_heavy_sentence(text)
    for table in table_blocks:
        table_bbox = table.get('bbox') or [0.0, 0.0, 0.0, 0.0]
        overlap_area = bbox_intersection_area(bbox, table_bbox)
        overlap_ratio = bbox_overlap_ratio(bbox, table_bbox)
        expanded_table_bbox = [table_bbox[0] - 40.0, table_bbox[1] - 120.0, table_bbox[2] + 40.0, table_bbox[3] + 56.0]
        near_table = is_near_region(bbox, table_bbox, expanded_bbox=expanded_table_bbox, vertical_gap_max=28.0, horizontal_overlap_min=0.3)
        if bbox_inside(bbox, table_bbox):
            return True
        if overlap_ratio >= 0.35:
            return True
        if is_caption_or_summary and (not numeric_heavy_sentence):
            continue
        if near_table and numeric_heavy_sentence:
            return True
        if (is_content_fragment or obvious_noise) and (overlap_area > 0.0 or near_table):
            return True
    return False

def post_process_page(page: Dict[str, Any]) -> None:
    blocks = page.get('blocks', [])
    page_w = float(page.get('width') or 0.0)
    page_h = float(page.get('height') or 0.0)
    for block in blocks:
        if isinstance(block.get('text'), str):
            if block.get('type') == 'formula':
                block['text'] = clean_formula_text(block['text'])
            else:
                block['text'] = clean_general_text(block['text'])
    title_blocks = [b for b in blocks if b.get('type') == 'title']
    first_title_order = None
    if title_blocks:
        first_title_order = min((b.get('reading_order', 10 ** 9) for b in title_blocks))
    figure_blocks = [b for b in blocks if b.get('type') == 'figure']
    table_blocks = [b for b in blocks if b.get('type') == 'table']
    has_table = bool(table_blocks)
    has_figure = bool(figure_blocks)
    processed_blocks: List[Dict[str, Any]] = []
    for block in blocks:
        block_type = block.get('type')
        text = (block.get('text') or '').strip() if isinstance(block.get('text'), str) else ''
        bbox = block.get('bbox') or [0.0, 0.0, 0.0, 0.0]
        if block_type in {'text', 'title'} and is_course_template_noise(text):
            continue
        if block_type == 'text' and is_single_char_ocr_garbage(text):
            continue
        if has_figure and block_type == 'text' and is_short_axis_or_legend_fragment(text, bbox, page_w, page_h):
            continue
        if block_type == 'text' and first_title_order is not None and (1 < first_title_order <= 3) and (block.get('reading_order', 10 ** 9) < first_title_order) and (not is_meaningful_course_content(block)):
            continue
        if block_type == 'table' and isinstance(block.get('text'), str):
            block['text'] = clean_general_text(block['text'])
        if block_type == 'formula' and block.get('text'):
            block['formula_quality'] = classify_formula_quality(block['text'])
        elif block_type == 'formula':
            block['formula_quality'] = 'truncated'
        meaningful_content = is_meaningful_course_content(block)
        if (has_figure or has_table) and (not meaningful_content) and is_obvious_noise_text_block(block):
            continue
        if has_table and block_type == 'text' and (not meaningful_content):
            if text:
                is_caption_or_summary = is_table_caption_or_summary(block)
                if (is_table_content_fragment_text(block) or is_obvious_noise_text_block(block)) and (not is_caption_or_summary):
                    continue
                if is_numeric_heavy_sentence(text) and (not text.lower().startswith('table')):
                    continue
        if figure_blocks and should_remove_figure_fragment(block, figure_blocks=figure_blocks):
            continue
        if table_blocks and should_remove_table_fragment(block, table_blocks=table_blocks):
            continue
        processed_blocks.append(block)
    for reading_order, block in enumerate(processed_blocks, start=1):
        block['reading_order'] = reading_order
    page['blocks'] = processed_blocks

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--course-id', default=DEFAULT_COURSE_ID, help='Course identifier (default: 5703).')
    parser.add_argument('--input', dest='input_path', help='Input JSON path. Defaults to data/processed/<course_id>/<course_id>_document.json')
    parser.add_argument('--output', dest='output_path', help='Output JSON path. Defaults to data/processed/<course_id>/<course_id>_processed.json')
    return parser.parse_args()

def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    base = Path('data/processed') / args.course_id
    input_path = Path(args.input_path) if args.input_path else base / f'{args.course_id}_document.json'
    output_path = Path(args.output_path) if args.output_path else base / f'{args.course_id}_processed.json'
    return (input_path, output_path)

def summarize(payload: Dict[str, Any]) -> Dict[str, int]:
    counts = {'text': 0, 'title': 0, 'figure': 0, 'table': 0, 'formula': 0}
    for doc in payload.get('documents', []):
        for page in doc.get('pages', []):
            for block in page.get('blocks', []):
                t = block.get('type')
                if t in counts:
                    counts[t] += 1
    return counts

def main() -> None:
    args = parse_args()
    input_path, output_path = resolve_paths(args)
    payload = load_json(input_path)
    before = summarize(payload)
    for document in payload.get('documents', []):
        for page in document.get('pages', []):
            post_process_page(page)
    after = summarize(payload)
    save_json(output_path, payload)
    print(f'Wrote post-processed document JSON to {output_path}')
    print(f'  block counts before -> after:')
    for t in ('text', 'title', 'figure', 'table', 'formula'):
        print(f'    {t:7s}  {before[t]:5d}  ->  {after[t]:5d}  (dropped {before[t] - after[t]})')
if __name__ == '__main__':
    main()
