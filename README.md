# AI Course Assistant

AI Course Assistant is a course-specific Retrieval-Augmented Generation (RAG)
system for answering student questions from structured course materials. Instead
of relying on a general-purpose chatbot's internal memory, the system converts
lecture slides and other course PDFs into a searchable knowledge base, retrieves
relevant course chunks, and generates answers grounded in those chunks with
source citations.

The project was developed for a GenAI final project by Serena Cheng, Fei Xue,
and Yining Tao.

## Project Objective

The goal is to build a transferable course knowledge assistant that can:

- ingest heterogeneous course materials such as slides, notes, quizzes, and
  homework PDFs;
- clean and organize them into a reusable course knowledge base;
- retrieve relevant evidence for student questions;
- generate answers that are grounded in the retrieved materials;
- cite the supporting lecture document and page numbers;
- evaluate retrieval quality using standard information retrieval metrics; and
- adapt the same pipeline to multiple courses with limited manual changes.

The system currently supports multiple indexed courses, including Elements of
Data Science (EODS), Applied Deep Learning (ADL), and Statistical Inference.

## Key Ideas

This project is not an open-domain chatbot. It is a course-bound RAG system.
That design choice is important because educational answers need to be
verifiable, aligned with course materials, and traceable to source documents.

The main technical ideas are:

- **Document conversion**: PDF slides are parsed into structured page/block JSON
  using Docling.
- **Course-specific cleaning**: Each course has a custom post-processing script to
  handle notebook screenshots, OCR noise, formulas, and code-like slide content.
- **Structured chunking**: The system builds both atomic chunks and semantic
  chunks.
- **Separated retrieval and generation fields**: Chunks store both
  `content_for_embedding` and `content_for_generation`.
- **Hybrid retrieval**: FAISS dense retrieval can be combined with BM25 keyword
  retrieval and reranking.
- **Grounded generation**: The answer generator receives only retrieved course
  context and is instructed to cite source labels.
- **Evaluation**: Retrieval is evaluated with gold question-evidence sets using
  recall, precision, MRR, and nDCG.
- **Transferability**: The pipeline is modular, so a new course can be added by
  placing PDFs in `data/raw/<course_id>/` and running the ingestion/indexing
  pipeline.

## System Pipeline

The end-to-end pipeline is:

```text
PDF course materials
→ scripts/extraction/extract_course_documents.py
→ data/processed/<course>/<course>_document.json
→ course-specific or generic post-processing
→ data/processed/<course>/<course>_processed.json
→ scripts/run_before_chunk.py
→ data/processed/<course>/<course>_merged.json
→ scripts/chunk/atomic_chunk.py
→ data/chunk/<course>_atomic_chunks.json
→ scripts/chunk/atomic_embedding.py
→ data/chunk/<course>_atomic_embeddings.json
→ scripts/chunk/semantic_chunk.py
→ data/chunk/<course>_semantic_chunks.json
→ scripts/chunk/semantic_embedding.py
→ data/chunk/<course>_semantic_embeddings.json
→ scripts/retrieval/build_faiss_index.py
→ data/retrieval/<course>_atomic.faiss
→ data/retrieval/<course>_semantic.faiss
→ scripts/retrieval/retrieve_faiss_bm25.py
→ scripts/retrieval/generate_answer.py
→ scripts/demo_server.py + demo_ui/
```

## Repository Structure

```text
data/
  raw/                  Raw course PDFs
  processed/            Extracted and cleaned document JSON
  chunk/                Atomic/semantic chunks and local embedding JSON files
  retrieval/            FAISS indexes and metadata
  test/                 Retrieval evaluation datasets

scripts/
  extraction/           PDF extraction and document post-processing
  chunk/                Atomic chunking, semantic chunking, embeddings
  retrieval/            FAISS index building, retrieval, generation, evaluation
  demo_server.py        Local backend server for the UI
  run_before_chunk.py   Prepares cleaned blocks before chunking

demo_ui/
  index.html            Demo frontend
  app.js                UI logic
  styles.css            UI styles
```

## Core Components

### 1. PDF Extraction

Main script:

```text
scripts/extraction/extract_course_documents.py
```

This script reads PDFs from:

```text
data/raw/<course_id>/*.pdf
```

and writes:

```text
data/processed/<course_id>/<course_id>_document.json
```

It uses Docling to parse slides into a normalized schema:

- `DocumentData`
- `PageData`
- `Block`

Each block stores text, type, page location, bounding box, reading order, and
optional image paths for figures or formulas. The extraction step maps Docling
items into simplified internal block types such as `title`, `text`, `figure`,
`table`, and `formula`.

Bounding boxes are important because later cleaning logic can identify whether
OCR text came from inside a figure or notebook screenshot.

### 2. Example of Post-Processing for EODS

Main script:

```text
scripts/extraction/post_process_document_eods.py
```

Input:

```text
data/processed/eods/eods_document.json
```

Output:

```text
data/processed/eods/eods_processed.json
```

This script is a custom cleaning layer for Elements of Data Science. EODS slides
contain many Jupyter notebook screenshots. Docling may extract those screenshots
both as figure crops and as OCR text. Some of that OCR text is useful code, while
some of it is noise such as axis labels, chart ticks, or fragmented table text.

The EODS post-processing script:

- detects whether text blocks are inside figure bounding boxes;
- identifies useful Python, shell, and notebook-style code;
- converts detected code blocks into `type = "code"`;
- splits code-like text into `code_segments`;
- removes figure OCR noise such as chart labels and short numeric fragments;
- cleans formula text;
- classifies formula quality as `good`, `noisy`, or `truncated`; and
- rewrites reading order after filtering blocks.

This step is important because it preserves useful notebook examples while
removing low-value OCR artifacts before chunking.

### 3. Pre-Chunk Processing

Main scripts:

```text
scripts/run_before_chunk.py
scripts/extraction/text_before_chunk.py
scripts/extraction/formula_before_chunk.py
scripts/extraction/figure_before_chunk.py
```

These scripts prepare cleaned blocks for retrieval-aware chunking. They add
fields such as:

- `section_title`
- `nearby_text_before`
- `nearby_text_after`
- `formula_focus`
- `formula_explanation`
- `visual_description`
- `math_spans`

The output is:

```text
data/processed/<course_id>/<course_id>_merged.json
```

### 4. Atomic Chunks

Main script:

```text
scripts/chunk/atomic_chunk.py
```

Atomic chunks are small, precise retrieval units. The system builds different
chunk types:

- `text`
- `formula`
- `figure`
- `text_inline_math`

Each chunk stores:

- `content_for_embedding`: concise text optimized for retrieval;
- `content_for_generation`: richer structured content for the LLM;
- `metadata`: document id, page number, block id, bounding box, and other source
  information; and
- `raw_fields`: additional source fields for debugging or future processing.

The separation between embedding and generation content is central to the
system. Retrieval benefits from concise, normalized text, while generation
benefits from richer structured context.

### 5. Semantic Chunks

Main script:

```text
scripts/chunk/semantic_chunk.py
```

Semantic chunks are built from atomic chunks. Instead of using a fixed window
size, the script uses adjacent embedding similarity and section information to
decide when to merge neighboring atomic chunks.

The semantic chunking logic considers:

- document id;
- page and block order;
- cosine similarity between chunk embeddings;
- section title continuity;
- token limits; and
- auxiliary non-indexable figure chunks.

Atomic chunks are precise, while semantic chunks provide broader context for
generation.

### 6. Embeddings and FAISS Indexes

Main scripts:

```text
scripts/chunk/atomic_embedding.py
scripts/chunk/semantic_embedding.py
scripts/retrieval/build_faiss_index.py
```

The embedding scripts use OpenAI embeddings to encode atomic and semantic
chunks. The FAISS builder reads embedding JSON files, normalizes vectors, and
stores local vector indexes:

```python
faiss.normalize_L2(matrix)
index = faiss.IndexFlatIP(matrix.shape[1])
index.add(matrix)
```

Because vectors are L2-normalized, inner product search behaves like cosine
similarity.

Embedding JSON files can be large and are ignored by Git:

```text
data/chunk/*_embeddings.json
```

### 7. Retrieval

Main script:

```text
scripts/retrieval/retrieve_faiss_bm25.py
```

The retrieval module supports four methods:

- `bm25`: keyword-based retrieval;
- `dense`: FAISS vector search;
- `hybrid`: FAISS + BM25 with Reciprocal Rank Fusion (RRF);
- `dense_rerank`: dense candidate retrieval followed by reranking with dense
  score, BM25 score, and heuristic bonuses/penalties.

Dense retrieval embeds the user query, normalizes it, and searches against the
stored FAISS index.

BM25 retrieval tokenizes the query and chunk text, then scores exact keyword
matches.

Hybrid retrieval combines rankings using RRF:

```text
combined_score += weight / (rrf_k + rank)
```

`dense_rerank` is the default retrieval method. It retrieves a larger dense
candidate pool, computes BM25 scores over those candidates, normalizes both
signals, and reranks with additional logic for:

- semantic/atomic diversity;
- formula preference;
- figure preference;
- low-information text penalties;
- query mismatch penalties; and
- same-page repetition penalties.

### 8. Grounded Answer Generation

Main script:

```text
scripts/retrieval/generate_answer.py
```

Generation happens after retrieval. The script:

1. receives the student query;
2. calls the retrieval module;
3. selects context chunks;
4. renders `content_for_generation` into a source-labeled context block;
5. builds a prompt containing the question, context, and optional conversation
   memory; and
6. calls the OpenAI Responses API to generate an answer.

The prompt instructs the model to:

- answer only from retrieved course context;
- say what is missing if the context is insufficient;
- cite source labels inline, such as `[S1]`;
- end with a short Sources section;
- avoid raw LaTeX notation in final answers; and
- use readable Unicode-style math when formulas are needed.

The system preserves formula fields such as `formula_latex`,
`formula_explanation`, and `math_spans` during retrieval, but final formula
formatting is controlled by the generation prompt rather than by a deterministic
LaTeX-to-Unicode converter.

### 9. Conversation Memory

The demo supports short conversational memory with:

- `recent_turns`
- `conversation_summary`
- `current_topic`

Recent turns help with follow-up questions such as "Can you give an example?"
or "Why?" Older context can be compressed into a rolling summary. The current
topic helps resolve vague follow-up references.

Memory is used only to interpret follow-up questions. The answer is still
grounded in retrieved course chunks.

### 10. Demo UI

Frontend:

```text
demo_ui/index.html
demo_ui/app.js
demo_ui/styles.css
```

Backend:

```text
scripts/demo_server.py
```

The UI supports:

- selecting an indexed course;
- asking questions;
- adjusting retrieval/generation settings;
- viewing generated answers;
- viewing cited sources;
- creating a new course;
- adding PDFs to an existing course; and
- running ingestion and indexing from the interface.

## Setup

Create the environment:

```bash
conda env create -f environment.yml
conda activate ai-course-assistant
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

If you are using the base conda environment instead of the named environment,
make sure the required packages are installed:

```bash
pip install openai faiss-cpu docling pillow paddleocr paddlepaddle
```

## Running the Demo UI

From the repository root:

```bash
python scripts/demo_server.py
```

If your shell does not resolve the correct Python, use the conda Python directly:

```bash
/opt/miniconda3/bin/python scripts/demo_server.py
```

Open:

```text
http://127.0.0.1:8000
```

If port 8000 is already in use:

```bash
python scripts/demo_server.py --port 8001
```

## Rebuilding a Course Index Manually

Place PDFs under:

```text
data/raw/<course_id>/
```

For EODS:

```bash
python scripts/extraction/extract_course_documents.py \
  --course-id eods \
  --course-name "Elements of Data Science"

python scripts/extraction/post_process_document_eods.py

python scripts/run_before_chunk.py --course-id eods

python scripts/chunk/atomic_chunk.py --course-id eods

python scripts/chunk/atomic_embedding.py --course-id eods

python scripts/chunk/semantic_chunk.py --course-id eods

python scripts/chunk/semantic_embedding.py --course-id eods

python scripts/retrieval/build_faiss_index.py --course-id eods --target both
```

For a course without a custom post-processing script, use the generic
post-processor:

```bash
python scripts/extraction/post_process_document_generic.py --course-id <course_id>
```

## Running Retrieval

Example:

```bash
python scripts/retrieval/retrieve_faiss_bm25.py \
  --course-id eods \
  --query "What does a p-value mean in hypothesis testing?" \
  --target both \
  --method dense_rerank \
  --top-k 4 \
  --candidate-k 30
```

## Running Generation

Example:

```bash
python scripts/retrieval/generate_answer.py \
  --course-id eods \
  --query "How does PCA reduce dimensionality while preserving variance?"
```

The answer should include source citations such as `[S1]` and a Sources section
with document/page references.

## Retrieval Evaluation

Gold retrieval evaluation files are stored in:

```text
data/test/
```

Examples:

```text
data/test/eods_retrieval_eval_20.json
data/test/eods_retrieval_eval_40.json
data/test/adl_retrieval_eval_20.json
data/test/adl_retrieval_eval_40.json
```

Run EODS evaluation:

```bash
python scripts/retrieval/test_retrieval.py \
  --course-id eods \
  --eval-json data/test/eods_retrieval_eval_40.json \
  --target both \
  --method dense_rerank \
  --candidate-k 4
```

The evaluator reports:

- `recall@k`: how many gold chunks are retrieved in the top k;
- `precision@k`: how many retrieved chunks are relevant;
- `MRR`: how early the first relevant chunk appears;
- `nDCG`: how well relevant chunks are ranked near the top.

## Results

The table below shows an EODS retrieval evaluation run on 40 manually selected
question-evidence examples using both atomic and semantic chunks with
`dense_rerank` retrieval:

```text
Evaluated 40 queries | target=both | method=dense_rerank
```

| k | Recall | Precision | MRR | nDCG |
|---|---:|---:|---:|---:|
| 2 | 0.4771 | 0.4875 | 0.6875 | 0.5699 |
| 3 | 0.5604 | 0.3833 | 0.7042 | 0.5505 |
| 4 | 0.6104 | 0.3187 | 0.7104 | 0.5769 |

These results show the expected retrieval tradeoff: increasing `k` improves
recall because more gold evidence chunks are retrieved, while precision becomes
lower because the result set includes more non-gold chunks. MRR and nDCG measure
whether relevant chunks appear near the top of the ranked list.

There is also an older EODS sanity test:

```bash
python scripts/chunk/test_retrieval_eods.py --targets both --top-k 5
```

This checks representative EODS queries against expected keywords and chunk
types.

## Example Questions

EODS examples:

```text
What does a p-value mean in hypothesis testing?
How does PCA reduce dimensionality while preserving variance?
How do train-test split and cross-validation help evaluate machine learning models?
What are the exam traps and key checks to remember when joining datasets?
```

Quiz-specific example:

```text
In the Week 8 Quiz housing data, why do we create a SqFtLot_missing column before filling SqFtLot with its mean?
```

## Evaluation Scope

The system is designed to evaluate:

- retrieval quality;
- answer grounding;
- citation accuracy;
- source traceability; and
- transferability across courses.

The current implementation focuses on RAG and prompt-based grounded generation.
Large-scale model pretraining, production deployment, video/audio processing,
and complex personalization are out of scope.

## Reproducibility Checklist

This repository includes the following items to support reproducible setup and
experiments:

- `environment.yml` for creating a consistent Python environment.
- Raw course material folders under `data/raw/`.
- Processed course JSON, chunk JSON, FAISS indexes, and retrieval metadata under
  `data/processed/`, `data/chunk/`, and `data/retrieval/`.
- Retrieval evaluation datasets under `data/test/`.
- End-to-end rebuild commands in this README for extraction, post-processing,
  chunking, embedding, and FAISS indexing.
- Evaluation commands using `scripts/retrieval/test_retrieval.py`.
- A local demo server and frontend for interactive testing.

To reproduce the EODS retrieval experiment, use:

```bash
python scripts/retrieval/test_retrieval.py \
  --course-id eods \
  --eval-json data/test/eods_retrieval_eval_40.json \
  --target both \
  --method dense_rerank \
  --candidate-k 4
```

Because dense retrieval and generation use OpenAI APIs, reproducibility requires
setting `OPENAI_API_KEY` and using the same embedding and generation model names
listed in the command-line defaults.

## Troubleshooting

- If the UI shows `Error`, first check the terminal running
  `scripts/demo_server.py`; the backend returns the detailed exception there.
- If the error says `OPENAI_API_KEY is not set`, export the key before starting
  the server:

  ```bash
  export OPENAI_API_KEY="your_api_key_here"
  python scripts/demo_server.py
  ```

- If the model is unavailable, change the generation model in the UI advanced
  settings or pass a model name your API key can access.
- If no sources are returned, confirm that the course has FAISS index and
  metadata files in `data/retrieval/`, for example:

  ```text
  data/retrieval/eods_atomic.faiss
  data/retrieval/eods_atomic_metadata.json
  data/retrieval/eods_semantic.faiss
  data/retrieval/eods_semantic_metadata.json
  ```

- If uploaded PDFs appear stuck on `extract`, check the server terminal logs.
  Large PDFs or duplicate uploads may take longer because the ingestion pipeline
  scans all unprocessed PDFs for that course.
- If `python` is not found, activate the conda environment or call the conda
  Python executable directly.
- If retrieval quality looks too broad, increase `BM25 Weight` for exact
  keyword, code, or variable-name questions. Increase `FAISS Weight` for more
  conceptual questions where semantic similarity matters more than exact words.

## Related Work

This project is motivated by work in dense retrieval, RAG, corrective RAG,
document conversion, contextual chunking, and visual document understanding:

- Karpukhin et al. Dense Passage Retrieval for Open-Domain Question Answering.
  https://arxiv.org/abs/2004.04906
- Lewis et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
  https://arxiv.org/abs/2005.11401
- Yan et al. Corrective Retrieval-Augmented Generation.
  https://arxiv.org/abs/2401.15884
- Guenther et al. Late Chunking: Contextual Chunk Embeddings Using Long-Context
  Embedding Models. https://arxiv.org/abs/2409.04701
- Auepo et al. Docling: An Efficient Open-Source Toolkit for AI-Driven Document
  Conversion. https://arxiv.org/abs/2501.17887
- Faysse et al. ColPali: Efficient Document Retrieval with Vision Language
  Models. https://arxiv.org/abs/2407.01449

## Notes

- `OPENAI_API_KEY` is required for embedding, dense retrieval, and generation.
- Embedding JSON files are large and ignored by Git.
- `.DS_Store` files are local macOS artifacts and should not be committed.
- If the UI shows `No conversation memory yet`, that is normal before a
  successful conversation. If it also shows `Error`, check the server terminal
  for API key, model, or index-file errors.
