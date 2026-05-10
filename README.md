# Course RAG: A Transferable Course Knowledge Assistant

This project is a course-specific Retrieval-Augmented Generation system for answering student questions from structured course materials. Instead of relying on a general-purpose chatbot's internal memory, the system converts lecture slides and course PDFs into a searchable knowledge base, retrieves relevant course chunks, and generates answers grounded in those chunks with source citations.

The project was developed for the STATS-G5293 Generative AI course final project by Serena Cheng, Fei Xue, and Yining Tao.

The system currently supports multiple indexed courses, including Elements of Data Science (EODS), Applied Deep Learning (ADL), and Statistical Inference (5703).

## Key Features

- **Ask questions across courses**: Select an indexed course and ask natural-language questions about its lectures, notes, quizzes, or homework materials.
- **Get cited answers**: Receive answers grounded in retrieved course content, with inline citations such as `[S1]`.
- **Inspect source evidence**: Review the retrieved text, formula, figure, and semantic chunks behind each citation, including document/page references and figure previews when available.
- **Short-term Memory**: Ask follow-up questions such as "Can you give an example?" while the assistant uses recent conversation context to interpret the request.
- **Tune the QA behavior**: Adjust retrieval and generation settings from the UI, including FAISS/BM25 weights, candidate count, memory window, and model choices.
- **Create new course assistants**: Add a new course ID/name, upload PDFs, and run the ingestion/indexing pipeline from the demo interface.
- **Update existing courses**: Add more PDFs to an existing course and rebuild the searchable knowledge base without leaving the UI.
- **Track processing progress**: Monitor ingestion stages while course documents are extracted, cleaned, chunked, embedded, and indexed.
- **Support multimodal course content**: Work with course materials containing prose, formulas, figures, code examples, notebook screenshots, and slide images.

## Repository Structure

```text
data/
  raw/                  Raw course PDFs
  processed/            Extracted and cleaned document JSON
  chunk/                Atomic/semantic chunks; large embedding JSON files are stored externally
  retrieval/            FAISS indexes and metadata
  test/                 Retrieval evaluation datasets

scripts/
  extraction/           PDF extraction and document post-processing
  chunk/                Atomic chunking, semantic chunking, embeddings
  retrieval/            FAISS index building, retrieval, generation, evaluation
  run_before_chunk.py   Prepares cleaned blocks before chunking
  demo_server.py        Local backend server for the UI
  evaluate_three_datasets.ipynb
                        Notebook for running retrieval evaluation across EODS, ADL, and 5703

demo_ui/
  index.html            Demo frontend
  app.js                UI logic
  styles.css            UI styles
```

## Architecture

### 1. Data Preprocessing

- `scripts/extraction/extract_course_documents.py` parses PDFs from `data/raw/<course_id>/` into normalized document JSON at `data/processed/<course_id>/<course_id>_document.json`.
- Course-specific post-processors, such as `post_process_document_eods.py`, `post_process_document_adl.py`, and `post_process_document_5703.py`, clean extracted blocks and handle artifacts such as OCR noise, code screenshots, figure text, and malformed formulas.
- `scripts/extraction/post_process_document_generic.py` provides a reusable fallback for courses without custom cleaning logic.

### 2. Pre-Chunk Enrichment

- `scripts/run_before_chunk.py` coordinates the follwing text, formula, and figure preparation before chunking.
- `scripts/extraction/text_before_chunk.py`, `formula_before_chunk.py`, and `figure_before_chunk.py` enrich blocks with fields such as `section_title`, `nearby_text_before`, `nearby_text_after`, `formula_focus`, `formula_explanation`, `visual_description`, and `math_spans`.
- The merged output is written to `data/processed/<course_id>/<course_id>_merged.json`.

### 3. Chunking

- `scripts/chunk/atomic_chunk.py` creates small retrieval units for text, formulas, figures, and inline-math content.
- `scripts/chunk/semantic_chunk.py` merges adjacent atomic chunks using embedding similarity, section continuity, document/page order, token limits, and auxiliary figure context.
- Atomic chunks maximize retrieval precision; semantic chunks provide broader context for generation.

### 4. Embedding

- `scripts/chunk/atomic_embedding.py` embeds atomic chunks.
- `scripts/chunk/semantic_embedding.py` embeds semantic chunks.
- Embedding records are stored under `data/chunk/` and keep retrieval-optimized text separate from generation-ready content.

### 5. Indexing

- `scripts/retrieval/build_faiss_index.py` builds local FAISS indexes for atomic and semantic embeddings.
- Vectors are L2-normalized, and FAISS inner-product search is used as cosine-similarity search.
- Index files and metadata are written to `data/retrieval/`.

### 6. Retrieval

- `scripts/retrieval/retrieve_faiss_bm25.py` supports `bm25`, `dense`, `hybrid`, and `dense_rerank` retrieval.
- Hybrid retrieval combines FAISS and BM25 rankings with Reciprocal Rank Fusion.
- Dense reranking combines dense scores, BM25 scores, diversity controls, formula/figure preferences, low-information penalties, and same-page repetition penalties.

### 7. Generation

- `scripts/retrieval/generate_answer.py` retrieves candidate chunks, selects source-grounded context, renders source labels, and calls the OpenAI Responses API.
- The generation prompt asks the model to answer only from retrieved course context, cite sources inline, and include a short Sources section.
- The demo server adds short conversation memory through recent turns, a rolling summary, and the current topic while keeping final answers grounded in retrieved chunks.

## Reproducible Workflow

### 1. Environment Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate ai-course-assistant
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

If you are using a different Python environment, install the core dependencies:

```bash
pip install openai faiss-cpu docling pillow paddleocr paddlepaddle pix2tex numpy pandas ipykernel jupyterlab
```

### 2. Build or Rebuild a Course Knowledge Base

Place course PDFs under:

```text
data/raw/<course_id>/
```

The examples below use `Elements of Data Science` as the sample course.

Run extraction:

```bash
python scripts/extraction/extract_course_documents.py \
  --course-id eods \
  --course-name "Elements of Data Science"
```

Run course-specific or generic post-processing:

```bash
python scripts/extraction/post_process_document_eods.py
```

For a course without a custom processor:

```bash
python scripts/extraction/post_process_document_generic.py --course-id <course_id>
```

Run the full chunking and indexing pipeline:

```bash
python scripts/run_before_chunk.py --course-id eods
python scripts/chunk/atomic_chunk.py --course-id eods
python scripts/chunk/atomic_embedding.py --course-id eods
python scripts/chunk/semantic_chunk.py --course-id eods
python scripts/chunk/semantic_embedding.py --course-id eods
python scripts/retrieval/build_faiss_index.py --course-id eods --target both
```

The embedding outputs, such as `<course_id>_atomic_embeddings.json` and `<course_id>_semantic_embeddings.json`, are too large to keep in the repository. Precomputed embedding JSON files are stored in [Google Drive](https://drive.google.com/drive/u/1/folders/1CFyPW-eOTLl8XfMbOfz1_KgifovqX-c4). To reuse them, download the files into `data/chunk/` before running `scripts/retrieval/build_faiss_index.py`.

### 3. Run Retrieval

```bash
python scripts/retrieval/retrieve_faiss_bm25.py \
  --course-id eods \
  --query "What does a p-value mean in hypothesis testing?" \
  --target both \
  --method dense_rerank \
  --top-k 4 \
  --candidate-k 30
```

### 4. Run Generation

```bash
python scripts/retrieval/generate_answer.py \
  --course-id eods \
  --query "How does PCA reduce dimensionality while preserving variance?"
```

The generated answer should include inline source citations such as `[S1]` and a Sources section with document/page references.

### 5. Evaluate Retrieval

Gold retrieval evaluation files are stored in `data/test/`.

Example:

```bash
python scripts/retrieval/test_retrieval.py \
  --course-id eods \
  --eval-json data/test/eods_retrieval_eval_40.json
```

Common options:

- `--course-id`: choose which course index to evaluate, such as `eods`, `adl`, or `5703`.
- `--eval-json`: choose the gold evaluation file, such as `data/test/eods_retrieval_eval_20.json` or `data/test/eods_retrieval_eval_40.json`.
- `--target`: choose the retrieval corpus: `atomic`, `semantic`, or `both`. The default is `both`.
- `--method`: choose the retrieval strategy: `bm25`, `dense`, `hybrid`, or `dense_rerank`. The default is `dense_rerank`.
- `--candidate-k`: choose how many candidate chunks are retrieved before evaluation. The default is `4`; larger values usually improve recall but may reduce precision.
- `--verbose`: print top retrieved chunks for each query, which is useful for debugging retrieval failures.

To evaluate all three course datasets in one place, open:

```text
scripts/evaluate_three_datasets.ipynb
```

The notebook directly calls `scripts/retrieval/test_retrieval.py` for EODS, ADL, and 5703, then summarizes recall, precision, MRR, and nDCG in pandas tables.

### 6. Reproducibility Checklist

- `environment.yml` defines the Python environment.
- Raw course materials are stored under `data/raw/`.
- Processed documents, chunks, FAISS indexes, and retrieval metadata are stored under `data/processed/`, `data/chunk/`, and `data/retrieval/`.
- Large embedding JSON files are stored externally in [Google Drive](https://drive.google.com/drive/u/1/folders/1CFyPW-eOTLl8XfMbOfz1_KgifovqX-c4) and should be placed under `data/chunk/` when reproducing precomputed indexes.
- Retrieval evaluation datasets are stored under `data/test/`.
- Rebuild commands cover extraction, post-processing, pre-chunk enrichment, chunking, embedding, indexing, retrieval, generation, and evaluation.
- OpenAI-dependent steps require `OPENAI_API_KEY` and the same embedding/generation model names for comparable results.

### 7. Evaluation Scope

The project focuses on retrieval quality, answer grounding, citation accuracy, source traceability, and transferability across courses. Large-scale model pretraining, production deployment, video/audio processing, and complex personalization are outside the current scope.

### 8. Troubleshooting Notes

- If the UI returns `Error`, check the terminal running `scripts/demo_server.py`; backend exceptions are printed there.
- If `OPENAI_API_KEY is not set`, export the key before starting the server.
- If no sources are returned, confirm that the course has FAISS index and metadata files in `data/retrieval/`, such as `eods_atomic.faiss`, `eods_atomic_metadata.json`, `eods_semantic.faiss`, and `eods_semantic_metadata.json`.
- If uploaded PDFs appear stuck during extraction, check server logs; large PDFs or duplicate uploads can take longer because ingestion scans unprocessed PDFs for that course.
- If retrieval is too broad, increase BM25 weight for exact keyword/code questions or FAISS weight for conceptual questions.

### 9. Related Work

This project is motivated by work in dense retrieval, RAG, corrective RAG, contextual chunking, document conversion, and visual document understanding, including Dense Passage Retrieval, Retrieval-Augmented Generation, Corrective RAG, Late Chunking, Docling, and ColPali.

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

## Demo Usage

Demo recording link: https://drive.google.com/file/d/1mv3pHohkSE9gQuw3Ky1dJ18SRuKCIdCh/view?usp=drive_link
<img width="2048" height="1158" alt="7b91163e26c984cf7741ba3c81af2359" src="https://github.com/user-attachments/assets/cb173ca4-39a7-4d98-8afc-51bbfb1e30b7" />


Start the local demo server from the repository root:

```bash
python scripts/demo_server.py
```

If your shell does not resolve the expected Python environment, run it with the conda Python directly:

```bash
/opt/miniconda3/bin/python scripts/demo_server.py
```

Open the demo at:

```text
http://127.0.0.1:8000
```

If port 8000 is already in use:

```bash
python scripts/demo_server.py --port 8001
```

In the demo UI, you can:

- select an indexed course;
- ask course-specific questions;
- adjust retrieval and generation settings;
- inspect generated answers and cited sources;
- view figure/image sources when available;
- create a new course;
- upload PDFs to an existing course; and
- run ingestion and indexing from the interface.

Example questions:

```text
adl: Show me the logistic regression computational graph
adl: How is Recall@k defined?
adl: What is logistic regression class probabilities formula?
5703: What does a p-value mean in hypothesis testing?
eods: How does PCA reduce dimensionality while preserving variance?
follow-up: Can you give an example?
```
