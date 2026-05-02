#!/usr/bin/env python3
"""Compare EODS retrieval quality for atomic and semantic embeddings."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None


DEFAULT_COURSE_ID = "eods"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_TOP_K = 5
DEFAULT_TARGETS = "both"


@dataclass(frozen=True)
class RetrievalTestCase:
    name: str
    query: str
    expected_keywords: Sequence[str]
    expected_atomic_chunk_types: Sequence[str] = ()


TEST_CASES: List[RetrievalTestCase] = [
    RetrievalTestCase(
        name="python-data-science-stack",
        query="What tools are in the Python data science stack and why use Anaconda?",
        expected_keywords=["python", "anaconda", "jupyter", "numpy", "pandas"],
        expected_atomic_chunk_types=["text"],
    ),
    RetrievalTestCase(
        name="python-data-structures",
        query="How are Python lists, dictionaries, tuples, and sets different?",
        expected_keywords=["list", "dict", "dictionary", "tuple", "set"],
        expected_atomic_chunk_types=["text"],
    ),
    RetrievalTestCase(
        name="pandas-dataframe-selection",
        query="How do I select rows and columns in a pandas DataFrame using loc and slicing?",
        expected_keywords=["pandas", "dataframe", "loc", "index", "slicing"],
        expected_atomic_chunk_types=["text"],
    ),
    RetrievalTestCase(
        name="eda-iqr-correlation",
        query="What are IQR and correlation used for in exploratory data analysis?",
        expected_keywords=["iqr", "quartile", "correlation", "spread"],
        expected_atomic_chunk_types=["text", "text_inline_math", "formula"],
    ),
    RetrievalTestCase(
        name="confidence-intervals",
        query="How do confidence intervals describe uncertainty in a sample statistic?",
        expected_keywords=["confidence interval", "sample", "statistic", "uncertainty"],
        expected_atomic_chunk_types=["text", "formula", "text_inline_math"],
    ),
    RetrievalTestCase(
        name="p-value-hypothesis-testing",
        query="What does a p-value mean in hypothesis testing with a null hypothesis?",
        expected_keywords=["p-value", "hypothesis", "null", "significance"],
        expected_atomic_chunk_types=["text", "text_inline_math"],
    ),
    RetrievalTestCase(
        name="linear-regression-ols",
        query="How does simple linear regression use ordinary least squares to fit a line?",
        expected_keywords=["linear regression", "ordinary least squares", "squared error", "coefficients"],
        expected_atomic_chunk_types=["text", "formula", "text_inline_math"],
    ),
    RetrievalTestCase(
        name="logistic-regression-classification",
        query="What is logistic regression used for and how does it model class probability?",
        expected_keywords=["logistic regression", "classification", "probability", "class"],
        expected_atomic_chunk_types=["text", "text_inline_math"],
    ),
    RetrievalTestCase(
        name="tree-ensemble-models",
        query="How do decision trees, random forests, and gradient boosted trees differ?",
        expected_keywords=["decision tree", "random forest", "gradient boosted", "ensemble"],
        expected_atomic_chunk_types=["text"],
    ),
    RetrievalTestCase(
        name="model-evaluation-generalization",
        query="Why do we use train test splits and cross validation to evaluate generalization?",
        expected_keywords=["train", "test", "cross-validation", "generalization", "validation"],
        expected_atomic_chunk_types=["text"],
    ),
    RetrievalTestCase(
        name="overfitting-regularization",
        query="What is overfitting and how can L1 or L2 regularization help?",
        expected_keywords=["overfitting", "regularization", "lasso", "ridge", "penalty"],
        expected_atomic_chunk_types=["text", "formula", "text_inline_math"],
    ),
    RetrievalTestCase(
        name="classification-errors-confusion-matrix",
        query="How do accuracy, precision, recall, and confusion matrices describe classification errors?",
        expected_keywords=["accuracy", "precision", "recall", "confusion matrix", "classification"],
        expected_atomic_chunk_types=["text", "figure"],
    ),
    RetrievalTestCase(
        name="missing-data-cleaning",
        query="How should missing values and NaNs be handled during data cleaning in pandas?",
        expected_keywords=["missing", "nan", "drop", "fill", "impute"],
        expected_atomic_chunk_types=["text"],
    ),
    RetrievalTestCase(
        name="feature-engineering",
        query="What is feature engineering and why add features before modeling?",
        expected_keywords=["feature engineering", "feature", "model", "data cleaning"],
        expected_atomic_chunk_types=["text"],
    ),
    RetrievalTestCase(
        name="joining-data-pandas",
        query="How do joins combine datasets in pandas using indexes, keys, and merge?",
        expected_keywords=["join", "merge", "index", "key", "pandas"],
        expected_atomic_chunk_types=["text"],
    ),
    RetrievalTestCase(
        name="dimensionality-reduction-pca",
        query="How does PCA reduce dimensionality while preserving variance?",
        expected_keywords=["pca", "principal component", "dimensionality reduction", "variance"],
        expected_atomic_chunk_types=["text", "formula", "text_inline_math"],
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--course-id",
        default=DEFAULT_COURSE_ID,
        help=f"Course identifier. Default: {DEFAULT_COURSE_ID}",
    )
    parser.add_argument(
        "--atomic-embeddings-path",
        help="Optional explicit *_atomic_embeddings.json path.",
    )
    parser.add_argument(
        "--semantic-embeddings-path",
        help="Optional explicit *_semantic_embeddings.json path.",
    )
    parser.add_argument(
        "--targets",
        choices=["atomic", "semantic", "both"],
        default=DEFAULT_TARGETS,
        help=f"Which embedding set(s) to test. Default: {DEFAULT_TARGETS}",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Embedding model for the query side. Default: {DEFAULT_EMBEDDING_MODEL}",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"How many hits to show per query. Default: {DEFAULT_TOP_K}",
    )
    parser.add_argument(
        "--query",
        action="append",
        help="Optional ad-hoc query. Can be passed multiple times. If omitted, built-in EODS test cases are used.",
    )
    return parser.parse_args()


def init_openai_client() -> Any:
    if OpenAI is None or not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OpenAI client is not available")
    return OpenAI()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def resolve_atomic_embeddings_path(course_id: str, explicit_path: str | None) -> Path:
    if explicit_path:
        return Path(explicit_path)
    return Path("data/chunk") / f"{course_id}_atomic_embeddings.json"


def resolve_semantic_embeddings_path(course_id: str, explicit_path: str | None) -> Path:
    if explicit_path:
        return Path(explicit_path)
    return Path("data/chunk") / f"{course_id}_semantic_embeddings.json"


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(a: Sequence[float]) -> float:
    return math.sqrt(dot(a, a))


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    denom = norm(a) * norm(b)
    if denom == 0:
        return 0.0
    return dot(a, b) / denom


def embed_query(client: Any, model: str, query: str) -> List[float]:
    response = client.embeddings.create(model=model, input=[query])
    return list(response.data[0].embedding)


def score_vectors(query_embedding: Sequence[float], vectors: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    for item in vectors:
        values = item.get("values") or []
        if not values:
            continue
        score = cosine_similarity(query_embedding, values)
        scored.append({"score": score, "vector": item})
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored


def compact_text(text: str | None, limit: int = 220) -> str | None:
    if not text:
        return None
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def vector_chunk_type(vector: Dict[str, Any], corpus: str) -> str:
    metadata = vector.get("metadata") or {}
    document = vector.get("document") or {}
    if corpus == "atomic":
        return str(metadata.get("chunk_type") or document.get("chunk_type") or "")
    return "semantic"


def vector_text(vector: Dict[str, Any]) -> str:
    document = vector.get("document") or {}
    return str(document.get("text") or "")


def vector_haystack(vector: Dict[str, Any], corpus: str) -> str:
    document = vector.get("document") or {}
    parts = [
        vector_chunk_type(vector, corpus),
        vector_text(vector),
        json.dumps(document.get("content_for_generation") or {}, ensure_ascii=False),
    ]
    if corpus == "atomic":
        parts.append(json.dumps(document.get("raw_fields") or {}, ensure_ascii=False))
    return " ".join(str(part or "") for part in parts).lower()


def keyword_matches(haystack: str, expected_keywords: Sequence[str]) -> List[str]:
    return [keyword for keyword in expected_keywords if keyword.lower() in haystack]


def hit_matches_expectation(
    *,
    item: Dict[str, Any],
    corpus: str,
    expected_atomic_chunk_types: Sequence[str],
    expected_keywords: Sequence[str],
) -> bool:
    vector = item["vector"]
    haystack = vector_haystack(vector, corpus)
    if not keyword_matches(haystack, expected_keywords):
        return False
    if corpus != "atomic" or not expected_atomic_chunk_types:
        return True
    chunk_type = vector_chunk_type(vector, corpus).lower()
    return chunk_type in {value.lower() for value in expected_atomic_chunk_types}


def build_ad_hoc_cases(queries: Sequence[str]) -> List[RetrievalTestCase]:
    return [
        RetrievalTestCase(
            name=f"custom-{index}",
            query=query,
            expected_keywords=[],
            expected_atomic_chunk_types=[],
        )
        for index, query in enumerate(queries, start=1)
    ]


def print_ranked_hits(corpus: str, ranked: Sequence[Dict[str, Any]], expected_keywords: Sequence[str]) -> None:
    print(f"{corpus.capitalize()} top hits:")
    for rank, item in enumerate(ranked, start=1):
        vector = item["vector"]
        metadata = vector.get("metadata") or {}
        chunk_type = vector_chunk_type(vector, corpus)
        haystack = vector_haystack(vector, corpus)
        matched_keywords = keyword_matches(haystack, expected_keywords)
        if corpus == "semantic":
            location = (
                f"{metadata.get('doc_id')} "
                f"p{metadata.get('page_start')}-{metadata.get('page_end')}"
            )
        else:
            location = f"{metadata.get('doc_id')} p{metadata.get('page_no')}"
        print(
            f"{rank}. score={item['score']:.4f} "
            f"id={vector.get('id')} type={chunk_type} {location}"
        )
        if matched_keywords:
            print(f"   matched: {', '.join(matched_keywords)}")
        print(f"   text: {compact_text(vector_text(vector))}")


def run_test_case_for_corpus(
    *,
    query_embedding: Sequence[float],
    corpus: str,
    vectors: Sequence[Dict[str, Any]],
    test_case: RetrievalTestCase,
    top_k: int,
) -> bool:
    ranked = score_vectors(query_embedding, vectors)[:top_k]
    if test_case.expected_keywords:
        passed = any(
            hit_matches_expectation(
                item=item,
                corpus=corpus,
                expected_atomic_chunk_types=test_case.expected_atomic_chunk_types,
                expected_keywords=test_case.expected_keywords,
            )
            for item in ranked
        )
    else:
        passed = True

    print(f"{corpus.capitalize()} result: {'PASS' if passed else 'FAIL'}")
    print_ranked_hits(corpus, ranked, test_case.expected_keywords)
    return passed


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    target_payloads: Dict[str, Dict[str, Any]] = {}
    if args.targets in {"atomic", "both"}:
        atomic_path = resolve_atomic_embeddings_path(args.course_id, args.atomic_embeddings_path)
        if not atomic_path.exists():
            raise FileNotFoundError(f"Atomic embeddings file not found: {atomic_path}")
        target_payloads["atomic"] = load_json(atomic_path)

    if args.targets in {"semantic", "both"}:
        semantic_path = resolve_semantic_embeddings_path(args.course_id, args.semantic_embeddings_path)
        if not semantic_path.exists():
            raise FileNotFoundError(f"Semantic embeddings file not found: {semantic_path}")
        target_payloads["semantic"] = load_json(semantic_path)

    target_vectors = {
        corpus: payload.get("vectors") or []
        for corpus, payload in target_payloads.items()
    }
    for corpus, vectors in target_vectors.items():
        if not vectors:
            raise ValueError(f"No vectors found in {corpus} embeddings payload")

    client = init_openai_client()
    test_cases = build_ad_hoc_cases(args.query) if args.query else TEST_CASES

    pass_counts = {corpus: 0 for corpus in target_vectors}

    for test_case in test_cases:
        print(f"\n[Test] {test_case.name}")
        print(f"Query: {test_case.query}")
        if test_case.expected_keywords:
            print(f"Expected keywords: {', '.join(test_case.expected_keywords)}")
        if test_case.expected_atomic_chunk_types:
            print(f"Expected atomic chunk types: {', '.join(test_case.expected_atomic_chunk_types)}")

        query_embedding = embed_query(client, args.embedding_model, test_case.query)
        for corpus, vectors in target_vectors.items():
            if run_test_case_for_corpus(
                query_embedding=query_embedding,
                corpus=corpus,
                vectors=vectors,
                test_case=test_case,
                top_k=args.top_k,
            ):
                pass_counts[corpus] += 1
        print("-" * 80)

    print("\nSummary:")
    for corpus, passed in pass_counts.items():
        print(f"{corpus}: {passed}/{len(test_cases)} tests passed")


if __name__ == "__main__":
    main()
