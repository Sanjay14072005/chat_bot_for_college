#!/usr/bin/env python3
"""Category-aware semantic search over precomputed RAG chunk embeddings.

This script is designed for datasets where chunk embeddings were created with
`all-mpnet-base-v2`. To keep embedding spaces aligned, query text is embedded
with the same model before retrieval.

Pipeline:
1) Embed query with `all-mpnet-base-v2`.
2) Predict the best matching `category` using category-centroid similarity.
3) Filter chunks to the predicted category.
4) Run cosine-similarity semantic search inside that category.

Expected CSV columns:
- text
- embedding (stringified Python list)
- category
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence

DEFAULT_CSV_PATH = "dataset/formated_chunks_and_embedding.csv"
DEFAULT_MODEL_NAME = "all-mpnet-base-v2"


@dataclass
class Row:
    text: str
    embedding: List[float]
    category: str
    metadata: Dict[str, str]


@dataclass
class Chunk:
    score: float
    text: str
    category: str
    metadata: Dict[str, str]


def parse_embedding(raw: str) -> List[float]:
    try:
        values = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as exc:
        raise ValueError("Invalid embedding format.") from exc
    if not isinstance(values, list):
        raise ValueError("Embedding must be a list.")
    return [float(x) for x in values]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(a: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"Embedding dimension mismatch: {len(a)} vs {len(b)}")
    denom = norm(a) * norm(b)
    if denom == 0:
        return 0.0
    return dot(a, b) / denom


def average(vectors: Sequence[Sequence[float]]) -> List[float]:
    if not vectors:
        raise ValueError("Cannot average empty vectors.")
    size = len(vectors[0])
    out = [0.0] * size
    for vec in vectors:
        if len(vec) != size:
            raise ValueError("Inconsistent embedding dimensions in dataset.")
        for i, value in enumerate(vec):
            out[i] += value
    count = float(len(vectors))
    return [x / count for x in out]


def embed_query_with_mpnet(query: str, model_name: str) -> List[float]:
    """Embed query with all-mpnet-base-v2.

    Priority:
    1) sentence-transformers (recommended)
    2) transformers + torch fallback with mean pooling
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        model = SentenceTransformer(model_name)
        result = model.encode(query, normalize_embeddings=False)
        return [float(x) for x in result.tolist()]
    except ImportError:
        pass

    try:
        import torch  # type: ignore
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Install either `sentence-transformers` (recommended) or "
            "`transformers` + `torch` to embed text queries with "
            f"{model_name}."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    encoded = tokenizer(
        query,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )

    with torch.no_grad():
        output = model(**encoded)

    token_embeddings = output.last_hidden_state
    attention_mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size())
    masked_embeddings = token_embeddings * attention_mask
    summed = masked_embeddings.sum(dim=1)
    counts = attention_mask.sum(dim=1).clamp(min=1e-9)
    mean_pooled = summed / counts
    return [float(x) for x in mean_pooled[0].tolist()]


def load_rows(csv_path: str) -> List[Row]:
    rows: List[Row] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {"text", "embedding", "category"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        for raw in reader:
            rows.append(
                Row(
                    text=raw["text"],
                    embedding=parse_embedding(raw["embedding"]),
                    category=raw.get("category", "").strip() or "UNCATEGORIZED",
                    metadata={
                        "scope": raw.get("scope", ""),
                        "department": raw.get("department", ""),
                        "program": raw.get("program", ""),
                        "year": raw.get("year", ""),
                        "source": raw.get("source", ""),
                    },
                )
            )
    return rows


def build_category_centroids(rows: Sequence[Row]) -> Dict[str, List[float]]:
    grouped: Dict[str, List[List[float]]] = {}
    for row in rows:
        grouped.setdefault(row.category, []).append(row.embedding)
    return {category: average(vectors) for category, vectors in grouped.items()}


def predict_query_category(
    query_embedding: Sequence[float], centroids: Dict[str, List[float]]
) -> tuple[str, float]:
    if not centroids:
        raise ValueError("No categories found in dataset.")
    best_category = ""
    best_score = -1.0
    for category, centroid in centroids.items():
        score = cosine_similarity(query_embedding, centroid)
        if score > best_score:
            best_category = category
            best_score = score
    return best_category, best_score


def semantic_search(
    rows: Sequence[Row],
    query_embedding: Sequence[float],
    top_k: int,
    min_score: float,
    forced_category: str | None,
) -> tuple[str, float, List[Chunk]]:
    centroids = build_category_centroids(rows)

    if forced_category:
        selected_category = forced_category
        if selected_category not in centroids:
            available = ", ".join(sorted(centroids))
            raise ValueError(
                f"Category '{selected_category}' not found. Available categories: {available}"
            )
        category_score = cosine_similarity(query_embedding, centroids[selected_category])
    else:
        selected_category, category_score = predict_query_category(query_embedding, centroids)

    filtered = [row for row in rows if row.category == selected_category]

    results: List[Chunk] = []
    for row in filtered:
        score = cosine_similarity(query_embedding, row.embedding)
        if score < min_score:
            continue
        metadata = dict(row.metadata)
        metadata["category"] = row.category
        results.append(
            Chunk(score=score, text=row.text, category=row.category, metadata=metadata)
        )

    results.sort(key=lambda item: item.score, reverse=True)
    return selected_category, category_score, results[:top_k]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Category-aware semantic search over RAG chunk embeddings"
    )
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH, help="Path to embeddings CSV")
    parser.add_argument("--query", help="Text query (embedded with all-mpnet-base-v2)")
    parser.add_argument(
        "--query-embedding",
        help="Optional precomputed query embedding list. Useful for testing.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Embedding model for text query. Keep as all-mpnet-base-v2.",
    )
    parser.add_argument(
        "--category",
        help="Optional: force a specific category instead of auto-detecting from query.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to return")
    parser.add_argument(
        "--min-score", type=float, default=-1.0, help="Minimum cosine score threshold"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if bool(args.query) == bool(args.query_embedding):
        print("Provide exactly one of --query or --query-embedding.", file=sys.stderr)
        return 2
    if args.top_k <= 0:
        print("--top-k must be > 0", file=sys.stderr)
        return 2

    try:
        rows = load_rows(args.csv)
        if args.query_embedding:
            query_embedding = parse_embedding(args.query_embedding)
        else:
            query_embedding = embed_query_with_mpnet(args.query, args.model)

        predicted_category, category_score, hits = semantic_search(
            rows=rows,
            query_embedding=query_embedding,
            top_k=args.top_k,
            min_score=args.min_score,
            forced_category=args.category,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Selected category: {predicted_category} (score={category_score:.4f})")

    if not hits:
        print("No results found in selected category.")
        return 0

    for i, hit in enumerate(hits, start=1):
        print(f"#{i} score={hit.score:.4f}")
        print(f"text: {hit.text}")
        print("metadata:")
        for key, value in hit.metadata.items():
            if value:
                print(f"  - {key}: {value}")
        print("-" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
