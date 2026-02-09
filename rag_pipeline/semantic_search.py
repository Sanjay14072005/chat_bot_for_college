#!/usr/bin/env python3
"""Semantic search over precomputed chunk embeddings stored in CSV.

Expected CSV columns (as in dataset/formated_chunks_and_embedding.csv):
- text
- embedding  (stringified Python list)

Usage examples:
1) Use OpenAI embeddings for query (requires OPENAI_API_KEY):
   python rag_pipeline/semantic_search.py --query "What is the admission process?" --top-k 5

2) Provide a precomputed query embedding directly:
   python rag_pipeline/semantic_search.py --query-embedding "[0.1, 0.2, ...]" --top-k 3
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


DEFAULT_CSV_PATH = "dataset/formated_chunks_and_embedding.csv"
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"


@dataclass
class Chunk:
    score: float
    text: str
    metadata: Dict[str, str]


def parse_embedding(raw: str) -> List[float]:
    try:
        values = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as exc:
        raise ValueError("Invalid embedding format in CSV.") from exc
    if not isinstance(values, list):
        raise ValueError("Embedding must be a list of floats.")
    return [float(x) for x in values]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(a: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(
            f"Embedding dimension mismatch: query={len(a)} vs row={len(b)}"
        )
    denom = norm(a) * norm(b)
    if denom == 0:
        return 0.0
    return dot(a, b) / denom


def embed_query_openai(query: str, model: str) -> List[float]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Provide --query-embedding or set the API key."
        )

    url = "https://api.openai.com/v1/embeddings"
    payload = json.dumps({"model": model, "input": query}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"OpenAI API error ({exc.code}): {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error while calling OpenAI API: {exc}") from exc

    try:
        return [float(x) for x in data["data"][0]["embedding"]]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected OpenAI response: {data}") from exc


def load_rows(csv_path: str) -> Iterable[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {"text", "embedding"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
        for row in reader:
            yield row


def semantic_search(
    csv_path: str,
    query_embedding: Sequence[float],
    top_k: int,
    min_score: float,
) -> List[Chunk]:
    results: List[Chunk] = []

    for row in load_rows(csv_path):
        row_embedding = parse_embedding(row["embedding"])
        score = cosine_similarity(query_embedding, row_embedding)
        if score < min_score:
            continue
        metadata = {
            "scope": row.get("scope", ""),
            "department": row.get("department", ""),
            "program": row.get("program", ""),
            "year": row.get("year", ""),
            "source": row.get("source", ""),
            "category": row.get("category", ""),
        }
        results.append(Chunk(score=score, text=row["text"], metadata=metadata))

    results.sort(key=lambda item: item.score, reverse=True)
    return results[:top_k]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic search over RAG chunk embeddings")
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH, help="Path to embeddings CSV")
    parser.add_argument("--query", help="Text query (embedded via OpenAI API)")
    parser.add_argument(
        "--query-embedding",
        help="Precomputed embedding as JSON/Python list string, e.g. '[0.1, 0.2]'",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument(
        "--min-score",
        type=float,
        default=-1.0,
        help="Minimum cosine similarity to include in output",
    )
    parser.add_argument(
        "--openai-model",
        default=DEFAULT_OPENAI_MODEL,
        help="OpenAI embedding model for --query",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if bool(args.query) == bool(args.query_embedding):
        print(
            "Provide exactly one of --query or --query-embedding.",
            file=sys.stderr,
        )
        return 2

    if args.top_k <= 0:
        print("--top-k must be > 0", file=sys.stderr)
        return 2

    try:
        if args.query_embedding:
            query_embedding = parse_embedding(args.query_embedding)
        else:
            query_embedding = embed_query_openai(args.query, args.openai_model)

        hits = semantic_search(
            csv_path=args.csv,
            query_embedding=query_embedding,
            top_k=args.top_k,
            min_score=args.min_score,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not hits:
        print("No results found.")
        return 0

    for i, hit in enumerate(hits, start=1):
        print(f"#{i}  score={hit.score:.4f}")
        print(f"text: {hit.text}")
        print("metadata:")
        for k, v in hit.metadata.items():
            if v:
                print(f"  - {k}: {v}")
        print("-" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
