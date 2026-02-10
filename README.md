# chat_bot_for_college

## RAG semantic search (category-aware)

This repository contains precomputed chunk embeddings in:

- `dataset/formated_chunks_and_embedding.csv`

Use `rag_pipeline/semantic_search.py` for retrieval.

### Retrieval logic

The script follows this pipeline to improve accuracy:

1. Embed query with the **same model used for dataset embeddings**: `all-mpnet-base-v2`
2. Predict which `category` the query belongs to
3. Filter rows to that category
4. Run cosine-similarity semantic search only inside that category

### Run with text query (recommended)

```bash
python rag_pipeline/semantic_search.py --query "what is the college timing?" --top-k 5
```

> Requirement: install one of the following:
> - `sentence-transformers` (recommended)
> - OR `transformers` + `torch`

### Run with precomputed query embedding (testing/debug)

```bash
python rag_pipeline/semantic_search.py --query-embedding "[0.1, 0.2, ...]" --top-k 5
```

### Useful flags

- `--csv`: path to embeddings CSV (default: `dataset/formated_chunks_and_embedding.csv`)
- `--model`: query embedding model (default: `all-mpnet-base-v2`)
- `--category`: force a specific category (skip auto category prediction)
- `--top-k`: number of chunks to return
- `--min-score`: only return matches above this cosine score
