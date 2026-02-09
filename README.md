# chat_bot_for_college

## Semantic search for preprocessed RAG chunks

This repository contains precomputed chunk embeddings in:

- `dataset/formated_chunks_and_embedding.csv`

Use `rag_pipeline/semantic_search.py` to run cosine-similarity search on that file.

### Option 1: search with text query (OpenAI embeddings)

```bash
export OPENAI_API_KEY="your_key"
python rag_pipeline/semantic_search.py --query "What scholarships are available for first-year students?" --top-k 5
```

### Option 2: search with a precomputed query embedding

```bash
python rag_pipeline/semantic_search.py --query-embedding "[0.1, 0.2, ...]" --top-k 5
```

### Useful flags

- `--csv`: choose a different embeddings file (default: `dataset/formated_chunks_and_embedding.csv`)
- `--top-k`: number of chunks to return
- `--min-score`: only return matches above this cosine score
- `--openai-model`: embedding model for `--query` (default: `text-embedding-3-small`)
