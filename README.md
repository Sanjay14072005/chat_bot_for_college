# 🎓 College RAG Chatbot
### *LLaMA-3 + Category-Aware Retrieval*

A Retrieval-Augmented Generation (RAG) chatbot built using **Meta LLaMA-3-8B-Instruct**, **SentenceTransformers**, and **Gradio**, designed to answer institutional queries using structured academic data.

---

## 🚀 Project Overview

This project implements a **category-aware RAG pipeline** for answering questions related to a college or institution.

The system:
- Loads precomputed text embeddings from a CSV file
- Computes category centroids for efficient semantic filtering
- Selects top relevant categories using cosine similarity
- Performs fine-grained chunk retrieval inside selected categories
- Injects retrieved context into a structured prompt
- Generates answers using Meta-LLaMA-3-8B-Instruct
- Serves responses through a Gradio chat interface

---

## 🧠 Architecture

### 🔎 Retrieval Layer
- **Embedding model:** `all-mpnet-base-v2`
- Category-level centroid similarity filtering
- Chunk-level cosine similarity scoring
- Top-*5* results per category

### 🤖 Generation Layer
- **Model:** `meta-llama/Meta-Llama-3-8B-Instruct`
- Prompt includes:
  - System instruction
  - Conversation history
  - Retrieved contextual chunks
  - User query

### 💬 Interface
- **Gradio ChatInterface**
- Multi-turn conversational memory
- Context-aware responses

---

## 📂 Dataset

The system expects a CSV file containing:

| text | embedding | category | department | source |
|------|-----------|----------|------------|--------|

> ⚠️ **Note:** The real institutional dataset is private and not included in this repository.

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/chat_bot_for_college.git
cd chat_bot_for_college
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🔑 HuggingFace Authentication

LLaMA-3 requires access approval. Login before running:

### Python
```python
from huggingface_hub import login
login()
```

### CLI
```bash
huggingface-cli login
```

---

## ▶️ Run the Application

```bash
python chat_bot.py
```

Gradio will launch a local interface.

---

## 🏗 How Retrieval Works

1. Query embedding is computed.
2. Cosine similarity is calculated against category centroids.
3. Top **2** categories are selected.
4. Chunk-level similarity is computed within selected categories.
5. Top-*5* results per category are passed to the LLM.

This two-stage retrieval improves precision and reduces noise.

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- SentenceTransformers
- Meta LLaMA-3-8B-Instruct
- HuggingFace Transformers
- Gradio

---

## 📌 Features

- ✔ Category-aware semantic filtering
- ✔ Multi-turn conversational memory
- ✔ Context-restricted answering
- ✔ Modular retrieval pipeline
- ✔ GPU-compatible

---

## 🚧 Future Improvements

- FAISS-based vector indexing
- Hybrid keyword + semantic retrieval
- Intent detection layer
- Structured timetable rendering
- Deployment to HuggingFace Spaces

---

## 👨‍💻 Author

Developed by **Sanjay S**  
Focused on building practical, scalable RAG systems.
