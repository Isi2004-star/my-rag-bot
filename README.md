# 🎙️ Multi-Modal Voice & Data RAG Assistant

An open-source, hybrid Retrieval-Augmented Generation (RAG) and Text-to-SQL assistant built with **Python, Streamlit, ChromaDB, SQLite, and Groq**. Designed to query both unstructured PDF lecture slides and structured CSV datasets through text or full-duplex voice.

## ✨ Key Features

- **Two-Stage Hybrid Document Retrieval:** Combines dense semantic search (`all-MiniLM-L6-v2` + ChromaDB) with a Cross-Encoder reranker (`BAAI/bge-reranker-base`) and page-level source citations.
- **Automated Query Routing & Text-to-SQL:** Dynamically routes tabular questions to a read-only SQLite engine (`pandas` + `sqlite3`), auto-rendering interactive bar charts and collapsible SQL audit logs—with a silent PDF-to-SQL fallback.
- **Multi-Turn Conversation Memory:** Rewrites follow-up queries using recent chat context to resolve pronouns before retrieval.
- **Persistent User Profile Memory:** Stores user preferences across sessions (`user_memory.json`) with a real-time toggle to adapt explanations and study tips.
- **Bidirectional Voice Pipeline:** Transcribes spoken questions via Groq `whisper-large-v3-turbo` and synthesizes natural speech responses via `edge-tts`.

## 🏗️ Project Structure

    my-rag-bot/
    ├── app.py                              # Main Streamlit web app (Voice + PDF RAG + Text-to-SQL + Memory)
    ├── ingest.py                           # CLI document indexing script
    ├── chat.py                             # CLI terminal prototype
    ├── eval/
    │   ├── questions.json                  # 23-question domain benchmark dataset
    │   └── run_eval.py                     # Automated retrieval & LLM-as-a-Judge evaluation script
    ├── Optical Mineralogy - Part I.pdf     # Sample lecture slides knowledge base
    ├── optical_minerals.csv                # Sample structured mineral properties dataset
    ├── requirements.txt                    # Python dependencies
    └── .gitignore                          # Excludes local DBs, secrets, and runtime caches

## 📊 Automated Evaluation Benchmark (`eval/run_eval.py`)

Evaluated across a 23-question optical mineralogy benchmark (20 factual questions + 3 out-of-scope hallucination traps):

| Metric | Score |
| :--- | :---: |
| **Retrieval Hit Rate @3** | **90.0%** (18/20) |
| **Avg Faithfulness (1–5)** | **4.78 / 5.00** |
| **Avg Answer Correctness (1–5)** | **4.57 / 5.00** |
| **Out-of-Scope Refusal Rate** | **100.0%** (3/3) |
| **Avg End-to-End Latency** | **2.16s** |

## 🚀 Quick Start

1. **Install dependencies:**

       pip install -r requirements.txt

2. **Launch the Streamlit app:**

       streamlit run app.py

3. **Configure & Chat:**
   - Paste your **Groq API Key** in the left sidebar.
   - Upload a PDF and/or CSV file (or use the included `Optical Mineralogy - Part I.pdf` and `optical_minerals.csv`) and click **Index PDF** / **Load CSV into SQLite**.
   - Type a question or click the microphone icon in the chat bar to interact via voice.

## 🧪 Running the Evaluation Suite

First index `Optical Mineralogy - Part I.pdf` so `./chroma_db` is populated, then run:

- **Windows (PowerShell):**

      $env:GROQ_API_KEY="gsk_your_api_key_here"
      python eval\run_eval.py

- **macOS / Linux:**

      export GROQ_API_KEY="gsk_your_api_key_here"
      python eval/run_eval.py

