import asyncio
import json
import os
import re
import sqlite3
import chromadb
import edge_tts
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from docx import Document
from groq import Groq
from pptx import Presentation
from pypdf import PdfReader
from sentence_transformers import CrossEncoder, SentenceTransformer

st.set_page_config(page_title="Multi-Modal Voice & Data RAG Assistant", page_icon="🎙️", layout="wide")
st.title("🎙️ Multi-Modal Document & Data RAG Assistant")

MEMORY_FILE = "user_memory.json"
SQLITE_DB = "tabular_data.db"


# --- 1. Load Models & Chroma Client Once (Cached for Speed & Cloud Stability) ---
@st.cache_resource
def load_resources():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder("BAAI/bge-reranker-base")
    try:
        db_client = chromadb.PersistentClient(path="./chroma_db")
    except Exception:
        db_client = chromadb.EphemeralClient()
    return embedder, reranker, db_client


embedder, reranker, client = load_resources()


# --- Helper A: Long-Term Persistent Memory (JSON) ---
def load_memories() -> list[str]:
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_memories(memories: list[str]):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memories, f, indent=2)


# --- Helper B: Multi-Format Document Text Extraction ---
def extract_document_sections(uploaded_file) -> list[tuple[str, str]]:
    """Extracts (location_label, text) pairs from PDF, DOCX, PPTX, HTML, or TXT/MD files."""
    filename = uploaded_file.name.lower()
    sections = []

    if filename.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        for page_num, page in enumerate(reader.pages, start=1):
            sections.append((f"Page {page_num}", page.extract_text() or ""))

    elif filename.endswith(".docx"):
        doc = Document(uploaded_file)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        table_rows = []
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    table_rows.append(row_text)
        combined = "\n".join(paragraphs + table_rows)
        if combined:
            sections.append(("Section 1", combined))

    elif filename.endswith(".pptx"):
        prs = Presentation(uploaded_file)
        for slide_num, slide in enumerate(prs.slides, start=1):
            slide_texts = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    slide_texts.append(shape.text_frame.text)
                if shape.has_table:
                    for row in shape.table.rows:
                        row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                        if row_text:
                            slide_texts.append(row_text)
            sections.append((f"Slide {slide_num}", "\n".join(slide_texts)))

    elif filename.endswith((".html", ".htm")):
        raw_html = uploaded_file.read().decode("utf-8", errors="ignore")
        soup = BeautifulSoup(raw_html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        sections.append(("Web Page", soup.get_text(separator="\n")))

    elif filename.endswith((".txt", ".md")):
        raw_text = uploaded_file.read().decode("utf-8", errors="ignore")
        sections.append(("Text", raw_text))

    return sections


# --- Helper C: Structured CSV -> SQLite Schema, Charts & Text-to-SQL ---
def get_active_csv_schema() -> tuple[str | None, str | None]:
    if not os.path.exists(SQLITE_DB):
        return None, None
    conn = sqlite3.connect(SQLITE_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    if not tables:
        conn.close()
        return None, None

    table_name = tables[0][0]
    cursor.execute(f"PRAGMA table_info('{table_name}');")
    cols = [f"{row[1]} ({row[2]})" for row in cursor.fetchall()]
    sample_df = pd.read_sql_query(f"SELECT * FROM '{table_name}' LIMIT 3;", conn)
    conn.close()

    schema_desc = (
        f"Table name: {table_name}\n"
        f"Columns: {', '.join(cols)}\n"
        f"Sample rows:\n{sample_df.to_string(index=False)}"
    )
    return table_name, schema_desc


def render_bar_chart(df: pd.DataFrame | None):
    if df is None or len(df) <= 1:
        return
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    if len(num_cols) >= 1 and len(cat_cols) >= 1:
        st.bar_chart(df.set_index(cat_cols[0])[num_cols[0]])


def route_query(llm: Groq, question: str, schema_desc: str | None) -> str:
    if not schema_desc:
        return "DOC"
    router_prompt = (
        "You are a query router. Decide whether the user's question is best answered by querying "
        "the active SQL/CSV table below OR by searching the uploaded documents (PDF/DOCX/PPTX/HTML/TXT).\n\n"
        f"Active SQL Table Schema:\n{schema_desc}\n\n"
        "Rules:\n"
        "- Choose CSV if the question asks for rows, rankings ('top 5'), averages, counts, filtering, "
        "or specific values covered by the table columns above.\n"
        "- Choose DOC if the question asks for conceptual definitions, theory, or unstructured document content.\n"
        "Reply with ONLY one word: CSV or DOC."
    )
    res = llm.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": router_prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.0,
    )
    content = (res.choices[0].message.content or "").strip().upper()
    return "CSV" if "CSV" in content else "DOC"


def query_csv_with_sql(llm: Groq, question: str, table_name: str, schema_desc: str, memory_block: str):
    sql_prompt = (
        "You are an expert SQLite data analyst. Convert the user's question into a valid, read-only SQLite SELECT query.\n"
        f"Database Schema:\n{schema_desc}\n\n"
        "Rules:\n"
        "1. Output ONLY raw SQL (no markdown fences, no explanations).\n"
        "2. Only use SELECT statements; never modify or drop tables.\n"
        "3. Wrap column names with spaces or special characters in double quotes."
    )
    sql_res = llm.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": sql_prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.0,
    )
    sql_query = sql_res.choices[0].message.content.strip().strip("`").replace("sql\n", "", 1).strip()

    if not sql_query.upper().startswith("SELECT"):
        raise ValueError("Blocked non-SELECT SQL query for safety.")

    conn = sqlite3.connect(SQLITE_DB)
    result_df = pd.read_sql_query(sql_query, conn)
    conn.close()

    summary_prompt = (
        "Summarize the SQL query results clearly to answer the user's question. "
        "Be accurate with numbers and units."
        f"{memory_block}"
    )
    ans_res = llm.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": summary_prompt},
            {
                "role": "user",
                "content": f"Question: {question}\nSQL Executed: {sql_query}\nResults:\n{result_df.head(20).to_string(index=False)}",
            },
        ],
        temperature=0.1,
    )
    return ans_res.choices[0].message.content, sql_query, result_df


# --- Helper D: Text-to-Speech (TTS) ---
def clean_text_for_speech(text: str) -> str:
    text = re.sub(r"\[[^\]]*(?:Page|Slide|Section|Web Page|Text)[^\]]*\]", "", text)
    text = re.sub(r"[*#`>_~]", "", text)
    return re.sub(r"\s+", " ", text).strip()


async def _synthesize_edge_tts(text: str, voice: str) -> bytes:
    communicate = edge_tts.Communicate(text=text, voice=voice)
    audio_bytes = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_bytes += chunk["data"]
    return audio_bytes


def text_to_speech_bytes(text: str, voice: str) -> bytes:
    spoken_text = clean_text_for_speech(text)
    if not spoken_text:
        return b""
    return asyncio.run(_synthesize_edge_tts(spoken_text, voice))


# --- Helper E: Multi-Turn Query Rewriting ---
def rewrite_query(llm: Groq, chat_history: list, latest_question: str) -> str:
    if not chat_history:
        return latest_question

    recent_turns = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in chat_history[-4:]
    )
    rewrite_prompt = (
        "Rewrite the user's latest message into a standalone search query for document/data retrieval. "
        "Resolve pronouns ('it', 'that') using chat history. Output ONLY the rewritten query (under 25 words)."
    )
    response = llm.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": rewrite_prompt},
            {"role": "user", "content": f"Chat History:\n{recent_turns}\n\nLatest Message: {latest_question}"},
        ],
        temperature=0.0,
    )
    rewritten = (response.choices[0].message.content or "").strip()
    return rewritten if rewritten else latest_question


# --- 2. Sidebar: Setup, Long-Term Memory, TTS & File Uploads ---
with st.sidebar:
    st.header("⚙️ Setup & Controls")
    api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")

    # --- Long-Term User Memory Section ---
    st.subheader("🧠 Long-Term User Memory")
    use_memory = st.toggle("Enable Persistent Memory", value=True)
    memories = load_memories()

    new_fact = st.text_input("Add memory note", placeholder="e.g., I am a 2nd-year Mineralogy student")
    if st.button("Save Memory") and new_fact.strip():
        memories.append(new_fact.strip())
        save_memories(memories)
        st.rerun()

    if memories:
        with st.expander(f"Saved Memories ({len(memories)})"):
            for idx, mem in enumerate(memories):
                col1, col2 = st.columns([5, 1])
                col1.caption(f"• {mem}")
                if col2.button("✕", key=f"del_mem_{idx}"):
                    memories.pop(idx)
                    save_memories(memories)
                    st.rerun()

    # --- Voice Output Settings ---
    st.subheader("🔊 Voice Output (TTS)")
    enable_tts = st.toggle("Speak Answers Aloud", value=False)
    selected_voice = st.selectbox(
        "Neural Voice",
        ["en-US-AriaNeural", "en-US-AndrewNeural", "en-GB-SoniaNeural"],
        disabled=not enable_tts,
    )

    # --- Multi-Modal Knowledge Base Upload (Documents + CSV) ---
    st.subheader("📂 Knowledge Base (Docs & CSV)")
    uploaded_doc = st.file_uploader(
        "Upload Document (PDF, DOCX, PPTX, HTML, TXT)",
        type=["pdf", "docx", "pptx", "html", "htm", "txt", "md"],
    )
    if uploaded_doc and st.button("Index Document"):
        with st.spinner(f"Indexing {uploaded_doc.name}..."):
            sections = extract_document_sections(uploaded_doc)
            chunks, metadata = [], []
            for loc_label, text in sections:
                for i in range(0, len(text), 450):
                    piece = text[i : i + 500].strip()
                    if len(piece) > 50:
                        chunks.append(piece)
                        metadata.append({"source": uploaded_doc.name, "page": loc_label})

            if not chunks:
                st.warning("No readable text found in the uploaded document.")
            else:
                try:
                    client.delete_collection("my_notes")
                except Exception:
                    pass
                collection = client.create_collection("my_notes")
                vectors = embedder.encode(chunks).tolist()
                collection.add(
                    ids=[f"card_{i}" for i in range(len(chunks))],
                    documents=chunks,
                    embeddings=vectors,
                    metadatas=metadata,
                )
                st.success(f"Indexed {len(chunks)} chunks from {uploaded_doc.name}!")

    uploaded_csv = st.file_uploader("Upload CSV Table", type=["csv"])
    if uploaded_csv and st.button("Load CSV into SQLite"):
        df = pd.read_csv(uploaded_csv)
        clean_table = re.sub(r"[^a-zA-Z0-9_]", "_", os.path.splitext(uploaded_csv.name)[0]).lower()
        conn = sqlite3.connect(SQLITE_DB)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        for (old_tbl,) in cursor.fetchall():
            cursor.execute(f"DROP TABLE IF EXISTS '{old_tbl}';")
        df.to_sql(clean_table, conn, if_exists="replace", index=False)
        conn.close()
        st.success(f"Loaded '{clean_table}' ({len(df)} rows, {len(df.columns)} cols)!")

    active_table, _ = get_active_csv_schema()
    if active_table:
        st.info(f"Active SQL Table: `{active_table}`")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- 3. Display Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "dataframe" in msg and msg["dataframe"] is not None:
            render_bar_chart(msg["dataframe"])
        if "sql_query" in msg:
            with st.expander("View Generated SQL & Raw Data"):
                st.code(msg["sql_query"], language="sql")
                if "dataframe" in msg and msg["dataframe"] is not None:
                    st.dataframe(msg["dataframe"], use_container_width=True)
        if "audio_bytes" in msg and msg["audio_bytes"]:
            st.audio(msg["audio_bytes"], format="audio/mp3", autoplay=False)
        if "rewritten_query" in msg:
            st.caption(f"🔍 *Searched for: {msg['rewritten_query']}*")
        if "sources" in msg:
            with st.expander("View Retrieved Document Sources"):
                for src in msg["sources"]:
                    st.caption(src)

# --- 4. Chat Input (Text + Optional Microphone) ---
user_submission = st.chat_input("Ask about your documents or CSV data...", accept_audio=True)

if user_submission:
    if not api_key:
        st.warning("Please enter your Groq API key in the left sidebar first.")
        st.stop()

    llm = Groq(api_key=api_key)
    query_text = user_submission.text

    if user_submission.audio:
        with st.spinner("Transcribing voice query..."):
            transcript = llm.audio.transcriptions.create(
                file=("voice.wav", user_submission.audio.getvalue()),
                model="whisper-large-v3-turbo",
            )
            query_text = transcript.text

    if query_text:
        standalone_query = rewrite_query(llm, st.session_state.messages, query_text)
        st.session_state.messages.append({"role": "user", "content": query_text})

        with st.chat_message("user"):
            st.markdown(query_text)
            if standalone_query.lower() != query_text.lower():
                st.caption(f"🔍 *Searched for: {standalone_query}*")

        # Build persistent memory context if toggle is ON
        memory_block = ""
        if use_memory and memories:
            formatted_mem = "\n".join(f"- {m}" for m in memories)
            memory_block = (
                "\n\nACTIVE USER MEMORY PROFILE:\n"
                f"{formatted_mem}\n"
                "INSTRUCTION: Adapt your explanation style, examples, and structure directly to this user profile. "
                "When answering study questions, add a dedicated '**Study & Exam Tip**' section tailored to their level "
                "(keep all factual claims strictly grounded in the retrieved context)."
            )

        table_name, schema_desc = get_active_csv_schema()
        route = route_query(llm, standalone_query, schema_desc)

        with st.chat_message("assistant"):
            msg_record = {"role": "assistant"}
            if standalone_query.lower() != query_text.lower():
                msg_record["rewritten_query"] = standalone_query

            # --- ROUTE 1: Structured CSV / SQLite Pipeline ---
            if route == "CSV" and table_name and schema_desc:
                with st.spinner(f"Querying SQLite table `{table_name}`..."):
                    try:
                        answer, sql_query, result_df = query_csv_with_sql(
                            llm, standalone_query, table_name, schema_desc, memory_block
                        )
                        st.markdown(answer)
                        render_bar_chart(result_df)
                        with st.expander("View Generated SQL & Raw Data"):
                            st.code(sql_query, language="sql")
                            st.dataframe(result_df, use_container_width=True)

                        msg_record.update({"content": answer, "sql_query": sql_query, "dataframe": result_df})
                    except Exception as e:
                        answer = f"Could not execute SQL query on `{table_name}`: {e}"
                        st.error(answer)
                        msg_record["content"] = answer

            # --- ROUTE 2: Unstructured Document Vector + Reranker Pipeline (with Silent SQL Fallback) ---
            else:
                collection = client.get_or_create_collection("my_notes")
                if collection.count() == 0:
                    answer = "Please upload and index a document (or load a CSV table) in the left sidebar first."
                    st.info(answer)
                    msg_record["content"] = answer
                else:
                    q_vector = embedder.encode([standalone_query]).tolist()
                    raw_results = collection.query(
                        query_embeddings=q_vector,
                        n_results=min(8, collection.count()),
                    )

                    docs = raw_results["documents"][0]
                    metas = raw_results["metadatas"][0]

                    pairs = [[standalone_query, d] for d in docs]
                    scores = reranker.predict(pairs)
                    ranked = sorted(zip(scores, docs, metas), key=lambda x: x[0], reverse=True)[:3]

                    context_blocks, source_previews = [], []
                    for score, doc, meta in ranked:
                        tag = f"[{meta['source']}, {meta['page']}]"
                        context_blocks.append(f"{tag}\n{doc}")
                        source_previews.append(f"**{tag}** (Relevance: {score:.2f}): {doc[:220]}...")

                    context = "\n\n---\n\n".join(context_blocks)

                    system_prompt = (
                        "You are a precise study and data assistant. Answer using ONLY the provided context. "
                        "Cite the exact source and section/page/slide tag (e.g., [filename.pdf, Page 4] or [slides.pptx, Slide 2]) for every claim. "
                        "If the context does not contain the answer, say: 'I do not have enough information in the document to answer that.'"
                        f"{memory_block}"
                    )

                    with st.spinner("Searching documents & generating answer..."):
                        response = llm.chat.completions.create(
                            model="openai/gpt-oss-120b",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {standalone_query}"},
                            ],
                            temperature=0.1,
                        )
                        answer = response.choices[0].message.content

                    # If document lacks the answer, silently fall back to the active SQL table
                    answered_via_sql = False
                    if "do not have enough information" in answer.lower() and table_name and schema_desc:
                        try:
                            sql_answer, sql_query, result_df = query_csv_with_sql(
                                llm, standalone_query, table_name, schema_desc, memory_block
                            )
                            if not result_df.empty:
                                st.markdown(sql_answer)
                                render_bar_chart(result_df)
                                with st.expander("View Generated SQL & Raw Data"):
                                    st.code(sql_query, language="sql")
                                    st.dataframe(result_df, use_container_width=True)
                                msg_record.update(
                                    {"content": sql_answer, "sql_query": sql_query, "dataframe": result_df}
                                )
                                answered_via_sql = True
                        except Exception:
                            pass

                    # Only show the document answer/refusal if SQL fallback did NOT answer it
                    if not answered_via_sql:
                        st.markdown(answer)
                        with st.expander("View Retrieved Document Sources"):
                            for src in source_previews:
                                st.caption(src)
                        msg_record.update({"content": answer, "sources": source_previews})

            # Optional TTS synthesis
            if enable_tts:
                with st.spinner("Synthesizing voice response..."):
                    audio_data = text_to_speech_bytes(msg_record["content"], selected_voice)
                if audio_data:
                    st.audio(audio_data, format="audio/mp3", autoplay=True)
                    msg_record["audio_bytes"] = audio_data

            st.session_state.messages.append(msg_record)