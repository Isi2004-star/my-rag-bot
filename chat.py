import os
import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer

# 1. Paste your free Groq API key here
os.environ["GROQ_API_KEY"] = "gsk_your_api_key_here"
llm = Groq()

# 2. Load the SAME embedding model and saved ChromaDB folder from ingest.py
embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="my_notes")


def ask_bot(question: str):
    # Step A: Turn your question into meaning numbers & grab top 3 cards
    q_vector = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=q_vector, n_results=3)

    # Combine the 3 retrieved chunks with their page numbers
    context_blocks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context_blocks.append(f"[{meta['source']}, Page {meta['page']}]\n{doc}")
    context = "\n\n---\n\n".join(context_blocks)

    # Step B: Give Llama 3.3 the strict open-book exam prompt
    system_prompt = (
        "You are a helpful study assistant. Answer the user's question using ONLY the provided context. "
        "Always cite the source and page number (e.g., [STUDENT-HANDBOOK-2025.pdf, Page 4]). "
        "If the answer is not in the context, say: 'I do not have enough information in the document to answer that.'"
    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    # Step C: Generate the answer with Llama 3.3 70B on Groq
    response = llm.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content


# --- Interactive Chat Loop ---
print("Mineralogy RAG Bot ready! Type 'exit' to quit.\n")
while True:
    user_q = input("You: ")
    if user_q.lower() in ["exit", "quit"]:
        break
    answer = ask_bot(user_q)
    print(f"\nBot: {answer}\n")