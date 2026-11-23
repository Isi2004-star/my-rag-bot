import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# 1. READ THE FILE: Extract text page-by-page from a PDF
reader = PdfReader("Optical Mineralogy - Part I.pdf")
chunks = []
metadata = []

for page_num, page in enumerate(reader.pages, start=1):
    text = page.extract_text() or ""
    
    # 2. CUT INTO INDEX CARDS: Split each page into ~500-character overlapping pieces
    chunk_size = 500
    overlap = 50
    for i in range(0, len(text), chunk_size - overlap):
        piece = text[i : i + chunk_size].strip()
        if len(piece) > 50:  # Skip tiny empty fragments
            chunks.append(piece)
            metadata.append({"source": "sample.pdf", "page": page_num})

print(f"Created {len(chunks)} index cards from {len(reader.pages)} pages.")

# 3. TRANSLATE INTO MEANING NUMBERS: Load a lightweight local embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")
vectors = embedder.encode(chunks).tolist()

# 4. SAVE IN A SMART FILING CABINET: Store in a local ChromaDB folder
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="my_notes")

collection.add(
    ids=[f"card_{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=vectors,
    metadatas=metadata,
)

print("Saved all index cards to ./chroma_db!")

# --- QUICK TEST: Pull the top 2 matching cards for a sample question ---
question = "What is the main topic of this document?"
q_vector = embedder.encode([question]).tolist()

results = collection.query(query_embeddings=q_vector, n_results=2)

print("\n--- Top Matching Cards Found ---")
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"[Page {meta['page']}]: {doc[:200]}...\n")