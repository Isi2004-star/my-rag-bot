import json
import os
import time
import chromadb
from groq import Groq
from sentence_transformers import CrossEncoder, SentenceTransformer

# 1. Configure Groq API Key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "gsk_your_api_key_here")
llm = Groq()

# 2. Load the exact same models & ChromaDB index used in app.py
print("Loading embedding and reranker models...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("BAAI/bge-reranker-base")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="my_notes")


def retrieve_and_answer(question: str):
    start_time = time.time()

    # Stage 1: Vector search (top 8)
    q_vector = embedder.encode([question]).tolist()
    raw_results = collection.query(query_embeddings=q_vector, n_results=8)
    docs = raw_results["documents"][0]
    metas = raw_results["metadatas"][0]

    # Stage 2: Cross-Encoder reranking (top 3)
    pairs = [[question, d] for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs, metas), key=lambda x: x[0], reverse=True)[:3]

    retrieved_pages = [meta["page"] for _, _, meta in ranked]
    context = "\n\n---\n\n".join(
        f"[{meta['source']}, Page {meta['page']}]\n{doc}" for _, doc, meta in ranked
    )

    # Stage 3: Grounded LLM Generation
    system_prompt = (
        "You are a precise study assistant. Answer using ONLY the provided context. "
        "Cite the exact source and page number (e.g., [Optical_Mineralogy_-_Part_I.pdf, Page 7]) for every claim. "
        "If the context does not contain the answer, say: 'I do not have enough information in the document to answer that.'"
    )

    response = llm.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.1,
    )
    latency = round(time.time() - start_time, 2)
    return response.choices[0].message.content, context, retrieved_pages, latency


def judge_answer(question: str, context: str, expected_answer: str, bot_answer: str, is_unanswerable: bool):
    judge_prompt = f"""You are an impartial RAG evaluation judge. Score the Bot Answer on two metrics from 1 to 5:

1. faithfulness (1-5): Does the Bot Answer rely ONLY on the Retrieved Context without inventing facts? (5 = 100% grounded or properly refused, 1 = major hallucination).
2. correctness (1-5): Does the Bot Answer accurately convey the Expected Answer? (If is_unanswerable is True, score 5 ONLY if the bot refused to answer).

Question: {question}
is_unanswerable: {is_unanswerable}
Expected Answer: {expected_answer}
Retrieved Context:
{context}

Bot Answer:
{bot_answer}

Respond ONLY in valid JSON with keys: "faithfulness" (int), "correctness" (int), "reason" (short string)."""

    res = llm.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(res.choices[0].message.content)


def main():
    with open("eval/questions.json", "r", encoding="utf-8") as f:
        payload = json.load(f)

    # Supports both {"questions": [...]} and a direct list [...]
    dataset = payload["questions"] if isinstance(payload, dict) else payload

    results = []
    hits = 0
    answerable_count = 0
    keyword_scores = []

    print(f"\nRunning evaluation on {len(dataset)} questions...\n")

    for item in dataset:
        is_unanswerable = item.get("is_unanswerable", len(item.get("expected_pages", [])) == 0)
        bot_answer, context, retrieved_pages, latency = retrieve_and_answer(item["question"])

        # 1. Retrieval Hit Rate @3
        hit = None
        if not is_unanswerable:
            answerable_count += 1
            hit = any(p in retrieved_pages for p in item["expected_pages"])
            if hit:
                hits += 1

        # 2. Deterministic Keyword Recall
        keywords = item.get("keywords", [])
        if keywords:
            matched_kw = sum(1 for kw in keywords if kw.lower() in bot_answer.lower())
            kw_recall = round((matched_kw / len(keywords)) * 100, 1)
            keyword_scores.append(kw_recall)
        else:
            kw_recall = None

        # 3. LLM-as-a-Judge (Faithfulness & Correctness)
        scores = judge_answer(
            item["question"], context, item["expected_answer"], bot_answer, is_unanswerable
        )

        record = {
            "id": item["id"],
            "question": item["question"],
            "expected_pages": item["expected_pages"],
            "retrieved_pages": retrieved_pages,
            "retrieval_hit": hit,
            "keyword_recall_pct": kw_recall,
            "faithfulness": scores["faithfulness"],
            "correctness": scores["correctness"],
            "latency_sec": latency,
            "reason": scores["reason"],
            "bot_answer": bot_answer,
        }
        results.append(record)
        print(
            f"[Q{item['id']:02d}] Hit: {hit} | KW: {kw_recall}% | "
            f"Faith: {scores['faithfulness']}/5 | Corr: {scores['correctness']}/5 | {latency}s"
        )

    hit_rate = (hits / answerable_count * 100) if answerable_count else 0
    avg_kw = (sum(keyword_scores) / len(keyword_scores)) if keyword_scores else 0
    avg_faith = sum(r["faithfulness"] for r in results) / len(results)
    avg_corr = sum(r["correctness"] for r in results) / len(results)
    avg_latency = sum(r["latency_sec"] for r in results) / len(results)

    summary = {
        "source_pdf": payload.get("source_pdf", "unknown") if isinstance(payload, dict) else "unknown",
        "total_questions": len(dataset),
        "retrieval_hit_rate_at_3_pct": round(hit_rate, 2),
        "avg_keyword_recall_pct": round(avg_kw, 2),
        "avg_faithfulness_out_of_5": round(avg_faith, 2),
        "avg_correctness_out_of_5": round(avg_corr, 2),
        "avg_latency_sec": round(avg_latency, 2),
        "details": results,
    }

    with open("eval/results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n================ BENCHMARK SUMMARY ================")
    print(f"Retrieval Hit Rate @3 : {hit_rate:.1f}% ({hits}/{answerable_count})")
    print(f"Avg Keyword Recall    : {avg_kw:.1f}%")
    print(f"Avg Faithfulness      : {avg_faith:.2f} / 5.00")
    print(f"Avg Answer Correctness: {avg_corr:.2f} / 5.00")
    print(f"Avg End-to-End Latency: {avg_latency:.2f}s")
    print("Full report saved to eval/results.json")


if __name__ == "__main__":
    main()