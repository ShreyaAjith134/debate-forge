import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("debate_corpus")

TOPIC_MAP = {
    "Should AI replace teachers?": "ai_education",
    "Is remote work better than office work?": "remote_work",
    "Should social media be banned for under 18s?": "social_media_age_ban",
    "Is universal basic income a good idea?": "universal_basic_income",
    "Should college education be free?": "free_college_education",
    "Is nuclear energy the future?": "nuclear_energy",
    "Should voting be mandatory?": "mandatory_voting",
    "Is capitalism the best economic system?": "capitalism"
}

def retrieve(topic, message, n_results=3):
    topic_filter = TOPIC_MAP.get(topic)
    query_embedding = model.encode([message]).tolist()

    where = {"topic": topic_filter} if topic_filter else None

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        where=where
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "score": round(max(0, 1 - results["distances"][0][i]), 3)
        })
    return chunks