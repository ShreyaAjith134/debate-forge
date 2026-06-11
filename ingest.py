import os
import re
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

CORPUS_DIR = "corpus"
CHUNK_SIZE = 60
CHUNK_OVERLAP = 15

def load_corpus():
    docs = []
    for filename in os.listdir(CORPUS_DIR):
        if filename.endswith(".txt"):
            topic = filename.replace(".txt", "")
            with open(os.path.join(CORPUS_DIR, filename), "r", encoding="utf-8") as f:
                text = f.read()
            paragraphs = [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]
            for i, para in enumerate(paragraphs):
                docs.append({
                    "text": para,
                    "topic": topic,
                    "chunk_id": f"{topic}_{i}"
                })
    return docs

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk = " ".join(words[start:start+size])
        chunks.append(chunk)
        start += size - overlap
    return chunks

def ingest():
    print("Loading corpus...")
    docs = load_corpus()

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Setting up ChromaDB...")
    client = chromadb.PersistentClient(path="./chroma_db")
    
    try:
        client.delete_collection("debate_corpus")
    except:
        pass
    
    collection = client.create_collection("debate_corpus")

    all_chunks = []
    all_ids = []
    all_metadata = []

    for doc in docs:
        chunks = chunk_text(doc["text"])
        for j, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{doc['chunk_id']}_{j}")
            all_metadata.append({
                "topic": doc["topic"],
                "source": doc["chunk_id"]
            })

    print(f"Embedding {len(all_chunks)} chunks...")
    embeddings = model.encode(all_chunks).tolist()

    collection.add(
        documents=all_chunks,
        embeddings=embeddings,
        ids=all_ids,
        metadatas=all_metadata
    )

    print(f"Done. {len(all_chunks)} chunks stored in ChromaDB.")

if __name__ == "__main__":
    ingest()