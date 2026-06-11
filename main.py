import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from retriever import retrieve
from groq import Groq

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

class DebateRequest(BaseModel):
    topic: str
    message: str

@app.get("/")
def root():
    return FileResponse("index.html")

@app.post("/debate")
def debate(req: DebateRequest):
    chunks = retrieve(req.message, n_results=3)
    
    context = "\n\n".join([
        f"[Source: {c['source']}]\n{c['text']}" 
        for c in chunks
    ])

    prompt = f"""You are a sharp, rigorous debate opponent. The debate topic is: "{req.topic}"

You have been given the following retrieved evidence from a knowledge base:
{context}

The user just said: "{req.message}"

Your job:
- Respond with ONE strong counterargument
- Ground your response in the retrieved evidence above
- Be concise (3-5 sentences max)
- Be direct and assertive, like a real debate opponent
- Do not use bullet points, just flowing argument

Respond now:"""

    chat = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    response_text = chat.choices[0].message.content

    return {
        "response": response_text,
        "sources": chunks
    }