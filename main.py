import os
import json
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
    stance: str = "pro"

def get_counterargument(topic, message, stance, context):
    stance_instruction = {
        "pro": "The user is arguing FOR the topic. You must argue AGAINST it.",
        "against": "The user is arguing AGAINST the topic. You must argue FOR it.",
        "neutral": "Challenge the user's argument from whichever angle exposes the weakest point."
    }.get(stance, "Challenge the user's argument.")

    prompt = f"""You are a sharp, rigorous debate opponent. The debate topic is: "{topic}"

{stance_instruction}

Retrieved evidence from knowledge base:
{context}

The user just said: "{message}"

Respond with ONE strong counterargument, grounded in the evidence above. 3-5 sentences, no bullet points, direct and assertive."""

    res = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return res.choices[0].message.content

def get_fallacy(message):
    prompt = f"""Analyze this debate argument for logical fallacies.

Argument: "{message}"

Respond ONLY with a JSON object, no markdown, no explanation outside the JSON:
{{"fallacy": "name of fallacy or null if none", "explanation": "one sentence explanation or null"}}"""

    res = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )
    text = res.choices[0].message.content.strip()
    try:
        return json.loads(text)
    except:
        return {"fallacy": None, "explanation": None}

def get_scores(message):
    prompt = f"""Score this debate argument on four metrics, each out of 10.

Argument: "{message}"

Respond ONLY with a JSON object, no markdown:
{{"logic": 0, "evidence": 0, "clarity": 0, "persuasiveness": 0}}"""

    res = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=60
    )
    text = res.choices[0].message.content.strip()
    try:
        return json.loads(text)
    except:
        return {"logic": 0, "evidence": 0, "clarity": 0, "persuasiveness": 0}

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

    counterargument = get_counterargument(req.topic, req.message, req.stance, context)
    fallacy = get_fallacy(req.message)
    scores = get_scores(req.message)

    return {
        "response": counterargument,
        "sources": chunks,
        "fallacy": fallacy,
        "scores": scores
    }