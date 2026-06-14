import os
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from groq import Groq
from retriever import retrieve

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


class Message(BaseModel):
    role: str
    content: str


class DebateRequest(BaseModel):
    topic: str
    message: str
    stance: str = "pro"
    history: List[Message] = []
    mode: str = "debate"


def get_counterargument(topic, message, stance, context, history):
    stance_instruction = {
        "pro": "The user is arguing FOR the topic. You must argue AGAINST it.",
        "against": "The user is arguing AGAINST the topic. You must argue FOR it.",
        "neutral": "Challenge the user's argument from whichever angle exposes the weakest point."
    }.get(stance, "Challenge the user's argument.")

    history_text = ""
    if history:
        history_text = "\n\nConversation so far:\n"
        for msg in history[-6:]:
            label = "User" if msg.role == "user" else "You"
            history_text += f"{label}: {msg.content}\n"

    prompt = f"""You are a sharp, rigorous debate opponent. The debate topic is: "{topic}"

{stance_instruction}

Retrieved evidence from knowledge base:
{context}
{history_text}

The user just said: "{message}"

Respond with ONE strong counterargument, grounded in the evidence above.
Reference previous points in the conversation if relevant.
3-5 sentences, no bullet points, direct and assertive."""

    res = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return res.choices[0].message.content


def get_fallacy(message):
    prompt = f"""Analyze this debate argument for logical fallacies.

Argument: "{message}"

Respond ONLY with a JSON object, no markdown:
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
    prompt = f"""You are a strict debate judge. Score this argument on four criteria, each out of 10.

Argument: "{message}"

Scoring rules:
LOGIC (1-10):
- 1-3: Conclusion does not follow from premise, circular reasoning, or non-sequitur
- 4-6: Some logical structure but contains unsupported leaps or weak reasoning
- 7-9: Clear logical flow, premise supports conclusion well
- 10: Airtight reasoning, no logical gaps

EVIDENCE (1-10):
- 1-3: Pure opinion or feeling, no facts, no examples, no data
- 4-6: Some general examples but no specific studies, stats, or citations
- 7-9: Specific facts, real examples, or named studies referenced
- 10: Multiple concrete statistics or peer-reviewed evidence cited

CLARITY (1-10):
- 1-3: Unclear, rambling, or hard to follow
- 4-6: Understandable but could be more precise
- 7-9: Clear, concise, well-structured
- 10: Perfectly articulated, no ambiguity

PERSUASIVENESS (1-10):
- 1-3: Would not convince anyone, emotionally weak, no rhetorical force
- 4-6: Somewhat convincing but lacks impact
- 7-9: Compelling, addresses the audience well
- 10: Highly persuasive, strong rhetorical force

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


def get_coaching(topic, message, counterargument, context):
    prompt = f"""You are a debate coach reviewing a student's argument. Be honest and direct.

Debate topic: "{topic}"
Student's argument: "{message}"
Opponent's counterargument: "{counterargument}"
Retrieved evidence available: {context}

Give coaching feedback in this exact JSON format, no markdown:
{{
  "did_well": ["point 1", "point 2"],
  "weaknesses": ["point 1", "point 2"],
  "sharper_version": "rewrite the student's argument in 2-3 sentences using better logic and the available evidence"
}}"""

    res = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400
    )
    text = res.choices[0].message.content.strip()
    try:
        return json.loads(text)
    except:
        return {"did_well": [], "weaknesses": [], "sharper_version": ""}


@app.get("/")
def root():
    return FileResponse("index.html")


@app.post("/debate")
def debate(req: DebateRequest):
    chunks = retrieve(req.topic, req.message, n_results=3)

    context = "\n\n".join([
        f"[Source: {c['source']}]\n{c['text']}"
        for c in chunks
    ])

    counterargument = get_counterargument(req.topic, req.message, req.stance, context, req.history)
    fallacy = get_fallacy(req.message)
    scores = get_scores(req.message)

    coaching = None
    if req.mode == "practice":
        coaching = get_coaching(req.topic, req.message, counterargument, context)

    return {
        "response": counterargument,
        "sources": chunks,
        "fallacy": fallacy,
        "scores": scores,
        "coaching": coaching
    }