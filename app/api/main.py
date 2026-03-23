"""FastAPI 백엔드 - 용인대학교 AI 챗봇"""
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional

from app.rag.pipeline import answer
from app.llm.ollama_client import is_available

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="용인대학교 AI 챗봇", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    history: Optional[List[Message]] = []


class ChatResponse(BaseModel):
    answer: str


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/YU_AI_CHATBOT")
def health():
    return {"status": "ok", "ollama": is_available()}


@app.post("/YU_AI_CHATBOT/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    history = [m.model_dump() for m in req.history] if req.history else []
    reply = answer(req.question, history=history)
    return ChatResponse(answer=reply)
