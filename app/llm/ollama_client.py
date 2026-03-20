"""Ollama API 클라이언트"""
import requests

DEFAULT_MODEL    = "exaone3.5:7.8b"
DEFAULT_BASE_URL = "http://localhost:11434"


def chat(messages: list, model: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL) -> str:
    """대화 형식 호출. messages: [{'role': 'user'|'assistant'|'system', 'content': '...'}]"""
    resp = requests.post(
        f"{base_url}/api/chat",
        json={"model": model, "messages": messages, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def is_available(base_url: str = DEFAULT_BASE_URL) -> bool:
    try:
        requests.get(f"{base_url}/api/tags", timeout=3)
        return True
    except Exception:
        return False
