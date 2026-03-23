"""터미널 챗봇 실행"""
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.llm.ollama_client import is_available
from app.rag.pipeline import answer

def main():
    print("=" * 50)
    print("  용인대학교 AI 챗봇 (EXAONE 3.5)")
    print("  종료: 'exit' 또는 Ctrl+C")
    print("=" * 50)

    if not is_available():
        print("[오류] Ollama가 실행되지 않고 있습니다.")
        print("  터미널에서 'ollama serve' 실행 후 다시 시도하세요.")
        return

    print("챗봇 준비 완료. 질문을 입력하세요.\n")
    history = []

    while True:
        try:
            question = input("학생: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "종료"):
            print("종료합니다.")
            break

        print("챗봇: ", end="", flush=True)
        try:
            reply = answer(question, history=history)
            print(reply)
            print()
            # 대화 이력 유지 (최근 6턴)
            history.append({"role": "user",      "content": question})
            history.append({"role": "assistant", "content": reply})
            if len(history) > 12:
                history = history[-12:]
        except Exception as e:
            print(f"[오류] {e}")

if __name__ == "__main__":
    main()
