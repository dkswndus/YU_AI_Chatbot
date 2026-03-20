"""검색 테스트 - 쿼리 입력 후 상위 결과 출력."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.chroma_search import search


def main():
    query = input("검색어: ").strip()
    if not query:
        print("검색어를 입력하세요.")
        return

    results = search(query, n_results=10)
    print(f"\n=== 상위 {len(results)}개 결과 ===\n")
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        t = meta.get("type", "")
        dist = r["distance"]

        print(f"[{i}] type={t}  distance={dist:.4f}")
        if t == "course":
            print(f"     {meta.get('course_name','')} / {meta.get('professor','')} / {meta.get('day','')} {meta.get('time_range','')} / {meta.get('room','')}")
        else:
            print(f"     section={meta.get('section','')}")
            print(f"     {r['text'][:150]}")
        print()


if __name__ == "__main__":
    main()
