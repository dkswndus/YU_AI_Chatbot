"""질문 입력 → ChromaDB 검색 → 상위 결과 확인 (검색 테스트)."""
import argparse
import sys

from app.retrieval.chroma_search import search


def print_result(rank, item, text_max_len=400):
    """한 건 결과를 보기 좋게 출력."""
    meta = item.get("metadata") or {}
    teacher_name = meta.get("teacher_name", "")
    course_name = meta.get("course_name", "")
    day = meta.get("day", "")
    time_range = meta.get("time_range", "")
    room = meta.get("room", "")
    structured_summary = ""
    if any([teacher_name, course_name, day, time_range, room]):
        parts = []
        if course_name:
            parts.append(course_name)
        if teacher_name:
            parts.append(teacher_name)
        if day or time_range:
            parts.append(f"{day} {time_range}".strip())
        if room:
            parts.append(room)
        structured_summary = " / ".join([p for p in parts if p])
    text = (item.get("text") or "").strip()
    if text_max_len and len(text) > text_max_len:
        text = text[:text_max_len] + "\n... (이하 생략)"
    distance = item.get("distance")
    dist_str = f" (거리: {distance:.4f})" if distance is not None else ""

    print(f"\n{'='*60}")
    print(f"[{rank}] {item.get('chunk_id', '')}{dist_str}")
    print(f"    category: {meta.get('category', '')}  |  title: {meta.get('title', '')}")
    print(f"    section:  {meta.get('section', '')}")
    if structured_summary:
        print(f"    summary:  {structured_summary}")
    print("-" * 60)
    print(text)
    print()


def main():
    parser = argparse.ArgumentParser(description="ChromaDB 검색 테스트 (질문 입력 → 상위 결과 확인)")
    parser.add_argument("query", nargs="*", help="질문 (미입력 시 대화형으로 입력)")
    parser.add_argument("-n", "--top", type=int, default=5, help="상위 N개 결과 (기본 5)")
    parser.add_argument("-c", "--category", type=str, default=None,
                        help="카테고리 한정: scholarship, academic, faq, calendar")
    parser.add_argument("--full", action="store_true", help="본문 전체 출력 (생략 없음)")
    args = parser.parse_args()

    if args.query:
        query = " ".join(args.query)
    else:
        print("ChromaDB 검색 테스트 (종료: 빈 줄 입력 또는 Ctrl+C)\n")
        query = input("질문을 입력하세요: ").strip()
        if not query:
            print("질문이 비어 있어 종료합니다.")
            sys.exit(0)

    text_max_len = None if args.full else 400
    n_results = max(1, min(args.top, 20))

    print(f"\n검색: \"{query}\"  (상위 {n_results}개)")
    if args.category:
        print(f"카테고리 한정: {args.category}")

    try:
        results = search(query, n_results=n_results, category=args.category)
    except Exception as e:
        print(f"검색 오류: {e}")
        print("ChromaDB가 아직 적재되지 않았을 수 있습니다. run_ingest_chroma.py 를 먼저 실행하세요.")
        sys.exit(1)

    if not results:
        print("\n검색 결과가 없습니다.")
        sys.exit(0)

    for rank, item in enumerate(results, 1):
        print_result(rank, item, text_max_len=text_max_len)

    print("=" * 60)
    print("검색 테스트 끝.")


if __name__ == "__main__":
    main()
