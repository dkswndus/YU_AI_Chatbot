"""온톨로지에서 교수·과목·요일·대학 등 관계 조합으로 수업 조회. Neo4j 사용 시 그래프 DB에서 조회."""
import argparse
import sys

from app.ontology.triples import (
    get_courses_by_teacher,
    get_schedule_by_course,
    get_schedule_by_filters,
    parse_schedule_query,
)
from app.ontology.neo4j_graph import (
    is_neo4j_available,
    get_courses_by_teacher_neo4j,
    get_schedule_by_course_neo4j,
    query_schedule_from_neo4j,
)


def _print_schedule_results(results, title_label):
    """조회 결과 공통 출력 (교수, 과목, 요일, 시간, 강의실, 학기)."""
    if not results:
        return
    print("=" * 60)
    for i, r in enumerate(results, 1):
        teacher = r.get("teacher_name", "")
        course = r.get("course_name", "")
        day = r.get("day", "")
        time_range = r.get("time_range", "")
        room = r.get("room", "")
        doc_title = r.get("title", "")
        parts = [f"교수: {teacher}", course]
        if day or time_range:
            parts.append(f"{day} {time_range}".strip())
        if room:
            parts.append(room)
        if doc_title:
            parts.append(f"({doc_title})")
        line = " / ".join(p for p in parts if p)
        print(f"[{i}] {line}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="온톨로지 조회: 교수명, 과목명, 요일+대학(무도대학 수요일 수업 등) 조합"
    )
    parser.add_argument("query", nargs="*", help="교수명 또는 '무도대학 수요일 수업' 같은 질의")
    parser.add_argument("--course", "-C", type=str, default=None, help="과목명으로 조회")
    parser.add_argument("--day", "-d", type=str, default=None, help="요일 (월/화/수/목/금)")
    parser.add_argument("--college", "-g", type=str, default=None, help="대학명 (예: 무도대학, 용오름대학)")
    args = parser.parse_args()

    raw = " ".join(args.query).strip() if args.query else ""

    # 1) 과목명 조회
    if args.course:
        course_name = args.course.strip()
        if not course_name:
            print("--course 뒤에 과목명을 입력하세요.")
            sys.exit(1)
        results = get_schedule_by_course_neo4j(course_name) if is_neo4j_available() else get_schedule_by_course(course_name)
        if not results:
            print(f"'{course_name}' 과목을 찾지 못했습니다.")
            sys.exit(0)
        src = " [Neo4j]" if is_neo4j_available() else ""
        print(f"\n[온톨로지 조회{src}] 과목 '{course_name}' 수업 일정 ({len(results)}건)\n")
        _print_schedule_results(results, None)
        print("조회 끝.")
        return

    # 2) 요일·대학 조합 (--day / --college 또는 "무도대학 수요일 수업" 같은 한 줄 질의)
    day = args.day
    college = args.college
    if not day and not college and raw:
        parsed = parse_schedule_query(raw)
        day = parsed.get("day") or day
        college = parsed.get("college") or college

    if day is not None or college is not None:
        if is_neo4j_available():
            results = query_schedule_from_neo4j(day=day or None, college=college or None)
        else:
            results = get_schedule_by_filters(day=day or None, college=college or None)
        if not results:
            print(f"조건에 맞는 수업이 없습니다. (요일: {day or '전체'}, 대학: {college or '전체'})")
            sys.exit(0)
        label = []
        if college:
            label.append(college)
        if day:
            label.append(f"{day}요일")
        label = " ".join(label) or "수업"
        src = " [Neo4j]" if is_neo4j_available() else ""
        print(f"\n[온톨로지 조회{src}] {label} 수업 ({len(results)}건)\n")
        _print_schedule_results(results, None)
        print("조회 끝.")
        return

    # 3) 교수명 조회
    if raw:
        teacher_name = raw
    else:
        teacher_name = input("교수명을 입력하세요: ").strip()

    if not teacher_name:
        print("교수명이 비어 있습니다.")
        sys.exit(0)

    results = get_courses_by_teacher_neo4j(teacher_name) if is_neo4j_available() else get_courses_by_teacher(teacher_name)
    if not results:
        print(f"'{teacher_name}' 교수님 수업을 찾지 못했습니다.")
        print("run_build_ontology.py 를 먼저 실행했는지, 교수명이 정확한지 확인하세요.")
        sys.exit(0)

    src = " [Neo4j]" if is_neo4j_available() else ""
    print(f"\n[온톨로지 조회{src}] {teacher_name} 교수님 수업 ({len(results)}건)\n")
    print("=" * 60)
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        course = r.get("course_name", "")
        day = r.get("day", "")
        time_range = r.get("time_range", "")
        room = r.get("room", "")
        parts = [course]
        if day or time_range:
            parts.append(f"{day} {time_range}".strip())
        if room:
            parts.append(room)
        if title:
            parts.append(f"({title})")
        line = " / ".join(p for p in parts if p)
        print(f"[{i}] {line}")
    print("=" * 60)
    print("조회 끝.")


if __name__ == "__main__":
    main()
