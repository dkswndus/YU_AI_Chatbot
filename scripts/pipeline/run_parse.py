"""txt → JSON 파싱/청킹. data/processed/chunks/{category}/{doc_id}.json 으로 저장."""
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.ingest.timetable_parser import parse_timetable

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def main():
    csv_path = DATA_RAW / "pdf" / "pdf_list.csv"
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        doc_id = row["id"]
        title = row["title"]
        category = row["category"]
        semester = row.get("updated_at", "")

        txt_path = DATA_PROCESSED / f"{doc_id}.txt"
        if not txt_path.exists():
            print(f"[SKIP] {doc_id}.txt 없음 - run_extract.py 먼저 실행하세요.")
            continue

        text = txt_path.read_text(encoding="utf-8")

        if category == "academic":
            records = parse_timetable(text, doc_id=doc_id, title=title, semester=semester)
        else:
            print(f"[SKIP] 알 수 없는 category: {category}")
            continue

        out_dir = DATA_PROCESSED / "chunks" / category
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{doc_id}.json"
        out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{category}] {doc_id} → {len(records)}개 레코드 → {out_path}")

    print("\n파싱 완료.")


if __name__ == "__main__":
    main()
