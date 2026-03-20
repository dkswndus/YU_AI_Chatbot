"""PDF → txt 추출. data/processed/{id}.txt 로 저장."""
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.loaders.pdf_loader import extract_text

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    csv_path = DATA_RAW / "pdf" / "pdf_list.csv"
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        doc_id = row["id"]
        pdf_path = PROJECT_ROOT / "data" / "raw" / row["source"].replace("raw/", "")
        out_path = DATA_PROCESSED / f"{doc_id}.txt"

        print(f"추출 중: {pdf_path.name} ...", end=" ")
        text = extract_text(pdf_path)
        out_path.write_text(text, encoding="utf-8")
        print(f"완료 ({len(text)}자) → {out_path.name}")

    print(f"\n총 {len(rows)}개 추출 완료 → {DATA_PROCESSED}")


if __name__ == "__main__":
    main()
