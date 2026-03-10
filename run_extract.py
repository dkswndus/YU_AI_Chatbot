"""문서 텍스트 추출: PDF 목록 전부 + URL 목록 전부 (카테고리별 모두 포함)."""
import csv
import json
from pathlib import Path

from app.loaders.pdf_loader import extract_text as extract_pdf
from app.loaders.url_loader import extract_text as extract_url

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def load_pdf_list():
    """pdf_list.csv 로드."""
    path = DATA_RAW / "pdf" / "pdf_list.csv"
    rows = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def load_url_list():
    """url_list.csv 로드."""
    path = DATA_RAW / "urls" / "url_list.csv"
    rows = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    pdf_rows = load_pdf_list()
    url_rows = load_url_list()

    # PDF 전부 + URL 전부 (scholarship, faq 등 카테고리 모두 포함)
    to_process = []
    for r in pdf_rows:
        to_process.append(("pdf", r))
    for r in url_rows:
        to_process.append(("url", r))

    for i, (kind, row) in enumerate(to_process, 1):
        doc_id = row["id"]
        title = row["title"]
        category = row["category"]
        source_type = row["source_type"]
        source = row["source"]
        updated_at = row.get("updated_at", "")

        print(f"[{i}/{len(to_process)}] {doc_id} {title[:30]}...")

        try:
            if kind == "pdf":
                # source: raw/pdf/xxx.pdf -> data/raw/pdf/xxx.pdf
                pdf_path = PROJECT_ROOT / "data" / "raw" / source.replace("raw/", "")
                text = extract_pdf(pdf_path)
            else:
                text = extract_url(source)
        except Exception as e:
            print(f"  실패: {e}")
            text = f"[추출 실패] {e}"

        # 저장: {id}.txt, {id}.meta.json
        out_txt = DATA_PROCESSED / f"{doc_id}.txt"
        out_meta = DATA_PROCESSED / f"{doc_id}.meta.json"
        out_txt.write_text(text, encoding="utf-8")
        out_meta.write_text(
            json.dumps(
                {
                    "id": doc_id,
                    "title": title,
                    "category": category,
                    "source_type": source_type,
                    "source": source,
                    "updated_at": updated_at,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  -> {out_txt.name} ({len(text)} chars)")

    print(f"\n완료: {len(to_process)}개 -> {DATA_PROCESSED}")


if __name__ == "__main__":
    main()
