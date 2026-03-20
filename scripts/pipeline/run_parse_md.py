"""academic_001.md → JSON 청크 파일 생성"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.ingest.markdown_chunker import chunk_markdown

MD_PATH  = PROJECT_ROOT / "data" / "processed" / "academic_001.md"
OUT_DIR  = PROJECT_ROOT / "data" / "processed" / "chunks" / "academic_md"
OUT_FILE = OUT_DIR / "academic_001_md.json"

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = chunk_markdown(
        md_path=MD_PATH,
        doc_id="academic_001_md",
        title="2026 1학기 종합강의시간표",
        category="academic",
        semester="2026-01-01",
    )
    OUT_FILE.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    by_type = {}
    for r in records:
        t = r.get("type", "?")
        by_type[t] = by_type.get(t, 0) + 1

    print(f"총 {len(records)}개 레코드")
    for t, cnt in sorted(by_type.items()):
        print(f"  {t}: {cnt}개")
    print(f"저장: {OUT_FILE}")

if __name__ == "__main__":
    main()
