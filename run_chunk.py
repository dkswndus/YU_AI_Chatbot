"""정제본(_clean.txt)을 답변 단위로 청킹해 카테고리별 JSON 저장."""
import json
from pathlib import Path

from app.ingest.chunking import chunk_text

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
CHUNKS_DIR = DATA_PROCESSED / "chunks"


def main():
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    for path in sorted(DATA_PROCESSED.glob("*_clean.txt")):
        # academic_001_clean.txt -> doc_id = academic_001
        doc_id = path.stem.replace("_clean", "")
        meta_path = DATA_PROCESSED / f"{doc_id}.meta.json"

        if not meta_path.exists():
            print(f"  skip {path.name}: no {meta_path.name}")
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        title = meta.get("title", doc_id)
        category = meta.get("category", "unknown")

        text = path.read_text(encoding="utf-8")
        chunks = chunk_text(text, doc_id=doc_id, title=title, category=category)

        # 카테고리별 폴더에 저장
        category_dir = CHUNKS_DIR / category
        category_dir.mkdir(parents=True, exist_ok=True)
        out_path = category_dir / f"{doc_id}.json"
        out_path.write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"{doc_id} ({category}): {len(chunks)} chunks -> {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
