"""청크 JSON → ChromaDB 적재."""
import json
from pathlib import Path
from typing import List, Dict, Any

from app.retrieval.chroma_client import get_client, get_collection

CHUNKS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "chunks"
BATCH_SIZE = 100


def _make_id(record: Dict[str, Any], idx: int) -> str:
    return record.get("chunk_id") or f"{record['doc_id']}_{idx}"


def _make_document(record: Dict[str, Any]) -> str:
    """임베딩할 텍스트 생성."""
    t = record.get("type")
    if t == "course":
        parts = [
            record.get("course_name", ""),
            record.get("professor", ""),
            record.get("day", ""),
            record.get("time_range", ""),
            record.get("room", ""),
        ]
        return " ".join(p for p in parts if p)
    if t == "professor":
        name = record.get("name", "")
        day = record.get("day", "")
        phone = record.get("phone", "")
        room = record.get("room", "")
        dept = record.get("dept", "")
        return f"{name} 교수 연구일: {day}, 전화: {phone}, 연구실: {room}, 학과: {dept}"
    if t == "dept_phone":
        dept = record.get("dept", "")
        phone = record.get("phone", "")
        college = record.get("college", "")
        return f"{dept} 전화번호: {phone} ({college})"
    return record.get("text", "")


def _make_metadata(record: Dict[str, Any], category: str) -> Dict[str, Any]:
    """ChromaDB 메타데이터 (str/int/float/bool 만 허용)."""
    base = {
        "category": category,
        "doc_id": record.get("doc_id", ""),
        "title": record.get("title", ""),
        "type": record.get("type", ""),
    }
    t = record.get("type")
    if t == "course":
        base.update({
            "course_number": record.get("course_number", ""),
            "course_name": record.get("course_name", ""),
            "professor": record.get("professor") or "",
            "day": record.get("day") or "",
            "time_range": record.get("time_range") or "",
            "room": record.get("room") or "",
            "course_type": record.get("course_type") or "",
            "year": record.get("year") or 0,
        })
    elif t == "professor":
        base.update({
            "name": record.get("name", ""),
            "dept": record.get("dept", ""),
            "college": record.get("college", ""),
            "day": record.get("day", ""),
            "phone": record.get("phone", ""),
            "room": record.get("room", ""),
        })
    elif t == "dept_phone":
        base.update({
            "dept": record.get("dept", ""),
            "college": record.get("college", ""),
            "phone": record.get("phone", ""),
        })
    else:
        base.update({
            "section": record.get("section", ""),
        })
    return base


def ingest_all():
    client = get_client()
    collection = get_collection(client)

    json_files = list(CHUNKS_DIR.rglob("*.json"))
    print(f"적재할 파일: {len(json_files)}개")

    total = 0
    for json_path in json_files:
        # academic_md 폴더도 category=academic 으로 통일
        folder = json_path.parent.name
        category = "academic" if folder.startswith("academic") else folder
        records: List[Dict[str, Any]] = json.loads(json_path.read_text(encoding="utf-8"))
        print(f"  [{category}] {json_path.name} → {len(records)}개", end=" ")

        ids, docs, metas = [], [], []
        for i, rec in enumerate(records):
            doc_text = _make_document(rec)
            if not doc_text.strip():
                continue
            ids.append(_make_id(rec, i))
            docs.append(doc_text)
            metas.append(_make_metadata(rec, category))

        # 배치 단위로 적재
        for start in range(0, len(ids), BATCH_SIZE):
            collection.upsert(
                ids=ids[start:start + BATCH_SIZE],
                documents=docs[start:start + BATCH_SIZE],
                metadatas=metas[start:start + BATCH_SIZE],
            )

        print(f"완료 ({len(ids)}개 적재)")
        total += len(ids)

    print(f"\n총 {total}개 적재 완료 → {collection.count()}개 저장됨")
