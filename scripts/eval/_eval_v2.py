"""
yongin_univ_questions_1000_natural_v2.json 기반
베이스라인(ChromaDB raw) / MD청크(ChromaDB) / Neo4j 정확도 비교
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.chroma_client import get_client, get_collection
from app.retrieval.chroma_search import search
from neo4j import GraphDatabase

Q_FILE     = PROJECT_ROOT / "data" / "yongin_univ_questions_1000_natural_v2.json"
CHUNK_FILE = PROJECT_ROOT / "data" / "processed" / "chunks" / "academic" / "academic_001.json"
OUT_FILE   = PROJECT_ROOT / "_eval_v2_result.json"

NEO4J_URI  = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "yu_chatbot_2026"
N_RESULTS  = 10


def load_keywords():
    data = json.loads(CHUNK_FILE.read_text(encoding="utf-8"))
    professors = set(r["professor"] for r in data if r.get("professor"))
    courses    = set(r["course_name"] for r in data if r.get("course_name"))
    return professors, courses


def extract_keywords(q, professors, courses):
    found_profs   = [p for p in professors if p in q]
    found_courses = [c for c in courses if c in q]
    return found_profs, found_courses


def eval_chroma(questions, professors, courses, label, doc_id_filter=None):
    """ChromaDB 정확도 평가. doc_id_filter='academic_001' 이면 베이스라인(raw course만)"""
    stats = {"prof": {"t": 0, "h": 0}, "course": {"t": 0, "h": 0},
             "both": {"t": 0, "h": 0}, "none": 0}

    # 베이스라인용: raw course 레코드만 있는 별도 컬렉션 생성 후 평가
    if doc_id_filter:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        tmp_client = chromadb.EphemeralClient()
        ef = SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
        tmp_col = tmp_client.create_collection("baseline_tmp", embedding_function=ef)

        raw = json.loads(CHUNK_FILE.read_text(encoding="utf-8"))
        ids, docs, metas = [], [], []
        for i, r in enumerate(raw):
            t = r.get("type")
            if t == "course":
                doc = " ".join(filter(None, [r.get("course_name",""), r.get("professor",""),
                                             r.get("day",""), r.get("time_range",""), r.get("room","")]))
            else:
                doc = r.get("text", "")
            if not doc.strip(): continue
            ids.append(f"{r.get('doc_id','x')}_{i}")
            docs.append(doc)
            metas.append({"type": t or "", "professor": r.get("professor","") or "",
                          "course_name": r.get("course_name","") or "",
                          "doc_id": r.get("doc_id","")})
        for s in range(0, len(ids), 100):
            tmp_col.upsert(ids=ids[s:s+100], documents=docs[s:s+100], metadatas=metas[s:s+100])

        def query_fn(q):
            res = tmp_col.query(query_texts=[q], n_results=N_RESULTS)
            return [{"metadata": m} for m in res["metadatas"][0]]
    else:
        def query_fn(q):
            return search(q, n_results=N_RESULTS)

    print(f"  [{label}] 평가 중...")
    for item in questions:
        q = item.get("question", item) if isinstance(item, dict) else item
        fp, fc = extract_keywords(q, professors, courses)
        hp, hc = len(fp) > 0, len(fc) > 0
        if not hp and not hc:
            stats["none"] += 1
            continue
        results = query_fn(q)
        hit = False
        for r in results:
            m = r["metadata"]
            rp = m.get("professor", "") or m.get("name", "")
            rc = m.get("course_name", "")
            if hp and any(p in rp for p in fp): hit = True; break
            if hc and any(c in rc for c in fc): hit = True; break
        key = "both" if (hp and hc) else ("prof" if hp else "course")
        stats[key]["t"] += 1
        if hit: stats[key]["h"] += 1

    def pct(h, t): return f"{h/t*100:.1f}%" if t else "N/A"
    return {
        "교수명 포함":  {"total": stats["prof"]["t"],  "hit": stats["prof"]["h"],  "accuracy": pct(stats["prof"]["h"],  stats["prof"]["t"])},
        "과목명 포함":  {"total": stats["course"]["t"], "hit": stats["course"]["h"], "accuracy": pct(stats["course"]["h"], stats["course"]["t"])},
        "교수+과목 포함":{"total": stats["both"]["t"],  "hit": stats["both"]["h"],  "accuracy": pct(stats["both"]["h"],  stats["both"]["t"])},
        "키워드 없음":  stats["none"],
    }


def eval_neo4j(questions, professors, courses):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    stats = {"prof": {"t": 0, "h": 0}, "course": {"t": 0, "h": 0},
             "both": {"t": 0, "h": 0}, "none": 0}
    print("  [Neo4j] 평가 중...")
    with driver.session() as session:
        for item in questions:
            q = item.get("question", item) if isinstance(item, dict) else item
            fp, fc = extract_keywords(q, professors, courses)
            hp, hc = len(fp) > 0, len(fc) > 0
            if not hp and not hc:
                stats["none"] += 1
                continue
            hit = False
            if hp:
                for pname in fp:
                    if session.run("MATCH (p:Professor {name:$n}) RETURN p LIMIT 1", n=pname).single():
                        hit = True; break
            if not hit and hc:
                for cname in fc:
                    if session.run("MATCH (c:Course {course_name:$n}) RETURN c LIMIT 1", n=cname).single():
                        hit = True; break
            key = "both" if (hp and hc) else ("prof" if hp else "course")
            stats[key]["t"] += 1
            if hit: stats[key]["h"] += 1
    driver.close()
    def pct(h, t): return f"{h/t*100:.1f}%" if t else "N/A"
    return {
        "교수명 포함":  {"total": stats["prof"]["t"],  "hit": stats["prof"]["h"],  "accuracy": pct(stats["prof"]["h"],  stats["prof"]["t"])},
        "과목명 포함":  {"total": stats["course"]["t"], "hit": stats["course"]["h"], "accuracy": pct(stats["course"]["h"], stats["course"]["t"])},
        "교수+과목 포함":{"total": stats["both"]["t"],  "hit": stats["both"]["h"],  "accuracy": pct(stats["both"]["h"],  stats["both"]["t"])},
        "키워드 없음":  stats["none"],
    }


def main():
    questions  = json.loads(Q_FILE.read_text(encoding="utf-8"))
    professors, courses = load_keywords()
    print(f"질문: {len(questions)}개 | 교수: {len(professors)}개 | 과목: {len(courses)}개")

    r_base  = eval_chroma(questions, professors, courses, "베이스라인", doc_id_filter="academic_001")
    r_md    = eval_chroma(questions, professors, courses, "MD 청크")
    r_neo4j = eval_neo4j(questions, professors, courses)

    result = {"베이스라인": r_base, "MD 청크": r_md, "Neo4j": r_neo4j}
    OUT_FILE.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 결과 ===")
    for name, r in result.items():
        print(f"\n[{name}]")
        for k, v in r.items():
            if isinstance(v, dict):
                print(f"  {k}: {v['hit']}/{v['total']} = {v['accuracy']}")
            else:
                print(f"  {k}: {v}개")


if __name__ == "__main__":
    main()
