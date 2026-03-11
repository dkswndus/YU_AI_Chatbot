"""academic 청크에서 교수–과목 트리플 추출 후 data/ontology/teaches.json 에 저장."""
from app.ontology.triples import build_triples_from_chunks, save_triples


def main():
    triples = build_triples_from_chunks()
    path = save_triples(triples)
    print(f"온톨로지 빌드 완료: {len(triples)}개 트리플 → {path}")


if __name__ == "__main__":
    main()
