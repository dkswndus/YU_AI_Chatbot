"""teaches.json 트리플을 Neo4j에 동기화. (실행 전 Neo4j 서버 실행 및 NEO4J_URI 등 설정)"""
import os
import sys
import getpass

from app.ontology.triples import load_triples
from app.ontology.neo4j_graph import get_driver, sync_triples_to_neo4j


def main():
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "")

    if not password:
        print("NEO4J_PASSWORD 환경변수가 비어 있습니다.")
        print("이 PowerShell 세션에서 임시로 설정하거나, 여기서 비밀번호를 입력하세요.")
        # 안전하게 입력(화면에 안 보임)
        password = getpass.getpass("Neo4j 비밀번호 입력(입력 내용 숨김): ").strip()
        if not password:
            print("비밀번호가 비어 있어 종료합니다.")
            sys.exit(1)

    triples = load_triples()
    if not triples:
        print("트리플이 없습니다. 먼저 run_build_ontology.py 를 실행하세요.")
        sys.exit(1)

    try:
        driver = get_driver(uri=uri, user=user, password=password)
        driver.verify_connectivity()
        n = sync_triples_to_neo4j(triples=triples, driver=driver)
        driver.close()
        print(f"Neo4j 동기화 완료: {n}건 TEACHES 관계 반영 (URI: {uri})")
    except Exception as e:
        print(f"Neo4j 연결 또는 동기화 실패: {e}")
        print("Neo4j 서버가 실행 중인지, URI/사용자/비밀번호가 맞는지 확인하세요.")
        print(f"- NEO4J_URI: {uri}")
        print(f"- NEO4J_USER: {user}")
        sys.exit(1)


if __name__ == "__main__":
    main()
