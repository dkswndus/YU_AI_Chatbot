# Neo4j 연동 (그래프 DB)

## 개요

트리플(교수–과목–요일·시간·강의실)을 Neo4j에 넣어 **그래프 조회**로 사용할 수 있습니다.

- **노드**: `Professor`, `Course`
- **관계**: `(Professor)-[:TEACHES {day, time_range, room, title, doc_id}]->(Course)`
- 조회 시 Cypher로 "무도대학 수요일 수업", "천양하 교수님 수업" 등을 그래프에서 조회합니다.

## 1. Neo4j 설치 및 실행

- [Neo4j Desktop](https://neo4j.com/download/) 또는 Docker 예:

  ```bash
  docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/비밀번호 neo4j:latest
  ```

- 브라우저에서 http://localhost:7474 접속 후 로그인 (기본 사용자 `neo4j`, 비밀번호는 설치 시 설정한 값).

## 2. 환경 변수

| 변수 | 설명 | 기본값 |
|------|------|--------|
| NEO4J_URI | Bolt 주소 | bolt://localhost:7687 |
| NEO4J_USER | 사용자명 | neo4j |
| NEO4J_PASSWORD | 비밀번호 | (필수 설정) |

Windows 예:

```cmd
:: 현재 터미널(세션)에만 적용
set NEO4J_URI=bolt://localhost:7687
set NEO4J_USER=neo4j
set NEO4J_PASSWORD=비밀번호

:: 영구 적용(새 터미널부터 적용됨)
setx NEO4J_URI "bolt://localhost:7687"
setx NEO4J_USER "neo4j"
setx NEO4J_PASSWORD "비밀번호"
```

Mac/Linux 예:

```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=비밀번호
```

## 3. 트리플 동기화

온톨로지(teaches.json)를 Neo4j에 넣습니다.

```bash
# 1) 트리플이 없으면 먼저 빌드
python run_build_ontology.py

# 2) Neo4j에 동기화
python run_neo4j_sync.py
```

성공 시 `Neo4j 동기화 완료: N건 TEACHES 관계 반영` 이 출력됩니다.

## 4. 조회

`NEO4J_URI`(및 비밀번호)가 설정되어 있으면 `run_ontology_query.py` 가 **자동으로 Neo4j**를 사용합니다.

```bash
python run_ontology_query.py "무도대학 수요일 수업"
python run_ontology_query.py 천양하
python run_ontology_query.py --course "데이터베이스"
```

출력에 `[Neo4j]` 가 붙으면 그래프 DB에서 조회한 결과입니다.  
Neo4j가 꺼져 있거나 환경 변수가 없으면 기존처럼 `data/ontology/teaches.json` 에서 조회합니다.

## 5. 코드 위치

- `app/ontology/neo4j_graph.py`: 드라이버, 동기화, Cypher 조회
- `run_neo4j_sync.py`: 트리플 → Neo4j 동기화 스크립트
