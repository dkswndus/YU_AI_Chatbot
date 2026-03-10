"""추출 텍스트 정제: 불필요한 줄 제거, 줄바꿈 정리."""
import re

# 한 줄 전체가 이 문자열이면 제거 (URL/페이지 공통 잡음)
JUNK_LINES = frozenset({
    "뒤로가기", "맨 위로", "공유하기", "kakao", "facebook", "URL copy", "인쇄",
    "이메일 무단수집거부", "창닫기", "닫기", "TOP", "취소", "등록",
    "홈", "대학생활", "용인스토리", "학교소개", "대학 · 대학원", "연구 · 교육",
    "학사지원", "입학안내", "장학금", "기숙사 입소", "보험", "출입국 신고",
    "수강신청", "휴학/복학", "졸업 요건", "인터넷 증명발급",
    "상담종류", "폭행", "성폭행", "이름", "연락처", "이메일", "내용",
})

# 이 문자열을 포함한 줄 제거 (긴 고정 문구·푸터)
JUNK_CONTAINS = [
    "용인대학교 홈페이지는 이메일 주소 무단수집",
    "정보통신망법에 의해 처벌",
    "저작권법에 의해서 보호",
    "학생인권지킴이",
    "본 사이트는 학생 여러분께서 학교생활을 하며 발생되는 선후배 간의 폭력",
    "상담전화",
    "010-9014-3119",
    "010-5420-5112",
    "이메일 또는 연락처를 남기지 않을 경우 답변이 불가",
    "개인정보 수집 동의",
    "「개인정보보호 법」",
    "개인정보 수집 , 이용에 대한 동의",
    "개인정보 수집 , 이용 목적과 제한",
    "개인정보의 보유기간 및 이용기간",
    "개인정보의 수집 항목",
    "전화번호(휴대폰), 이메일 등 본페이지에서 입력",
    "이용약관에 동의합니다",
    "인터넷 증명서 발급 안내",
    "한국정보인증(주) 바로가기",
    "해외 우편 증명서",
    "개인정보처리방침 또는 이용약관",
    "본교는 귀하께서",
    "본교는 본인식별 및 본인의사 확인",
    "귀하의 개인정보는 수집목적 또는 제공받은 목적이 달성되면 파기",
    "- 폭행관련",
    "- 성폭력관련",
]

# 패턴으로 제거: 페이지 번호 "- 1 -", "- 154 -"
PAGE_NUMBER_RE = re.compile(r"^\s*-\s*\d+\s*-\s*$")
# 목차 점선만 있는 줄 ·········· 1
TOC_DOTS_RE = re.compile(r"^[\s·\.\-]+\d*\s*$")


def _is_junk_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if s in JUNK_LINES:
        return True
    if PAGE_NUMBER_RE.match(s) or TOC_DOTS_RE.match(s):
        return True
    for sub in JUNK_CONTAINS:
        if sub in s:
            return True
    return False


def clean_text(text: str) -> str:
    """불필요한 줄 제거 후 줄바꿈 정리."""
    lines = []
    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if _is_junk_line(line):
            continue
        lines.append(line)
    # 빈 줄 연속을 최대 1개로
    out = []
    prev_blank = False
    for line in lines:
        is_blank = line == ""
        if is_blank and prev_blank:
            continue
        out.append(line)
        prev_blank = is_blank
    # 앞뒤 빈 줄 제거
    while out and out[0] == "":
        out.pop(0)
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out)
