"""협력학습 세션 파이프라인.

3단계 구조: 학습자 분석 → 교수자 의사결정 → AI 학생 발화.
전역 변수를 쓰지 않고 config/prompts/learner_models/api 를 명시적으로 주입 받는다.

확장 기능:
- AI 학생 3명 지원 (ai_1 민준·상, ai_2 서연·중, ai_3 연우·하)
- 침묵 유도: 사용자가 60초 이상 말하지 않으면 nudge_on_silence()로 한 명의 AI가 선제 발화
- 사용자 교수자 모드: 사용자가 설명자 모드가 되면 AI는 학습자 모드로 전환해 짧게 반응
"""

import functools as _ft
import re
import time
import sys as _sys
# v1.37: Colab thread stdout buffering 회피 — 모든 print를 자동 flush
_orig_print = print
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    _orig_print(*args, **kwargs)

import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from .llm_api import (
    extract_json, render_prompt,
    DEFAULT_HAIKU, DEFAULT_SONNET,
    DEFAULT_GEMINI_FLASH, DEFAULT_GEMINI_FLASH_LITE, DEFAULT_GEMINI_25,
    DEFAULT_OPENAI_MINI, DEFAULT_OPENAI_FULL,
)

# 프롬프트에 삽입할 도메인 지식 최대 문자 수 (전체를 통째로 넣지 않고 상단만)
DOMAIN_TRUNCATE_CHARS = 2500


# 침묵 유도 임계치 (초)
SILENCE_THRESHOLD_SECONDS = 60
SILENCE_ESCALATION_SECONDS = 120

# 사용자 설명자 모드 감지 기준
TEACHER_MODE_MIN_SENTENCES = 3
TEACHER_MODE_KEYWORDS = ("때문에", "왜냐하면", "이유는", "즉", "따라서", "약수", "정의", "만약", "반대로")


def _sentence_count(text: str) -> int:
    """대략적인 문장 수 계산 (., !, ?, 줄바꿈 기준)."""
    import re
    if not text:
        return 0
    chunks = re.split(r"[\.!?\n]+", text.strip())
    return len([c for c in chunks if c.strip()])


def sanitize_ai_output(text: str) -> str:
    """AI 발화 응답에서 메타-추론/헤더 프리픽스를 제거."""
    import re

    if not text:
        return ""

    t = text.strip()

    if t.startswith("```") and t.endswith("```"):
        inner = t[3:-3].strip()
        first_nl = inner.find("\n")
        if first_nl != -1 and " " not in inner[:first_nl] and "=" not in inner[:first_nl]:
            inner = inner[first_nl+1:].strip()
        t = inner

    label_pat = re.compile(
        r"^\s*(?:발화|발\s*화|Utterance|utterance|대사|응답|output|Output|OUTPUT|final|Final)\s*[:：]?\s*$",
        re.MULTILINE,
    )
    matches = list(label_pat.finditer(t))
    if matches:
        t = t[matches[-1].end():].strip()

    student_names = ("민준", "서연", "연우")
    kept = []
    for ln in t.split("\n"):
        stripped = ln.strip()
        if not stripped:
            kept.append(ln)
            continue
        if re.match(r"^#{1,6}\s", stripped):
            continue
        if stripped in student_names:
            continue
        m = re.match(r"^[\*\-\u2022]\s+\**([^:：\n]{1,25})[:：]", stripped)
        if m:
            continue
        if re.match(r"^\**(현재\s*상황\s*분석|분석|reasoning|추론|상황)\**\s*[:：]\s*$", stripped):
            continue
        if re.match(r"^\*\*[^*]{1,30}\*\*\s*$", stripped):
            continue
        kept.append(ln)
    t = "\n".join(kept).strip()

    matches = list(label_pat.finditer(t))
    if matches:
        t = t[matches[-1].end():].strip()

    if len(t) >= 2 and t[0] in ('"', '"', "「") and t[-1] in ('"', '"', "」"):
        t = t[1:-1].strip()

    return t or text.strip()


def detect_user_mode(user_utterance: str) -> str:
    """사용자가 설명자(teacher) 모드인지 협력자(collaborator) 모드인지 감지."""
    if not user_utterance:
        return "collaborator"
    if _sentence_count(user_utterance) >= TEACHER_MODE_MIN_SENTENCES:
        if any(k in user_utterance for k in TEACHER_MODE_KEYWORDS):
            return "teacher"
    return "collaborator"


def _detect_user_addressed_ai(user_utterance: str):
    """사용자 발화에서 특정 AI 한 명을 명시적으로 지목했는지 감지.

    반환값:
      - "ai_1" (민준) / "ai_2" (서연) / "ai_3" (연우): 정확히 한 명만 지목
      - None: 아무도 지목 안 했거나, 둘 이상 언급 (모호 → 오버라이드 안 함)

    예시:
      "민준아 너는 어떻게 생각해?"      → ai_1
      "서연이한테 물어볼래"              → ai_2
      "연우야 이거 맞아?"                → ai_3
      "민준이나 서연이"                   → None (2명 언급 → 일반 로직)
      "약수가 뭐야?"                     → None
    """
    if not user_utterance:
        return None
    name_map = {"ai_1": "민준", "ai_2": "서연", "ai_3": "연우"}
    mentioned = set()
    for aid, name in name_map.items():
        if name in user_utterance:
            mentioned.add(aid)
    if len(mentioned) == 1:
        return mentioned.pop()
    return None


def _detect_stage_advance_consent(user_utterance: str) -> str:
    """사용자 발화에서 '다음 단계로 넘어가도 되는지'에 대한 승인/거부 신호 감지.

    반환값: "consent" | "reject" | "unclear"
    - consent : "응/네/좋아/맞아/그래/넘어가자/가자/오케이" 등
    - reject  : "아직/잠깐/좀 더/기다려/근데/아니" 등
    - unclear : 사용자가 다른 내용 계속 설명 중 (명시적 응답 없음)
    """
    if not user_utterance:
        return "unclear"
    u = user_utterance.strip().lower()

    consent_tokens = (
        "응", "네", "좋아", "좋다", "맞아", "맞지", "그래", "ㅇㅇ",
        "넘어가자", "넘어가도", "넘어가도 돼", "넘어가도 될", "다음",
        "가자", "고고", "오케이", "ok", "okay", "ㅇㅋ", "yes",
        "알겠어", "이해했어", "확실해", "문제 없어", "됐어",
    )
    reject_tokens = (
        "아직", "잠깐", "기다려", "멈춰", "좀 더", "더 얘기",
        "아니", "안 돼", "안돼", "모르겠어", "헷갈려",
        "근데", "그런데", "no", "not yet",
    )

    # reject를 먼저 체크 (consent 키워드와 겹칠 수 있음: "아직 잘 모르겠어")
    for tok in reject_tokens:
        if tok in u:
            return "reject"
    for tok in consent_tokens:
        if tok in u:
            return "consent"
    return "unclear"


def _trim_to_last_complete_sentence(text: str) -> str:
    """잘린 발화를 마지막 완결 문장(. ? ! 종결)까지 트림.

    완결 문장이 하나도 없으면 빈 문자열 반환 → caller가 generic fallback 사용.
    """
    if not text:
        return ""
    # 끝의 말줄임표·공백 제거
    s = text.rstrip().rstrip("…").rstrip(".")
    s = s.rstrip()
    # 마지막 종결부호 찾기
    for i in range(len(s) - 1, -1, -1):
        if s[i] in ".!?":
            return s[: i + 1]
    return ""


_DIVISOR_PATTERNS = [
    # "X의 약수는 1, 2" / "X의 약수가 1과 2" / "X의 약수를 1,2,4"
    re.compile(r"\d+\s*의\s*약수(는|가|를|로)\s*\d+\s*[,과와및]"),
    # "X는 약수가 1과 2" / "X은 약수로 1, 2"
    re.compile(r"\d+\s*(은|는|이|가)\s*약수(가|로|는|를)\s*\d+\s*[,과와및]"),
    # "X는 약수가 N개" / "X는 약수 N개" — 개수 단언 (조사 가/는 옵션)
    re.compile(r"\d+\s*(은|는|이|가)\s*약수(가|는|이)?\s*\d+\s*개"),
    # "약수가 1, 2 (과/와/만/뿐/야)" — 약수 본인이 풀어주기
    re.compile(r"약수(가|는)\s*\d+\s*[,과와]\s*\d+"),
    # "약수는 1과 자기자신" / "약수가 1과 N뿐"
    re.compile(r"약수(는|가)\s*1\s*[,과와]\s*(자기|자신|본인|\d+)"),
    # "X의 약수가 N개" 식 단언 ("4의 약수가 3개잖아")
    re.compile(r"\d+\s*의\s*약수(가|는)\s*\d+\s*개"),
    # "약수가 N개야/잖아/이야" — 주어 없이 개수만 단언
    re.compile(r"약수(가|는)\s*\d+\s*개\s*(야|이야|잖|네|이|라)"),
]


def _strip_divisor_listing(text: str) -> str:
    """약수 나열을 포함한 발화에서 위반 부분을 제거하고 질문 형태로 치환.

    재생성도 실패한 경우의 최후 fallback. AI 발화의 '2의 약수는 1, 2고'
    같은 부분을 통째로 제거하고 가장 안전한 generic 질문으로 대체.

    실측 케이스:
      "예를 들어 2랑 3을 봐볼래? 2의 약수는 1, 2고 3의 약수는 1, 3이야. 공통점이 뭐 있어?"
      → "예를 들어 2랑 3을 봐볼래? 약수가 뭐가 있는지 같이 세어볼까?"
    """
    if not text:
        return "음, 같이 한 번 봐볼래?"
    s = text.strip()
    # 약수 나열 절을 통째로 제거 (문장 단위 분할 후 위반 문장 drop)
    sentences = re.split(r'(?<=[.!?])\s+', s)
    safe = [sent for sent in sentences if not _ai_lists_divisors(sent)]
    if safe:
        result = " ".join(safe).strip()
        # 결과가 너무 짧으면 질문 보강
        if len(result) < 8:
            result = result.rstrip(".?!") + " 약수가 뭘까?"
        return result
    # 모든 문장이 위반이면 generic
    return "음, 약수가 뭐가 있는지 같이 세어볼래?"


_PARTICLE_CHARS = "은는이가을를과와에서도만뿐"  # 빈도 높은 조사 모음
_HANGUL_RE = re.compile(r"[가-힣]")


def _hint_to_fuzzy_regex(hint):
    """detection_hint 문자열을 조사·공백 변형 허용 정규식으로 변환.

    일반화 규칙 (v1.32):
      1. 어절 끝에 명시적 조사가 있으면 그 조사를 광범위 옵셔널 그룹으로 변환
         예: "약수가" → "약수[은는이가을를과와에서도만뿐]?"
      2. 어절 끝이 한글이지만 명시적 조사가 없어도 옵셔널 조사 허용
         예: "약수" → "약수[은는이가을를과와에서도만뿐]?"
      3. 어절 사이 공백은 \\s* (0+ 공백)
      4. 숫자·영문은 escape 그대로

    예:
      "약수가 3개"   → "약수[은는이가을를과와에서도만뿐]?\\s*3개[은는이가을를과와에서도만뿐]?"
      "1과 자기자신"  → "1[은는이가을를과와에서도만뿐]?\\s*자기\\s*자신[은는이가을를과와에서도만뿐]?"
      "23은 소수"    → "23[은는이가을를과와에서도만뿐]?\\s*소수[은는이가을를과와에서도만뿐]?"

    이 변환으로 "약수가 3개" hint가 "4의 약수는 3개인거같은데" text를 매칭한다.
    """
    if not hint:
        return None
    parts = hint.split()
    out_parts = []
    for part in parts:
        if not part:
            continue
        # 어절 끝이 명시적 조사면 그 자리에 옵셔널 조사 그룹
        if len(part) >= 2 and part[-1] in _PARTICLE_CHARS:
            head = re.escape(part[:-1])
            out_parts.append(f"{head}[{_PARTICLE_CHARS}]?")
        # 끝이 한글(받침 무관)이면 옵셔널 조사 허용
        elif _HANGUL_RE.match(part[-1]):
            out_parts.append(re.escape(part) + f"[{_PARTICLE_CHARS}]?")
        else:
            # 숫자, 영문, 특수문자
            out_parts.append(re.escape(part))
    return r"\s*".join(out_parts)


def _safe_re_search(pattern, text):
    """re.search 안전 래퍼 — 잘못된 정규식 패턴 방어."""
    try:
        return bool(re.search(pattern, text))
    except re.error:
        return False


def _fuzzy_match(hint, text):
    """hint를 fuzzy 정규식으로 변환해 text에서 매칭 시도.

    Strict substring과 fuzzy 정규식 둘 다 시도해 하나라도 맞으면 True.
    """
    if not hint or not text:
        return False
    if hint in text:
        return True
    pat = _hint_to_fuzzy_regex(hint)
    if pat:
        try:
            if re.search(pat, text):
                return True
        except re.error:
            pass
    return False


_DEFINITION_PATTERNS = [
    # "소수는 약수가 2개인 수" — 소수 정의 단언
    re.compile(r"소수\s*(는|란|이란)\s*[^?\n]*약수\s*(가|는|이)?\s*\d+\s*개"),
    # "합성수는 약수가 N개 이상인 수" — 합성수 정의
    re.compile(r"합성수\s*(는|란|이란)\s*[^?\n]*약수\s*(가|는|이)?\s*\d+\s*개"),
    # "약수가 N개면/이면/이라면/이상이면 소수" — 조건절 정의 (이상/이하/초과/미만 허용)
    re.compile(r"약수\s*(가|는|이|의\s*개수가)?\s*\d+\s*개\s*(이상|이하|초과|미만)?\s*(면|이면|이라면|인|인\s*수|인수)\s*[^?\n]*소수"),
    # "약수가 N개면/이상이면/이라면 합성수"
    re.compile(r"약수\s*(가|는|이|의\s*개수가)?\s*\d+\s*개\s*(이상|이하|초과|미만)?\s*(면|이면|이라면|인|인\s*수|인수)\s*[^?\n]*합성수"),
    # "소수는 ~ 합성수는 ~" — 한 문장에서 둘 다 정의
    re.compile(r"소수\s*(는|란|이란)[^.!?\n]{1,40}(이고|이며|이고는|이고,|,)\s*합성수\s*(는|란|이란)"),
    # "소수는 ~ 합성수는 그게 아닌" — 사용자 사례 (직접 금지)
    re.compile(r"소수\s*(는|란)[^.!?\n]*합성수\s*(는|란)\s*[^.!?\n]*아닌"),
    # "1과 자기 자신만 (을) 약수로" — 정의 일부
    re.compile(r"1\s*(과|와)\s*자기\s*자신(만|뿐)?\s*[^?\n]*약수"),
    # "약수가 자기자신과 1뿐"
    re.compile(r"약수\s*(가|는)\s*[^?\n]*자기\s*자신[^?\n]*뿐"),
]


def _ai_states_definition(text: str) -> bool:
    """AI 발화가 소수/합성수 정의를 직접 단언하는지 정규식으로 탐지.

    s1-2(소수 정의) / s1-3(합성수 정의) 체크포인트 보호용. 정의는 사용자가
    귀납으로 도출해야 hit. AI가 통째로 제시하면 학습 기회가 사라진다.

    탐지 대상 (모두 위반):
      - "소수는 약수가 2개인 수"
      - "합성수는 약수가 3개 이상인 수"
      - "약수가 2개면 소수"
      - "소수는 ~고 합성수는 그게 아닌 수"
      - "1과 자기 자신만 약수로 가지는"

    탐지 제외 (질문 형태):
      - "소수가 뭐야?" / "합성수가 뭐 같아?" (정의 묻기)
      - "약수가 몇 개야?" (개수 묻기)
    """
    if not text:
        return False
    s = text.strip()
    for pat in _DEFINITION_PATTERNS:
        if pat.search(s):
            return True
    return False


_WHY_ONE_PATTERNS = [
    # "1은 왜 소수도 합성수도 아닌 거야" / "1이 왜 ... 아닐까" / "아니야"
    re.compile(r"1\s*(은|는|이|가)\s*왜\s*[^?\n]*소수[^?\n]*합성수[^?\n]*(아닌|아니|아냐|아닐)"),
    # "왜 1은 소수도 합성수도 아니야"
    re.compile(r"왜\s*1\s*(은|는|이|가)?\s*[^?\n]*소수[^?\n]*합성수[^?\n]*(아닌|아니|아냐|아닐)"),
    # "1이 둘 다 아닌 이유" / "1은 둘 다 아닌 이유"
    re.compile(r"1\s*(은|는|이|가)\s*둘\s*다\s*아닌\s*이유"),
    # "1이 왜 둘 다 아닌" / "왜 둘 다 아닐까"
    re.compile(r"1\s*(은|는|이|가)\s*왜\s*[^?\n]*둘\s*다\s*(아닌|아니|아냐|아닐)"),
    # "1은 소수도 합성수도 아닌 이유"
    re.compile(r"1\s*(은|는)\s*소수[^?\n]*합성수[^?\n]*아닌\s*이유"),
]


def _ai_asks_why_one_excluded(text: str) -> bool:
    """AI 발화가 '1이 왜 둘 다 아닌지' 이유를 묻는지 탐지.

    s1-3/s1-5(1의 예외성) 체크포인트 보호용. 이런 질문 자체가 "1은 둘 다 아니다"를
    전제로 깔아 사용자가 그 명제를 도출할 기회를 박탈한다.

    허용 형태(detect 대상 아님):
      - "1은 소수야 합성수야?" (binary 분류 질문)
      - "1의 약수는 뭐야?"
      - "1은 어떻게 분류해야 할까?"

    탐지 대상(모두 위반):
      - "1은 왜 소수도 합성수도 아닌 거야?"
      - "1이 왜 둘 다 아닐까?"
      - "1이 둘 다 아닌 이유가 뭐야?"
    """
    if not text:
        return False
    s = text.strip()
    for pat in _WHY_ONE_PATTERNS:
        if pat.search(s):
            return True
    return False


def _ai_lists_divisors(text: str) -> bool:
    """AI 발화가 약수를 직접 나열/단언하는지 정규식으로 탐지.

    s1-1(약수 어휘 사용) 체크포인트 보호용. 사용자가 약수를 발화하기 전에
    AI가 약수 정보를 발화하면 hit이 박탈된다. 모델이 프롬프트 규칙을 무시하는
    실측 사례가 잦아 런타임 차단 필터로 보강.

    탐지 대상 (모두 위반):
      - "2의 약수는 1, 2" (직접 나열)
      - "2는 약수가 1과 2" (관계절)
      - "2는 약수 2개" (개수 단언)
      - "약수가 1과 2" (주어 생략된 나열)

    탐지 제외:
      - "약수가 뭐야?" / "약수 세어볼래?" (질문)
      - "1의 약수는 1뿐이야" — 1은 사용자가 흔히 도출하는 자명 케이스가 아니므로
        그래도 탐지 대상에 포함 (이때도 사용자에게 묻는 게 맞음)
    """
    if not text:
        return False
    s = text.strip()
    for pat in _DIVISOR_PATTERNS:
        if pat.search(s):
            return True
    return False


def _is_incomplete_utterance(s: str) -> bool:
    """AI 발화가 중간에 잘렸는지 휴리스틱 판정.

    규칙:
      - 빈 문자열 또는 5자 미만 → incomplete (정말 짧은 garbled만 잡음)
      - **말줄임표(... 또는 …)로 끝나면 incomplete** (truncation 표시)
      - 단일 문장부호(. ! ?)로 끝나면 완결
      - 그 외에는 다중 음절 종결패턴만 완결로 인정 (단일 한글 음절은 모호)

    민준 페르소나가 "매우 짧게 (1~2문장)"로 운용되므로 11~19자 짧은 응답
    ('어디 막혔어?', '음, 약수부터.')이 정상. 20자 minimum은 너무 빡빡해서 5로 완화.
    """
    if not s:
        return True
    stripped = s.strip().rstrip('"\'」』')
    if len(stripped) < 5:
        return True
    # 1) 말줄임표 종결 → **incomplete** (가장 먼저 체크, 단일 . 보다 우선)
    if stripped.endswith("...") or stripped.endswith("…"):
        return True
    # 2) 단일 문장부호 종결 → 완결
    if stripped[-1] in ".!?":
        return False

    # 2) 다중 음절 한국어 종결 패턴만 완결로 인정 (2~4글자 확실한 패턴)
    multi_char_terminators = (
        # 의문·청유
        "어때", "어떨까", "어떨래", "어떡해",
        "볼까", "할까", "할래", "할거야", "해볼까", "해볼래",
        # 선언·동의
        "있어", "있지", "있네", "있다", "있으니",
        "맞아", "맞지", "맞네", "맞다",
        "같아", "같지", "같네", "같다",
        "잖아", "잖니", "지요", "이야", "이야",
        # 축약 동사
        "헷갈려", "모르겠어", "알겠어", "알아", "알지",
        "생각해", "이해했어", "됐어", "괜찮아",
        "해봐", "봐봐", "봐줘", "해줘", "알려줘",
        # 격식체
        "습니다", "입니다", "해요", "어요", "네요", "죠", "군요",
    )
    if any(stripped.endswith(t) for t in multi_char_terminators):
        return False

    # 3) 이 외에는 단일 한글 음절/조사 종결 → 모호 → incomplete 취급
    #    (정상 종결인데도 재시도하는 false-positive는 비용이 적으므로 수용)
    return True


class CollaborativeSession:
    """사용자 1명 + AI 학생 3명의 협력학습 세션."""

    AI_KEYS = ("ai_1", "ai_2", "ai_3")

    def __init__(self, config, prompts, learner_models, api):
        self.config = config
        self.prompts = prompts
        self.learner_models = learner_models
        self.api = api

        self.task = config["tasks"]
        self.current_stage = 1
        self.conversation = []
        self.turn_count = 0
        self.stage_turn_count = 0
        self.last_tutor_decision = None

        self.last_user_utterance_ts = time.time()
        self.last_silence_agent = None
        self.current_user_mode = "collaborator"

        # Stage 완료 2단계 게이트: 조건 충족 → 서연이 확인 질문 → 사용자 승인 → 실제 advance
        # "pending" 상태에서는 stage_complete=False로 강제해 자동 advance 방지.
        self.pending_stage_complete = False
        self.pending_stage_complete_since_turn = None

    def recent_dialogue(self, n=6):
        recent = self.conversation[-n:]
        return "\n".join(f"[{m['speaker']}]: {m['content']}" for m in recent) or "(아직 대화 없음)"

    def speaker_frequency(self, n=8):
        """최근 n턴의 AI별 발화 횟수 + 마지막 발화 기준 턴 간격을 요약한다."""
        recent = self.conversation[-n:] if n and n > 0 else self.conversation
        names = {
            "ai_1": self.config["personas"]["ai_students"]["ai_1"]["name"],
            "ai_2": self.config["personas"]["ai_students"]["ai_2"]["name"],
            "ai_3": self.config["personas"]["ai_students"]["ai_3"]["name"],
        }
        count = {"ai_1": 0, "ai_2": 0, "ai_3": 0, "user": 0}
        last_gap = {"ai_1": None, "ai_2": None, "ai_3": None}
        total = len(self.conversation)
        for i, m in enumerate(self.conversation):
            turns_ago = total - 1 - i
            speaker = m.get("speaker", "")
            aid = m.get("agent_id")
            if not aid:
                for k, nm in names.items():
                    if nm == speaker:
                        aid = k
                        break
            if aid in last_gap:
                last_gap[aid] = turns_ago
        for m in recent:
            speaker = m.get("speaker", "")
            if speaker == "사용자":
                count["user"] += 1
                continue
            aid = m.get("agent_id")
            if not aid:
                for k, nm in names.items():
                    if nm == speaker:
                        aid = k
                        break
            if aid in count:
                count[aid] += 1

        parts = [
            f"최근 {len(recent)}턴 중",
            f"{names['ai_1']}(ai_1) {count['ai_1']}회",
            f"{names['ai_2']}(ai_2) {count['ai_2']}회",
            f"{names['ai_3']}(ai_3) {count['ai_3']}회",
            f"사용자 {count['user']}회",
        ]
        never_spoken = [a for a, g in last_gap.items() if g is None]
        quiet_hints = []
        if never_spoken:
            quiet_hints.append(
                " / ".join(f"{names[a]}({a}) 아직 발화 없음" for a in never_spoken)
            )
        else:
            max_gap_aid = max(last_gap, key=lambda a: last_gap[a])
            max_gap = last_gap[max_gap_aid]
            if max_gap is not None and max_gap >= 4:
                quiet_hints.append(
                    f"{names[max_gap_aid]}({max_gap_aid})가 마지막으로 말한 뒤 {max_gap}턴 경과 → rotation 권장"
                )
        summary = " · ".join(parts)
        if quiet_hints:
            summary += "\n" + " / ".join(quiet_hints)
        return summary

    def _recent_ai_speakers(self, n):
        """최근 n턴의 AI 발화자 리스트(시간순)를 ai_1/ai_2/ai_3 키로 반환."""
        names_to_id = {
            self.config["personas"]["ai_students"][aid]["name"]: aid
            for aid in self.AI_KEYS
        }
        out = []
        for m in self.conversation[-n:]:
            aid = m.get("agent_id")
            if not aid:
                aid = names_to_id.get(m.get("speaker", ""))
            if aid in self.AI_KEYS:
                out.append(aid)
        return out

    def _enforce_rotation_guard(self, decision):
        """LLM이 반환한 speaking_agents를 rotation HARD 규칙으로 후처리.

        규칙:
          HARD-2 : 최근 6턴에서 서연이 2회 이하 & 이번 턴 미포함 & 침묵턴 아님 → 서연 강제 추가
          HARD-3 : 최근 3턴 중 2회 이상 발화한 AI가 이번 턴에 포함되어 있으면 제외
          HARD-B : 최근 3턴이 민준·연우 핑퐁이고 이번 턴이 민준/연우만이면 서연 강제 포함

        침묵 유도 턴(decision.silence_trigger=True)은 예외 — 단일 AI 선제 발화이므로
        로테이션 개입 없이 통과시킨다.
        """
        # 침묵 턴은 건드리지 않는다
        if decision.get("silence_trigger"):
            return

        speakers_in = list(decision.get("speaking_agents") or [])
        speakers = list(speakers_in)
        recent_all = self._recent_ai_speakers(6)
        recent3 = self._recent_ai_speakers(3)
        print(f"       · [rotation-guard] LLM speakers={speakers_in} recent3={recent3} "
              f"ai_2_count_6={recent_all.count('ai_2')}")

        def has_directive(aid):
            return decision.get(f"{aid}_directive") is not None

        def ensure_sooyeon_directive():
            if not decision.get("ai_2_directive"):
                decision["ai_2_directive"] = {
                    "role": "진행자 + 되짚기",
                    "speech_goal": "지금까지 나온 학습자의 용어·핵심어를 학습자 언어 그대로 짧게 묶어 확인한다",
                    "must_include": "학습자가 방금 쓴 핵심어 + 확인 질문 1개",
                    "must_avoid": "정답·정의 단정, 새 개념 도입, 두 개 이상의 질문",
                }

        # HARD-3: 최근 3턴에 2회 이상 발화한 AI는 이번 턴에서 제외
        for aid in list(speakers):
            if recent3.count(aid) >= 2:
                speakers.remove(aid)
                # directive도 무효화 (LLM이 생성한 것을 버림)
                decision[f"{aid}_directive"] = None

        # HARD-B: 최근 3턴이 민준·연우만으로 구성된 핑퐁 + 이번 턴도 민준·연우뿐
        only_12 = recent3 and all(a in ("ai_1", "ai_3") for a in recent3) and len(recent3) >= 2
        current_only_12 = speakers and all(a in ("ai_1", "ai_3") for a in speakers)
        if only_12 and current_only_12 and "ai_2" not in speakers:
            # 둘 중 최근에 더 많이 말한 쪽을 서연으로 교체
            counts3 = {a: recent3.count(a) for a in ("ai_1", "ai_3")}
            victim = max(counts3, key=counts3.get) if counts3 else None
            if victim and victim in speakers:
                speakers.remove(victim)
                decision[f"{victim}_directive"] = None
            speakers.append("ai_2")
            ensure_sooyeon_directive()

        # HARD-2 (완화): 서연이 **최근 6턴에 한 번도 발화 안 했고**, 이번 턴도 미포함일 때만
        # 강제 추가. 기존의 "2회 이하 → 강제"는 서연 반복 투입의 주범이라 완전 폐기.
        # 최근 3턴에 이미 서연이 있었다면 절대 추가하지 않는다 (반복 정리 차단).
        if (recent_all.count("ai_2") == 0
                and "ai_2" not in recent3
                and "ai_2" not in speakers):
            # 이미 두 명이 잡혔어도 교체하지는 않음 — 서연이 오랫동안 말 못 했을 때만
            # "끼워넣기" 정도로 제한. 과도한 교체는 다른 AI의 역할도 방해.
            if len(speakers) == 0:
                speakers.append("ai_2")
                ensure_sooyeon_directive()

        # 중복 제거 + 순서 보존
        seen = set()
        deduped = []
        for a in speakers:
            if a not in seen:
                seen.add(a)
                deduped.append(a)
        decision["speaking_agents"] = deduped

        # directive 정합성: speaking_agents에 없는 AI의 directive는 None
        for aid in self.AI_KEYS:
            if aid not in deduped:
                decision[f"{aid}_directive"] = None
        if deduped != speakers_in:
            print(f"       · [rotation-guard] 교정: {speakers_in} → {deduped}", flush=True)

    def _cap_single_speaker(self, decision):
        """한 턴 기본 1명 발화로 강제. 두 명이 유사한 호응을 반복하는 문제 방지.

        **2명 허용 예외**:
          - silence_trigger=True (침묵 유도 턴, 이미 단일 AI로 처리됨)
          - directive role이 **명백히 상호보완적**인 경우:
              * 한쪽 role에 "질문자·유도·반례·설명" 포함 (main)
              * 다른쪽 role에 "정리·요약·되짚기·확인" 포함 (complementary)
              * 그 외 중복 가능 조합은 main 1명으로 축소

        우선순위(중복 시 남길 AI): 사용자 발화에 가장 적합한 역할자.
        기본: 이미 LLM이 선정한 순서에서 첫 번째.
        """
        speakers = list(decision.get("speaking_agents") or [])
        if len(speakers) <= 1:
            return

        directives = {a: (decision.get(f"{a}_directive") or {}) for a in speakers}
        roles = {a: (d.get("role", "") or "") + " " + (d.get("speech_goal", "") or "")
                 for a, d in directives.items()}

        main_markers = ("질문자", "유도", "반례", "설명", "답 힌트", "근거", "판정")
        comp_markers = ("정리", "요약", "되짚기", "확인", "재진술", "진행자")

        def _tag(text):
            has_main = any(m in text for m in main_markers)
            has_comp = any(m in text for m in comp_markers)
            if has_main and not has_comp:
                return "main"
            if has_comp and not has_main:
                return "comp"
            return "mixed"

        tags = {a: _tag(r) for a, r in roles.items()}
        tagset = set(tags.values())

        # main + comp 조합이면 2명 유지 — **단, comp 반복 방지 조건 있음**
        if "main" in tagset and "comp" in tagset and len(speakers) == 2:
            # 최근 2턴에 이미 comp 역할 발화가 있었다면 이번 턴에서 comp 제외
            # (서연 정리 반복 차단 — "지금까지 얘기한 거 정리해볼게" 연타 방지)
            recent2 = self._recent_ai_speakers(2)
            if recent2:
                # 최근 2턴 중 comp성 AI가 있었는지 휴리스틱 판단:
                # 서연(ai_2)이 최근 2턴 안에 발화했으면 comp 역할로 연속 투입 안 함
                if "ai_2" in recent2 and any(tags.get(a) == "comp" and a == "ai_2" for a in speakers):
                    dropped_aid = "ai_2"
                    decision[f"{dropped_aid}_directive"] = None
                    speakers = [a for a in speakers if a != dropped_aid]
                    decision["speaking_agents"] = speakers
                    print(f"       · [single-speaker-cap] comp 반복 방지 — 서연 최근 2턴 내 이미 발화 → 제외", flush=True)
                    return
            return

        # 그 외: 1명으로 축소. keeper 선정 우선순위:
        #   1) tag == "main" 인 AI (설명·질문 주도)
        #   2) 없으면 LLM이 먼저 선정한 순서상 첫 번째
        keeper = None
        for a in speakers:
            if tags.get(a) == "main":
                keeper = a
                break
        if keeper is None:
            keeper = speakers[0]

        dropped = [a for a in speakers if a != keeper]
        for a in dropped:
            decision[f"{a}_directive"] = None
        decision["speaking_agents"] = [keeper]
        print(f"       · [single-speaker-cap] {speakers} → [{keeper}] "
              f"(tags={tags}, dropped={dropped})")

    def _stage_complete_safety_net(self, decision, user_utterance):
        """Stage 완료 판정.

        **주 경로**: tasks.json의 `stage_complete_required` 목록에 지정된
        필수 체크포인트 ID가 사용자 progress에 **모두 hit**되어 있으면 완료.

        **보조 경로** (체크포인트 업데이트가 늦은 경우 대비): 사용자 최근 발화
        키워드로 직접 검사. 보수적으로 — 각 체크포인트의 핵심 내용이
        명시적으로 발화에 등장해야 인정.
        """
        if decision.get("stage_complete"):
            return  # 이미 true면 그대로 둠

        # 최근 3턴의 사용자 발화를 합쳐서 판정
        recent_user = [user_utterance or ""]
        for m in reversed(self.conversation[-10:]):
            if m.get("speaker") == "사용자" and m.get("content") != user_utterance:
                recent_user.append(m.get("content", ""))
            if len(recent_user) >= 3:
                break
        combined = " ".join(recent_user)

        stage_num = self.current_stage
        stage_info = self.current_stage_info()
        hit_reason = None

        # === 주 경로: stage_complete_required 전부 hit 됐는지 progress에서 확인 ===
        required = stage_info.get("stage_complete_required") or []
        if required:
            user_prog = (
                (self.learner_models.get("user", {}).get("checkpoint_progress") or {})
                .get(str(stage_num)) or {}
            )
            hit_ids = [cid for cid in required
                       if user_prog.get(cid, {}).get("hit")]
            missing = [cid for cid in required if cid not in hit_ids]
            print(f"       · [stage-guard] required={required} hit={hit_ids} missing={missing}", flush=True)
            if not missing:
                hit_reason = f"stage_complete_required 전부 hit ({required})"

        # === 보조 경로: 발화 키워드 — progress 업데이트 늦을 때만 활용 ===
        # 단, 주 경로가 미완료여도 여기서 자동 완료시키지는 않고 진단 목적으로만 로그.
        if not hit_reason and stage_num == 1:
            # v1.51: 단편적 발화로는 hit 안 되도록 — '소수'/'합성수' 단어 또는
            # 명시적 일반화 표현이 있을 때만 카운트
            s12 = any(k in combined for k in [
                "소수는 약수가 2", "소수의 약수가 2", "소수는 약수 2", "소수의 약수 2",
                "소수는 약수 두", "1과 자기 자신만 약수", "1과 자기자신만 약수",
                "1과 본인만 약수", "약수가 1과 자기자신",
                "1과 자기자신뿐인 수가 소수",
            ])
            s13 = any(k in combined for k in [
                "합성수는 약수가 3", "합성수의 약수가 3", "합성수는 약수 3",
                "합성수는 약수가 여러", "합성수는 약수가 많",
                "합성수는 1과 자기 자신 외", "합성수는 1과 자기자신 외",
                "약수가 3개 이상인 수가 합성수", "다른 약수도 있는 수가 합성수",
                "1과 자기자신 외에도 약수",
            ])
            print(f"       · [stage-guard s1 키워드] s1-2={s12} s1-3={s13}", flush=True)

        elif not hit_reason and stage_num == 2:
            has_23 = "23" in combined
            has_29 = "29" in combined
            others_covered = any(k in combined for k in [
                "나머지는 합성수", "나머지가 합성수", "다른 건 합성수",
                "다른건 합성수", "다른 수는 합성수", "이외는 합성수",
            ])
            print(f"       · [stage-guard s2 키워드] 23={has_23} 29={has_29} 나머지={others_covered}", flush=True)

        elif not hit_reason and stage_num == 3:
            # s3: '소수=홀수 거짓 + 반례 2' / '5의 배수=합성수 거짓 + 반례 5'
            has_2_counter = any(k in combined for k in [
                "2는 소수", "2가 소수", "2도 소수", "2는 짝수", "2 반례",
            ])
            has_5_counter = any(k in combined for k in [
                "5는 소수", "5가 소수", "5도 소수", "5 반례", "5는 5의 배수",
            ])
            has_false_1 = any(k in combined for k in ["거짓", "아니다", "틀렸"])
            print(f"       · [stage-guard s3 키워드] 거짓판단={has_false_1} "
                  f"2반례={has_2_counter} 5반례={has_5_counter}", flush=True)

        if hit_reason:
            decision["stage_complete"] = True
            print(f"       · [stage-guard] 완료 기준 충족 → stage_complete=true (reason: {hit_reason})", flush=True)

    def _apply_user_addressed_override(self, decision, user_utterance):
        """사용자가 특정 AI를 명시적으로 지목했다면 그 AI 단독 발화로 강제.

        rotation/cap/stage_gate 이후 최종 단계에 실행되어, 사용자 의도가 모든
        자동 규칙을 이기도록 한다. 지목된 AI에 directive가 없으면 기본 "직접
        응답" directive를 생성. 다른 AI의 directive는 None으로 초기화.
        """
        addressed = _detect_user_addressed_ai(user_utterance)
        if not addressed:
            return
        name_map = {"ai_1": "민준", "ai_2": "서연", "ai_3": "연우"}
        addressed_name = name_map.get(addressed, addressed)

        current = decision.get("speaking_agents") or []
        if current == [addressed]:
            # 이미 해당 AI만 선정됨 — 별도 작업 없음
            return

        print(f"       · [user-addressed] 사용자가 {addressed_name}({addressed}) 지목 → 단독 발화 강제 (기존: {current})", flush=True)

        # 지목된 AI의 directive가 없으면 기본 응답 directive 생성
        if not decision.get(f"{addressed}_directive"):
            decision[f"{addressed}_directive"] = {
                "role": f"{addressed_name} — 사용자 직접 응답자",
                "speech_goal": (
                    f"사용자가 나({addressed_name})를 직접 지목해 말 걸었다. "
                    f"내 페르소나 말투로 사용자의 질문·요청에 직접 응답한다."
                ),
                "must_include": "사용자가 요청한 내용에 대한 내 대답 또는 반응",
                "must_avoid": "다른 AI에게 넘기기, 대답 회피, 주제 돌리기",
            }

        # 다른 AI directive는 모두 비우고 speaking_agents 단일화
        for other in self.AI_KEYS:
            if other != addressed:
                decision[f"{other}_directive"] = None
        decision["speaking_agents"] = [addressed]

    def _check_quiz_answer(self, user_utterance, quiz):
        """completion_quiz의 사용자 답변을 엄격 검증 (v1.51).

        강화 룰:
          - quiz question에서 핵심 숫자를 추출 (예: 21).
          - 사용자 답에 그 숫자가 등장해야 정답·오답 판정 시작.
          - 숫자 없이 단순 '오른쪽'/'합성수' 단독 발화는 unclear (서연이 다시 묻기).
          - 정답 키워드 매칭 → correct
          - 오답 키워드 매칭 → incorrect

        반환: 'correct' | 'incorrect' | 'unclear'
        """
        if not user_utterance or not quiz:
            return "unclear"
        text = user_utterance

        # quiz 질문에서 핵심 숫자 추출
        question = quiz.get("question") or ""
        nums = re.findall(r"\d+", question)
        target_num = nums[0] if nums else None

        # 사용자 답에 핵심 숫자 없으면 unclear (단편 발화 차단)
        if target_num and target_num not in text:
            return "unclear"

        wrong_keys = quiz.get("wrong_keywords") or []
        right_keys = quiz.get("correct_answers_keywords") or []

        right_hit = any(k in text for k in right_keys)
        wrong_hit = any(k in text for k in wrong_keys)

        if right_hit:
            return "correct"
        if wrong_hit:
            return "incorrect"
        return "unclear"

    def _apply_stage_gate(self, decision, user_utterance):
        """Stage 완료 2단계 게이트: 조건 충족 → 서연 스몰 퀴즈 → 정답 시 advance.

        v1.48: '동의 묻기'에서 '스몰 퀴즈 정답 검증'으로 변경.
        - pending 상태에서 사용자 답변을 quiz와 매칭
          - correct → advance
          - incorrect → "다시 생각해봐" 힌트 directive + pending 유지
          - unclear → pending 유지 (서연이 다시 묻기)
        - 조건 충족 첫 턴에 서연 단독 발화로 quiz 출제 directive 주입
        - pending 3턴 경과하면 자동 해제 (stuck 방지)

        Stage에 completion_quiz가 없으면 종래 consent 방식으로 fallback.
        """
        stage = self.current_stage_info()
        quiz = stage.get("completion_quiz")

        # quiz 없는 stage: 종래 consent 방식
        if not quiz:
            self._apply_stage_gate_consent_legacy(decision, user_utterance)
            return

        if self.pending_stage_complete:
            verdict = self._check_quiz_answer(user_utterance, quiz)
            print(f"       · [stage-gate quiz] verdict={verdict} "
                  f"answer={user_utterance[:60]!r}", flush=True)
            if verdict == "correct":
                print(f"       · [stage-gate] 퀴즈 정답 → stage_complete=True 확정", flush=True)
                decision["stage_complete"] = True
                self.pending_stage_complete = False
                self.pending_stage_complete_since_turn = None
            elif verdict == "incorrect":
                print(f"       · [stage-gate] 퀴즈 오답 → 서연이 힌트 후 재시도 유도", flush=True)
                decision["stage_complete"] = False
                # 서연이 짧은 힌트 + 재질문
                decision["speaking_agents"] = ["ai_2"]
                decision["ai_1_directive"] = None
                decision["ai_3_directive"] = None
                decision["ai_2_directive"] = {
                    "role": "진행자 (퀴즈 오답 → 힌트 후 재시도)",
                    "speech_goal": (
                        f"사용자가 퀴즈 답을 틀렸다 (정답은 '{quiz.get('correct_answer','')}'). "
                        f"바로 답을 알려주지 말고 약수를 같이 세어보자고 부드럽게 유도. "
                        f"예: '음, 21을 한 번 나눠볼까? 21은 어떤 수로 나눠지지?'"
                    ),
                    "must_include": "약수를 같이 세어보자는 짧은 유도 + 다시 분류 질문",
                    "must_avoid": "정답 직접 발화, 추궁",
                }
            else:  # unclear
                held_turns = self.turn_count - (self.pending_stage_complete_since_turn or self.turn_count)
                if held_turns >= 3:
                    print(f"       · [stage-gate] pending 3턴 경과 → 해제, Stage 유지", flush=True)
                    self.pending_stage_complete = False
                    self.pending_stage_complete_since_turn = None
                decision["stage_complete"] = False
        elif decision.get("stage_complete"):
            # 최초로 완료 조건 충족 감지 → 퀴즈 출제 모드로 전환
            print(f"       · [stage-gate] Stage {self.current_stage} 완료 조건 충족 "
                  f"— 서연이 스몰 퀴즈 출제", flush=True)
            self.pending_stage_complete = True
            self.pending_stage_complete_since_turn = self.turn_count
            decision["stage_complete"] = False  # quiz 정답 전엔 advance 막음
            decision["speaking_agents"] = ["ai_2"]
            decision["ai_1_directive"] = None
            decision["ai_3_directive"] = None
            decision["ai_2_directive"] = {
                "role": "진행자 (Stage 완료 확인 — 스몰 퀴즈)",
                "speech_goal": (
                    f"사용자가 양쪽 그룹 차이를 잘 설명했다. 마지막으로 적용 퀴즈로 확인. "
                    f"퀴즈 질문 그대로: '{quiz.get('question','')}'"
                ),
                "must_include": f"퀴즈 질문 그대로 한 번 묻기 — '{quiz.get('question','')}'",
                "must_avoid": "정답 직접 발화, 답 힌트, 추가 설명",
            }

    def _apply_stage_gate_consent_legacy(self, decision, user_utterance):
        """completion_quiz가 없는 Stage용 종래 consent 방식 (Stage 2/3)."""
        consent = _detect_stage_advance_consent(user_utterance)
        if self.pending_stage_complete:
            if consent == "consent":
                print(f"       · [stage-gate consent] 사용자 승인 → stage_complete=True", flush=True)
                decision["stage_complete"] = True
                self.pending_stage_complete = False
                self.pending_stage_complete_since_turn = None
            elif consent == "reject":
                print(f"       · [stage-gate consent] 거부 → pending 해제, Stage 유지", flush=True)
                decision["stage_complete"] = False
                self.pending_stage_complete = False
                self.pending_stage_complete_since_turn = None
            else:
                held_turns = self.turn_count - (self.pending_stage_complete_since_turn or self.turn_count)
                if held_turns >= 2:
                    print(f"       · [stage-gate consent] pending 2턴 경과 → 해제", flush=True)
                    self.pending_stage_complete = False
                    self.pending_stage_complete_since_turn = None
                decision["stage_complete"] = False
        elif decision.get("stage_complete"):
            print(f"       · [stage-gate consent] 완료 조건 충족 — 서연 확인 질문", flush=True)
            self.pending_stage_complete = True
            self.pending_stage_complete_since_turn = self.turn_count
            decision["stage_complete"] = False
            decision["speaking_agents"] = ["ai_2"]
            decision["ai_1_directive"] = None
            decision["ai_3_directive"] = None
            decision["ai_2_directive"] = {
                "role": "진행자 (Stage 완료 승인 확인)",
                "speech_goal": (
                    "지금까지 사용자가 한 설명을 한 줄로 짧게 요약해 재진술한 뒤, "
                    "'이렇게 이해하고 다음 단계로 넘어가도 될까?' 라고 분명하게 승인을 묻는다."
                ),
                "must_include": "사용자 핵심 표현 짧게 인용 + 승인 질문",
                "must_avoid": "새로운 개념 도입, 긴 설명",
            }

    def _detect_user_repeat_or_frustration(self, user_utterance):
        """사용자가 같은 답변 반복 OR 짜증 신호를 보냈는지 일반적으로 감지.

        주제·숫자에 무관한 일반 휴리스틱:
          (A) 짜증 어구 — "했잖아", "방금 말", "라고 했다", "이미 ~", "내가 ~라고"
          (B) 반복 답변 — 최근 사용자 발화 3개 중 2개 이상이 같은 핵심 토큰
              (숫자 N + 단위 "개"·"뿐" 또는 동일 분류어 "소수"·"합성수")
              을 공유

        반환: 감지되면 dict {"signal": "frustration"|"repeat", "evidence": str},
              아니면 None.
        """
        if not user_utterance:
            return None

        # (A) 짜증 어구 — 일반화 패턴 (특정 인사말·서술어 무관)
        frustration_pat = re.compile(
            r"(했잖아|했다고|했었|했어|"  # ~했어 단언
            r"라고\s*했|다고\s*했|"        # ~라고 했/~다고 했
            r"방금\s*말|이미\s*말|"
            r"내가\s*[^?\n]*라고|"
            r"진짜\s*아까|아까\s*말|"
            r"또\s*묻|또\s*같은|또\s*물어|"
            r"그건\s*아까|"
            r"답\s*했|들었잖|봤잖|"
            r"라고요?$|라고\s*했어|다고\s*했어)"  # 발화 끝 "~라고"
        )
        if frustration_pat.search(user_utterance):
            return {"signal": "frustration", "evidence": user_utterance[:60]}
        # 짧은 발화 끝이 "라고/라고."이면 단독 짜증 시그널
        stripped = user_utterance.strip().rstrip(".?!").rstrip()
        if len(stripped) <= 30 and (stripped.endswith("라고") or stripped.endswith("다고")):
            return {"signal": "frustration", "evidence": stripped}

        # (B) 반복 답변 — 최근 사용자 3개 발화에서 핵심 토큰 비교
        recent_user = [user_utterance]
        for m in reversed(self.conversation[-12:]):
            if (m.get("speaker") == "사용자"
                    and m.get("content") != user_utterance):
                recent_user.append(m.get("content", "") or "")
            if len(recent_user) >= 3:
                break
        if len(recent_user) < 2:
            return None

        # 핵심 토큰 추출: 숫자+단위, 분류어, 정의 표현
        def _key_tokens(u):
            tokens = set()
            # 숫자 + 약수/개/뿐
            for m in re.finditer(r"(\d+)\s*(개|뿐)", u):
                tokens.add(f"{m.group(1)}{m.group(2)}")
            # "X의 약수" 패턴
            for m in re.finditer(r"(\d+)\s*의\s*약수", u):
                tokens.add(f"div_{m.group(1)}")
            # 분류어
            if "소수" in u and ("아니" not in u and "둘 다" not in u):
                tokens.add("classify_prime")
            if "합성수" in u and ("아니" not in u):
                tokens.add("classify_composite")
            if "둘 다 아니" in u or "소수도 합성수도" in u:
                tokens.add("classify_neither")
            return tokens

        token_sets = [_key_tokens(u) for u in recent_user[:3]]
        # 두 발화 이상에 같은 토큰이 등장하면 반복으로 간주
        from collections import Counter
        all_tokens = []
        for s in token_sets:
            all_tokens.extend(s)
        cnt = Counter(all_tokens)
        repeated = [t for t, c in cnt.items() if c >= 2]
        if repeated:
            return {"signal": "repeat", "evidence": ",".join(repeated)}

        return None

    # 소수/합성수 식별용 (Stage 1 generalization 체크)
    _PRIMES_30 = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
    _COMPOSITES_30 = {4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25, 26, 27, 28}

    def _check_generalization_via_examples(self, stage_num=None):
        """사용자 최근 발화에서 소수/합성수 예시가 2개 이상 누적되면 일반화 hit.

        s1-2 (소수=약수 2개) 일반화 판정 보강:
          - 단일 사례 '2의 약수는 2개'만으로는 hit 안 됨
          - 사용자가 최근 6턴 안에 **서로 다른 소수 2개 이상**을 "약수가 2개"
            패턴과 함께 발화했으면 일반화로 보고 s1-2 hit
        s1-3 (합성수=약수 3개+) 동일 로직 적용.

        반환: hit 처리할 cp_id 리스트 (예: ["s1-2"], ["s1-3"], 또는 둘 다).
        """
        stage_num = stage_num or self.current_stage
        if stage_num != 1:
            return []

        # 최근 사용자 발화 모으기 (최근 6개)
        recent_user = []
        for m in reversed(self.conversation[-20:]):
            if m.get("speaker") == "사용자":
                recent_user.append(m.get("content", "") or "")
            if len(recent_user) >= 6:
                break

        primes_with_2 = set()
        composites_with_3plus = set()

        for u in recent_user:
            if not u:
                continue
            # "X의 약수는 2개" / "X는 약수가 2개" / "X 약수 2개" — 약수가 X 직후
            for m in re.finditer(
                r"(\d+)\s*[은는의이가]?\s*약수[은는이가도]?[^.!?\n]{0,8}?2\s*개", u
            ):
                try:
                    n = int(m.group(1))
                    if n in self._PRIMES_30:
                        primes_with_2.add(n)
                except ValueError:
                    pass
            # "X의 약수는 3개~10개" 합성수 패턴
            for m in re.finditer(
                r"(\d+)\s*[은는의이가]?\s*약수[은는이가도]?[^.!?\n]{0,8}?([3-9]|10)\s*개", u
            ):
                try:
                    n = int(m.group(1))
                    if n in self._COMPOSITES_30:
                        composites_with_3plus.add(n)
                except ValueError:
                    pass

        hits = []
        if len(primes_with_2) >= 2:
            print(f"       · [generalization] 소수 2+ 예시 누적 → s1-2 hit (예시: {sorted(primes_with_2)})", flush=True)
            hits.append("s1-2")
        if len(composites_with_3plus) >= 2:
            print(f"       · [generalization] 합성수 2+ 예시 누적 → s1-3 hit (예시: {sorted(composites_with_3plus)})", flush=True)
            hits.append("s1-3")
        return hits

    def _verify_llm_hits(self, llm_hits, user_utterance, stage):
        """LLM이 반환한 checkpoint_hits를 검증 (v1.57 — 맥락 결합 검증 추가).

        검증 경로 (둘 중 하나라도 통과하면 인정):
          (1) 사용자 발화 단독에 detection_hints/patterns 매칭 (보수적, 기본)
          (2) 사용자 발화 + 직전 AI 발화 합산 매칭 + 사용자가 짧은 긍정 OR 숫자 발화 (맥락)

        예시:
          AI "23 소수 맞아?" + 사용자 "응" → 맥락 인정
          AI "20~30 중 소수?" + 사용자 "23이지" → 맥락 인정 (사용자가 숫자 지목)
        """
        if not llm_hits or not user_utterance:
            return []
        cps_by_id = {cp.get("id"): cp for cp in (stage.get("checkpoints") or [])}

        # 직전 AI 발화 (맥락 검증용)
        last_ai_text = ""
        for m in reversed(self.conversation[-5:]):
            if m.get("agent_id") and m.get("content"):
                last_ai_text = m["content"]
                break

        # 사용자 짧은 긍정 신호 (응, 맞, 그치, ㅇㅇ 등)
        user_stripped = user_utterance.strip()
        user_affirm = bool(re.match(
            r"^(응|네|맞|그래|그치|그렇|좋|ㅇㅇ|어\s*맞|예|네에)",
            user_stripped[:8]
        ))
        # 사용자가 숫자 지목 (예: "23", "23, 29")
        user_has_number = bool(re.search(r"\d", user_utterance))

        verified = []
        rejected = []
        for cid in llm_hits:
            cp = cps_by_id.get(cid)
            if not cp:
                rejected.append((cid, "unknown_cp"))
                continue
            hints = cp.get("detection_hints") or []
            patterns = cp.get("detection_patterns") or []

            # (1) 사용자 단독 매칭
            user_match = (
                any(_fuzzy_match(h, user_utterance) for h in hints) or
                any(_safe_re_search(p, user_utterance) for p in patterns)
            )
            if user_match:
                verified.append(cid)
                continue

            # (2) 맥락 결합 검증
            if last_ai_text:
                ai_has_cp = (
                    any(_fuzzy_match(h, last_ai_text) for h in hints) or
                    any(_safe_re_search(p, last_ai_text) for p in patterns)
                )
                if ai_has_cp and (user_affirm or user_has_number):
                    verified.append(cid)
                    print(f"       · [llm-hit-verify] {cid} 맥락 인정 "
                          f"(AI 직전 발화 + 사용자 긍정/숫자)", flush=True)
                    continue

            rejected.append((cid, "no_match"))

        if rejected:
            print(f"       · [llm-hit-verify] 매칭 실패 → 무시: {rejected}",
                  flush=True)
        return verified

    def _next_missing_required(self, stage_num=None):
        """현재 Stage에서 사용자 미달성 필수(prio=필수) 체크포인트 첫 번째 반환.

        없으면 미달성 권장, 그것도 없으면 None.
        """
        stage_num = stage_num or self.current_stage
        stage = self.task["stages"].get(str(stage_num)) or {}
        user_prog = (
            (self.learner_models.get("user", {}).get("checkpoint_progress") or {})
            .get(str(stage_num)) or {}
        )
        prio_rank = {"필수": 0, "권장": 1, "보너스": 2}
        sorted_cps = sorted(
            stage.get("checkpoints") or [],
            key=lambda cp: (prio_rank.get(cp.get("priority"), 99),
                            cp.get("id", "")),
        )
        for cp in sorted_cps:
            if not user_prog.get(cp.get("id"), {}).get("hit"):
                return cp
        return None

    def _apply_loop_pivot(self, decision, user_utterance):
        """사용자 반복 답변/짜증 감지 시 directive를 다음 미달성 체크포인트로 강제 pivot.

        v1.31 까지의 1번 약수 하드코딩을 제거하고 일반화. 주제·숫자 무관하게:
          1. 짜증 신호("했잖아") OR 같은 답변 2회 반복 감지
          2. 다음 미달성 필수 체크포인트 식별
          3. directive `speech_goal`을 그 체크포인트의 knowledge로 향하게 override
        """
        signal = self._detect_user_repeat_or_frustration(user_utterance)
        if not signal:
            return

        next_cp = self._next_missing_required()
        if not next_cp:
            # 모든 체크포인트 hit — Stage 종료 흐름이 처리할 일
            return

        cp_id = next_cp.get("id")
        cp_know = next_cp.get("knowledge", "")
        print(f"       · [loop-pivot] 사용자 {signal['signal']} 감지 "
              f"(evidence='{signal['evidence']}') → directive를 {cp_id} pivot")

        # 발화자 선정: 진행자(서연) 우선. 단 직전 turn에 서연이 발화했으면 민준.
        last_speaker_aid = None
        for m in reversed(self.conversation[-6:]):
            if m.get("agent_id"):
                last_speaker_aid = m.get("agent_id")
                break
        speaker = "ai_2" if last_speaker_aid != "ai_2" else "ai_1"

        decision["speaking_agents"] = [speaker]
        for other in self.AI_KEYS:
            if other != speaker:
                decision[f"{other}_directive"] = None
        decision[f"{speaker}_directive"] = {
            "role": f"{'서연' if speaker == 'ai_2' else '민준'} — pivot to {cp_id}",
            "speech_goal": (
                f"사용자가 직전 답변을 충분히 이미 말했다. 짧게 인정한 뒤, "
                f"다음 미달성 체크포인트({cp_id}: {cp_know})로 자연스럽게 넘어가는 "
                f"새 질문을 1개 던진다."
            ),
            "must_include": (
                "사용자 직전 답에 대한 짧은 인정 1마디 + "
                f"{cp_id}({cp_know})를 향한 새 질문 1개"
            ),
            "must_avoid": (
                "사용자가 직전에 답한 내용 재질문 절대 금지, 정의 단언, "
                "약수 직접 나열, 30자 초과, 같은 주제 반복"
            ),
        }

    def current_stage_info(self):
        return self.task["stages"][str(self.current_stage)]

    def _keyword_match_checkpoints(self, user_utterance, stage):
        """사용자 발화에 detection_hints가 등장하면 그 체크포인트 hit (fuzzy 매칭).

        v1.32부터 strict substring 매칭 → fuzzy 정규식 자동 변환:
          - 조사 정규화: "약수가 3개" → "약수[가는이]?\\s*3\\s*개"
            → "약수가 3개", "약수는 3개", "약수 3개" 모두 매칭
          - 접속사 정규화: "1과 자기자신" → "1[과와]\\s*자기\\s*자신"
          - 공백 유연화: 한글 단어 사이 공백 0~2개 허용
          - 숫자 단위: "3개" 다음에 "야/네/잖/뿐/만" 등 종결어 무관

        LLM이 hit 판정을 놓치거나 detection_hints가 사용자 발화 변형을 못 잡는
        경우 안전장치. fuzzy 변환으로 false negative 대폭 감소.

        반환: hit된 cp_id 리스트 (중복 제거).
        """
        if not user_utterance:
            return []
        text = user_utterance
        hit_ids = []
        for cp in stage.get("checkpoints") or []:
            cid = cp.get("id")
            matched = False
            # 1. detection_patterns (선택) — 명시적 정규식
            for pat_str in cp.get("detection_patterns") or []:
                try:
                    if re.search(pat_str, text):
                        matched = True
                        break
                except re.error:
                    continue
            # 2. detection_hints (자동 fuzzy 변환)
            if not matched:
                for hint in cp.get("detection_hints") or []:
                    if not hint:
                        continue
                    if _fuzzy_match(hint, text):
                        matched = True
                        break
            if matched:
                hit_ids.append(cid)
        # 중복 제거 + 순서 보존
        seen = set()
        out = []
        for cid in hit_ids:
            if cid not in seen:
                seen.add(cid)
                out.append(cid)
        return out

    def _format_stage_checkpoints(self, stage):
        """analyze_and_decide 프롬프트에 주입할 체크포인트 목록 문자열 생성.

        LLM이 checkpoint_hits 필드에 어떤 id를 넣어야 하는지 알려주기 위해
        id/priority/knowledge + 구체 hit 조건을 한 줄씩 열거.
        """
        cps = stage.get("checkpoints") or []
        if not cps:
            return "(이 Stage에 등록된 체크포인트 없음)"
        lines = []
        for cp in cps:
            cid = cp.get("id", "?")
            prio = cp.get("priority", "")
            know = cp.get("knowledge", "")
            hints = cp.get("detection_hints") or []
            hint_examples = ", ".join(f'"{h}"' for h in hints[:5])  # 최대 5개만
            hint_part = f" | hit 단서 예: {hint_examples}" if hint_examples else ""
            lines.append(f"  - {cid} [{prio}] {know}{hint_part}")
        lines.append("")
        lines.append(
            "  ▶ hit 판정: 위 어휘·표현·숫자 중 하나라도 사용자 발화에 등장하면 그 체크포인트 hit. "
            "예: 사용자가 '약수' 단어를 쓰면 s1-1 hit. '3의 약수는 1과 3'이라고 말하면 s1-1(어휘) + s1-4(예시) 둘 다 hit."
        )
        return "\n".join(lines)

    def _domain_text(self, max_chars=DOMAIN_TRUNCATE_CHARS):
        dk = self.config.get("domain_knowledge") or {}
        text = dk.get("combined_text") or ""
        if not text:
            return "(도메인 자료가 로드되지 않음)"
        if max_chars and max_chars > 0 and len(text) > max_chars:
            return text[:max_chars] + f"\n\n(이하 생략 — 전체 {len(text):,}자 중 앞부분만)"
        return text

    def _lm_summary(self, lm):
        flat = self._flatten_lm(lm)
        out = {}
        for mk, mv in flat.items():
            sub = {}
            for dk, dv in mv.items():
                if isinstance(dv, dict):
                    sub[dk] = dv.get("post") if dv.get("post") is not None else dv.get("pre")
                else:
                    sub[dk] = dv
            out[mk] = sub
        return out

    def _flatten_lm(self, lm):
        out = {}
        for mk, mv in lm["models"].items():
            per_dim = {}
            for dk, dv in mv.items():
                if isinstance(dv, dict) and "value" in dv:
                    per_dim[dk] = dv["value"]
                elif isinstance(dv, dict) and ("pre" in dv or "post" in dv):
                    per_dim[dk] = {"pre": dv.get("pre"), "post": dv.get("post")}
                else:
                    per_dim[dk] = None
            out[mk] = per_dim
        return out

    def analyze_user_utterance(self, user_utterance):
        stage = self.current_stage_info()
        prompt = render_prompt(self.prompts["learner_analysis"], {
            "task_title": self.task["task_title"],
            "stage_title": stage["title"],
            "core_question": stage["core_question"],
            "user_utterance": user_utterance,
            "recent_dialogue": self.recent_dialogue(6),
            "current_learner_model": self._lm_summary(self.learner_models["user"]),
            "learner_model_schema": "(스키마 생략 — updates의 model/dimension 키만 정확히 기재)",
            "domain_knowledge": self._domain_text(),
            "stage_rubric": stage.get("assessment_rubric", "(해당 Stage 루브릭 없음)"),
            "stage_checklist": stage.get("ai_checklist", "(해당 Stage 체크리스트 없음)"),
        })
        # provider별 분석용 경량 모델: Gemini는 flash-lite, Anthropic은 Haiku
        fast_model = None
        _prov = getattr(self.api, "provider", None)
        if _prov == "openai":
            fast_model = DEFAULT_OPENAI_MINI
        elif _prov == "gemini":
            fast_model = DEFAULT_GEMINI_FLASH_LITE
        elif _prov == "anthropic":
            fast_model = DEFAULT_HAIKU
        try:
            raw = self.api.call(prompt, max_tokens=500, temperature=0.3,
                                model=fast_model)
            result = extract_json(raw)
        except Exception as e:
            print(f"  ⚠️ 학습자 분석 실패: {e}", flush=True)
            return {
                "updates": [],
                "misconception_changes": {"added": [], "removed": []},
                "observation_summary": "",
            }

        user_lm = self.learner_models["user"]["models"]
        for u in result.get("updates", []):
            mk, dk, nv = u["model"], u["dimension"], u["new_value"]
            if mk in user_lm and dk in user_lm[mk]:
                user_lm[mk][dk]["value"] = nv
                user_lm[mk][dk]["history"].append({
                    "stage": self.current_stage,
                    "value": nv,
                    "evidence": u.get("evidence", ""),
                })

        mis_containers = []
        if "task_achievement" in user_lm and "misconceptions" in user_lm["task_achievement"]:
            mis_containers.append(user_lm["task_achievement"]["misconceptions"])
        if "cognitive_state" in user_lm and "misconceptions" in user_lm["cognitive_state"]:
            mis_containers.append(user_lm["cognitive_state"]["misconceptions"])

        for mis in mis_containers:
            if not isinstance(mis.get("value"), list):
                mis["value"] = []
            mis.setdefault("history", [])
            for add in result.get("misconception_changes", {}).get("added", []):
                if add not in mis["value"]:
                    mis["value"].append(add)
                    mis["history"].append({
                        "stage": self.current_stage,
                        "turn": self.turn_count,
                        "event": "added",
                        "item": add,
                        "value": list(mis["value"]),
                        "evidence": "",
                    })
            for rem in result.get("misconception_changes", {}).get("removed", []):
                if rem in mis["value"]:
                    mis["value"].remove(rem)
                    mis["history"].append({
                        "stage": self.current_stage,
                        "turn": self.turn_count,
                        "event": "removed",
                        "item": rem,
                        "value": list(mis["value"]),
                        "evidence": "",
                    })
        return result

    def seconds_since_user_spoke(self) -> float:
        return time.time() - self.last_user_utterance_ts

    def _pick_silence_agent(self) -> str:
        preference = ["ai_3", "ai_1", "ai_2"]
        if self.seconds_since_user_spoke() >= SILENCE_ESCALATION_SECONDS:
            preference = ["ai_1", "ai_3", "ai_2"]
        for aid in preference:
            if aid != self.last_silence_agent:
                return aid
        return preference[0]

    def tutor_decision(self, user_silence_seconds: float = 0.0, user_mode: str = "collaborator"):
        stage = self.current_stage_info()
        silence_trigger = user_silence_seconds >= SILENCE_THRESHOLD_SECONDS
        tm = self.config["tutor_model"]["tutor_model"]
        prompt = render_prompt(self.prompts["tutor_decision"], {
            "learning_objectives": self.task["learning_objectives"],
            "user_learner_model": self._lm_summary(self.learner_models["user"]),
            "ai_1_learner_model": self._lm_summary(self.learner_models["ai_1"]),
            "ai_2_learner_model": self._lm_summary(self.learner_models["ai_2"]),
            "ai_3_learner_model": self._lm_summary(self.learner_models["ai_3"]),
            "recent_dialogue": self.recent_dialogue(8),
            "user_silence_seconds": f"{user_silence_seconds:.0f}",
            "last_silence_trigger_agent": self.last_silence_agent or "(없음)",
            "user_mode_hint": user_mode,
            "speaker_frequency": self.speaker_frequency(8),
        })
        # provider별 경량 모델 라우팅 (legacy fallback 경로도 동일하게 다이어트)
        fast_model = None
        _prov = getattr(self.api, "provider", None)
        if _prov == "openai":
            fast_model = DEFAULT_OPENAI_MINI
        elif _prov == "gemini":
            fast_model = DEFAULT_GEMINI_FLASH_LITE
        elif _prov == "anthropic":
            fast_model = DEFAULT_HAIKU
        try:
            raw = self.api.call(prompt, max_tokens=500, temperature=0.5, model=fast_model)
            decision = extract_json(raw)
        except Exception as e:
            print(f"  ⚠️ 교수자 의사결정 실패: {e}", flush=True)
            if silence_trigger:
                aid = self._pick_silence_agent()
                directive = {
                    "role": "침묵 유도자",
                    "speech_goal": "사용자가 오래 말하지 않아 부드럽게 대화를 연다",
                    "must_include": "짧은 질문 하나",
                    "must_avoid": "정답 제시, 추궁조",
                }
                decision = {
                    "diagnosis": "사용자 침묵",
                    "user_mode": user_mode,
                    "silence_trigger": True,
                    "strategy": "침묵 유도 (폴백)",
                    "speaking_agents": [aid],
                    "ai_1_directive": directive if aid == "ai_1" else None,
                    "ai_2_directive": directive if aid == "ai_2" else None,
                    "ai_3_directive": directive if aid == "ai_3" else None,
                    "pedagogical_goal": "대화 재개",
                    "stage_complete": False,
                }
            elif user_mode == "teacher":
                decision = {
                    "diagnosis": "사용자 설명자 모드",
                    "user_mode": "teacher",
                    "silence_trigger": False,
                    "strategy": "사용자 설명 존중, 연우가 학습자 모드로 짧게 반응",
                    "speaking_agents": ["ai_3"],
                    "ai_1_directive": None,
                    "ai_2_directive": None,
                    "ai_3_directive": {
                        "role": "학습자(듣기 역할) + 질문자",
                        "speech_goal": "사용자 설명에 짧게 동조하고 한 단계 더 나아가는 모르는 척 질문 1개",
                        "must_include": "사용자 언어 되짚기",
                        "must_avoid": "정답 교정, 두 개 이상 질문",
                    },
                    "pedagogical_goal": "learning by teaching 유지",
                    "stage_complete": False,
                }
            else:
                decision = {
                    "diagnosis": "일반 대화",
                    "user_mode": "collaborator",
                    "silence_trigger": False,
                    "strategy": "기본 진행 (폴백)",
                    "speaking_agents": ["ai_3", "ai_1"],
                    "ai_1_directive": {
                        "role": "개념 설명자",
                        "speech_goal": "막힌 지점 확인 후 단계적 설명",
                        "must_include": "이해 확인 질문",
                        "must_avoid": "정답 직접 말하기",
                    },
                    "ai_2_directive": None,
                    "ai_3_directive": {
                        "role": "질문자",
                        "speech_goal": "모르는 척 짧은 질문으로 사용자 설명 유도",
                        "must_include": "짧은 질문 하나",
                        "must_avoid": "아는 척, 여러 질문",
                    },
                    "pedagogical_goal": "",
                    "stage_complete": False,
                }
        if "speaking_agents" not in decision or decision["speaking_agents"] is None:
            decision["speaking_agents"] = [
                aid for aid in self.AI_KEYS
                if decision.get(f"{aid}_directive")
            ]
        self._enforce_rotation_guard(decision)
        self.last_tutor_decision = decision
        return decision

    def _build_ai_prompt(self, student_key, directive, user_utterance,
                         user_mode="collaborator", silence_trigger=False,
                         user_silence_seconds=0.0):
        persona = self.config["personas"]["ai_students"][student_key]
        # 발화 생성엔 말투·역할만 필수. JSON 덩어리를 줄여 TTFT 단축.
        persona_slim = {
            "name": persona.get("name"),
            "level": persona.get("level"),
            "role": persona.get("role"),
            "speech_style": persona.get("speech_style"),
        }
        stage = self.current_stage_info()
        directive = directive or {}

        # 이 AI가 알고 있는 체크포인트를 prompt에 주입.
        # stage['checkpoints']에서 알고 있는 id의 knowledge 문자열을 뽑아 정리.
        known_cp_lines = []
        just_learned_cp_lines = []
        try:
            from .learner_model import known_checkpoint_ids
            known_ids = set(known_checkpoint_ids(
                self.learner_models[student_key], self.current_stage
            ))
            just_learned = set(
                (getattr(self, "_recent_ai_checkpoint_gains", None) or {}).get(student_key, [])
            )
            for cp in stage.get("checkpoints") or []:
                cid = cp.get("id")
                if cid in known_ids:
                    marker = "🆕" if cid in just_learned else "•"
                    known_cp_lines.append(f"  {marker} {cid}: {cp.get('knowledge')}")
                    if cid in just_learned:
                        just_learned_cp_lines.append(f"  {cid}: {cp.get('knowledge')}")
        except Exception:
            pass

        known_cp_block = "\n".join(known_cp_lines) if known_cp_lines else "  (이 Stage에서 아직 이해한 지식이 없음)"
        just_learned_block = ("\n".join(just_learned_cp_lines)
                              if just_learned_cp_lines else "(없음)")

        # 사용자가 아직 달성 못한 다음 체크포인트 (힌트 상한선)
        # 필수(0) → 권장(1) → 보너스(2) 순, 동일 순위 내에서는 id 오름차순.
        # AI가 이 지점 이상으로 먼저 뛰어넘어 답을 흘리지 않도록 프롬프트에 명시.
        next_cp_block = "(다음 체크포인트 정보 없음)"
        try:
            user_prog = (
                (self.learner_models.get("user", {}).get("checkpoint_progress") or {})
                .get(str(self.current_stage)) or {}
            )
            prio_rank = {"필수": 0, "권장": 1, "보너스": 2}
            sorted_cps = sorted(
                stage.get("checkpoints") or [],
                key=lambda cp: (prio_rank.get(cp.get("priority"), 99), cp.get("id", "")),
            )
            next_cp = None
            for cp in sorted_cps:
                if not user_prog.get(cp.get("id"), {}).get("hit"):
                    next_cp = cp
                    break
            if next_cp:
                next_cp_block = (
                    f"  - id: {next_cp.get('id')}\n"
                    f"  - priority: {next_cp.get('priority')}\n"
                    f"  - knowledge: {next_cp.get('knowledge')}\n"
                    f"  - (이 체크포인트까지만 힌트 허용. 이후 체크포인트로 건너뛰지 말 것)"
                )
            else:
                next_cp_block = "(이 Stage의 모든 체크포인트 사용자 달성 — Stage 종료 준비)"
        except Exception:
            pass

        # 이 Stage에서 허용된 비계(사용자 요청 시 제공 가능) 목록
        scaffold_lines = []
        for sc in stage.get("allowed_scaffolds") or []:
            triggers = " / ".join(f'"{t}"' for t in (sc.get("when_user_asks") or []))
            scaffold_lines.append(
                f"  - [{sc.get('id')}] 사용자가 {triggers} 류의 요청 → "
                f"{sc.get('provide')} (유지: {sc.get('must_still')})"
            )
        allowed_scaffolds_block = ("\n".join(scaffold_lines)
                                    if scaffold_lines else "  (이 Stage에 등록된 비계 없음)")

        return render_prompt(self.prompts["ai_student"], {
            "student_name": persona["name"],
            "my_persona": persona_slim,
            # 자신의 full 학습자모델은 발화에 거의 영향 없음 → 생략해 토큰 절감
            "my_learner_state": "(간략화: 페르소나 참조)",
            "my_known_checkpoints": known_cp_block,
            "my_just_learned_checkpoints": just_learned_block,
            "user_next_uncovered_checkpoint": next_cp_block,
            "allowed_scaffolds": allowed_scaffolds_block,
            "stage_title": stage["title"],
            "stage_prompt": stage["prompt"],
            "role": directive.get("role", ""),
            "speech_goal": directive.get("speech_goal", ""),
            "must_include": directive.get("must_include", ""),
            "must_avoid": directive.get("must_avoid", ""),
            "user_mode": user_mode,
            "silence_trigger": str(bool(silence_trigger)).lower(),
            "user_silence_seconds": f"{user_silence_seconds:.0f}",
            "recent_dialogue": self.recent_dialogue(10),  # v1.40: 4→10턴 (대화 흐름 살리기)
            "user_utterance": user_utterance,
        })

    def generate_ai_utterance(self, student_key, directive, user_utterance,
                              user_mode="collaborator", silence_trigger=False,
                              user_silence_seconds=0.0):
        """LLM 호출 → sanitize → 그대로 반환. v1.35: 필터·다단 재시도 모두 제거.
        규칙은 프롬프트와 directive로만 통제. 빈 응답일 때만 짧은 generic.
        """
        prompt = self._build_ai_prompt(
            student_key, directive, user_utterance,
            user_mode=user_mode,
            silence_trigger=silence_trigger,
            user_silence_seconds=user_silence_seconds,
        )
        try:
            raw = self.api.call(prompt, max_tokens=500, temperature=1.0)
        except Exception as e:
            print(f"       · [ai_utterance {student_key}] API 호출 실패: {e}", flush=True)
            raw = ""
        text = sanitize_ai_output(raw)
        print(f"       · [ai_utterance {student_key}] raw len={len(raw or '')} "
              f"text={text[:80]!r}")
        if not text or len(text.strip()) < 3:
            print(f"       · [ai_utterance {student_key}] 빈 응답 — temperature 1.0 재시도", flush=True)
            try:
                raw = self.api.call(prompt, max_tokens=500, temperature=1.0)
                text = sanitize_ai_output(raw)
                print(f"       · [ai_utterance {student_key}] 재시도 raw len="
                      f"{len(raw or '')} text={text[:80]!r}")
            except Exception as e:
                print(f"       · [ai_utterance {student_key}] 재시도 실패: {e}", flush=True)
        if not text or len(text.strip()) < 3:
            persona_generic = {
                "ai_1": "어디가 막혔어?",
                "ai_2": "지금까지 얘기한 거 같이 정리해볼까?",
                "ai_3": "어떻게 생각해?",
            }
            text = persona_generic.get(student_key, "어떻게 생각해?")
        return text

    def analyze_and_decide(self, user_utterance):
        stage = self.current_stage_info()
        tm = self.config["tutor_model"]["tutor_model"]
        # 레이턴시 다이어트: ai 학생 학습자모델은 분석에 불필요 → 프롬프트에서 생략.
        #                 대화 히스토리도 8→4 턴으로 축소. 입력 토큰 ~40% 감소.
        prompt = render_prompt(self.prompts["analyze_and_decide"], {
            "task_title": self.task["task_title"],
            "stage_title": stage["title"],
            "core_question": stage["core_question"],
            "user_utterance": user_utterance,
            "recent_dialogue": self.recent_dialogue(8),  # v1.40: 4→8턴 (분석 흐름 살리기)
            "user_learner_model": self._lm_summary(self.learner_models["user"]),
            "ai_1_learner_model": "(생략)",
            "ai_2_learner_model": "(생략)",
            "ai_3_learner_model": "(생략)",
            "stage_rubric": stage.get("assessment_rubric", "(해당 Stage 루브릭 없음)"),
            "stage_checklist": stage.get("ai_checklist", "(해당 Stage 체크리스트 없음)"),
            "stage_checkpoints": self._format_stage_checkpoints(stage),
            "learning_objectives": self.task["learning_objectives"],
            "user_silence_seconds": "0",
            "last_silence_trigger_agent": self.last_silence_agent or "(없음)",
            "user_mode_hint": self.current_user_mode,
            "speaker_frequency": self.speaker_frequency(6),
        })
        # 분석용 경량 모델 + JSON 모드 강제 + 30초 타임아웃.
        # provider별 light 모델 라우팅 + JSON mode 지원.
        fast_model = None
        use_json_mode = False
        provider_now = getattr(self.api, "provider", None)
        if provider_now == "openai":
            # 4o-mini는 이미 light + JSON 모드 안정적
            fast_model = DEFAULT_OPENAI_MINI
            use_json_mode = True
        elif provider_now == "gemini":
            fast_model = DEFAULT_GEMINI_FLASH_LITE
            use_json_mode = True
        # anthropic은 이미 Haiku(기본)라 오버라이드 불필요

        def _invoke():
            # max_tokens=550: 레이턴시 단축. 실측상 analysis updates 1~3 +
            # cps_tags 0~2 + checkpoint_hits 0~3 + decision directive 1개로 충분.
            # 이전 700은 verbose evidence 텍스트로 부풀려져 발화 응답까지 지연.
            return self.api.call(prompt, max_tokens=550, temperature=0.4,
                                 model=fast_model, json_mode=use_json_mode)

        t0 = time.time()
        raw = None
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(_invoke)
                raw = fut.result(timeout=30)
            dt = time.time() - t0
            print(f"       · analyze_and_decide 응답 {dt:.1f}s (model={fast_model or self.api.model}) len={len(raw)}", flush=True)
            merged = extract_json(raw)
            # 진단: LLM이 반환한 키 구조 로그
            _a = merged.get("analysis") or {}
            _d = merged.get("decision") or {}
            print(f"       · [analyze keys] top={sorted(merged.keys())} analysis={sorted(_a.keys())}", flush=True)
        except Exception as e:
            dt = time.time() - t0
            print(f"  ⚠️ 통합 분석/결정 실패({dt:.1f}s): {e} — 기본 결정으로 대체 (fallback 호출 생략)", flush=True)
            # fallback 경로는 API 2회 더 호출 → 레이턴시 지옥. 그 대신 최소 결정을 즉석 구성해
            # UI가 100초 내에 반드시 움직이게 한다.
            analysis = {
                "updates": [], "misconception_changes": {"added": [], "removed": []},
                "observation_summary": "(분석 타임아웃 — 기본 결정 사용)",
            }
            last_ai = None
            for m in reversed(self.conversation[-6:]):
                if m.get("agent_id") in self.AI_KEYS:
                    last_ai = m.get("agent_id")
                    break
            rotation = {"ai_1": "ai_2", "ai_2": "ai_3", "ai_3": "ai_1"}
            next_aid = rotation.get(last_ai, "ai_1")
            decision = {
                "diagnosis": "분석 타임아웃",
                "user_mode": self.current_user_mode,
                "silence_trigger": False,
                "speaking_agents": [next_aid],
                f"{next_aid}_directive": {
                    "role": "호응자",
                    "speech_goal": "사용자 발화에 짧게 반응하며 질문을 하나 던진다",
                    "must_include": "한 문장 + 짧은 질문",
                    "must_avoid": "긴 설명, 정답 제시",
                },
                "stage_complete": False,
                "strategy": "타임아웃 기본",
            }
            self._enforce_rotation_guard(decision)
            self._cap_single_speaker(decision)  # 기본 1명 발화 강제

            # fallback 경로에서도 키워드 기반 체크포인트 hit 적용 (LLM 응답 없을 때
            # 적어도 코드 매칭이라도 작동해야 진척이 멈추지 않음).
            try:
                kw_hits = self._keyword_match_checkpoints(user_utterance, stage)
                # 일반화 검증 hits 합치기 (s1-2/s1-3 일반화 누적 사례)
                gen_hits = self._check_generalization_via_examples()
                for cid in gen_hits:
                    if cid not in kw_hits:
                        kw_hits.append(cid)
                if kw_hits:
                    from .learner_model import (
                        apply_checkpoint_hits, propagate_checkpoints_to_ai,
                    )
                    apply_checkpoint_hits(
                        self.learner_models["user"], kw_hits,
                        stage=self.current_stage, turn=self.turn_count, source="user",
                    )
                    ai_levels = {
                        aid: self.config["personas"]["ai_students"][aid].get("level", "중")
                        for aid in self.AI_KEYS
                        if aid in self.config["personas"]["ai_students"]
                    }
                    propagate_checkpoints_to_ai(
                        self.learner_models, kw_hits,
                        stage=self.current_stage, turn=self.turn_count,
                        ai_levels=ai_levels,
                    )
                    print(f"       · [checkpoint fallback-keyword] hit={kw_hits}", flush=True)
            except Exception as e:
                print(f"       · [checkpoint fallback-keyword] 실패: {e}", flush=True)

            self._stage_complete_safety_net(decision, user_utterance)
            # 사용자 반복 답변 감지 → directive 강제 pivot (stage_gate 이전)
            self._apply_loop_pivot(decision, user_utterance)
            # fallback 경로에서도 2단계 게이트 적용 (조건 충족 시 바로 advance 방지)
            self._apply_stage_gate(decision, user_utterance)
            # 사용자가 특정 AI를 지목했다면 모든 자동 규칙보다 우선 적용
            self._apply_user_addressed_override(decision, user_utterance)
            self.last_tutor_decision = decision
            return {"analysis": analysis, "decision": decision}

        analysis = merged.get("analysis") or {}
        decision = merged.get("decision") or {}

        user_lm = self.learner_models["user"]["models"]
        for u in analysis.get("updates", []):
            mk, dk, nv = u.get("model"), u.get("dimension"), u.get("new_value")
            if mk in user_lm and dk in user_lm[mk]:
                user_lm[mk][dk]["value"] = nv
                user_lm[mk][dk]["history"].append({
                    "stage": self.current_stage,
                    "value": nv,
                    "evidence": u.get("evidence", ""),
                })

        mis_containers = []
        if "task_achievement" in user_lm and "misconceptions" in user_lm["task_achievement"]:
            mis_containers.append(user_lm["task_achievement"]["misconceptions"])
        if "cognitive_state" in user_lm and "misconceptions" in user_lm["cognitive_state"]:
            mis_containers.append(user_lm["cognitive_state"]["misconceptions"])

        # v1.45: 오개념 감지 진단 로그
        mc_added = analysis.get("misconception_changes", {}).get("added", []) or []
        mc_removed = analysis.get("misconception_changes", {}).get("removed", []) or []
        if mc_added or mc_removed:
            print(f"       · [misconception] added={mc_added} removed={mc_removed}",
                  flush=True)

        for mis in mis_containers:
            if not isinstance(mis.get("value"), list):
                mis["value"] = []
            mis.setdefault("history", [])
            for add in mc_added:
                if add not in mis["value"]:
                    mis["value"].append(add)
                    mis["history"].append({
                        "stage": self.current_stage, "turn": self.turn_count,
                        "event": "added", "item": add, "value": list(mis["value"]),
                        "evidence": "",
                    })
            for rem in mc_removed:
                if rem in mis["value"]:
                    mis["value"].remove(rem)
                    mis["history"].append({
                        "stage": self.current_stage, "turn": self.turn_count,
                        "event": "removed", "item": rem, "value": list(mis["value"]),
                        "evidence": "",
                    })

        # --- 체크포인트 적용 (safety_net이 progress를 보기 전에 먼저 갱신) ---
        # 이전 버그: safety_net 호출 시점에 user progress가 아직 갱신 안 돼서
        #             stage_complete_required 판정이 항상 한 턴 늦게 적용됐음.
        # 이번 수정: 체크포인트 hits를 가장 먼저 적용 → safety_net이 최신 progress 사용.
        llm_hits_early = (
            analysis.get("checkpoint_hits")
            or merged.get("checkpoint_hits")
            or []
        )
        if isinstance(llm_hits_early, str):
            llm_hits_early = [llm_hits_early]
        llm_hits_early = [str(h).strip() for h in llm_hits_early if h]
        # v1.52: LLM이 AI 발화 키워드를 사용자 hit으로 잘못 분류하는 hallucination 방지
        llm_hits_early = self._verify_llm_hits(llm_hits_early, user_utterance, stage)
        keyword_hits_early = self._keyword_match_checkpoints(user_utterance, stage)
        # 일반화 검증 hits — 누적 사례 기반 s1-2/s1-3 추가 hit
        gen_hits_early = self._check_generalization_via_examples()
        seen_e = set(llm_hits_early)
        merged_hits = list(llm_hits_early)
        for cid in keyword_hits_early:
            if cid not in seen_e:
                seen_e.add(cid)
                merged_hits.append(cid)
        for cid in gen_hits_early:
            if cid not in seen_e:
                seen_e.add(cid)
                merged_hits.append(cid)
        print(f"       · [checkpoint pre-safety_net] llm={llm_hits_early} keyword={keyword_hits_early} gen={gen_hits_early} merged={merged_hits}", flush=True)
        if merged_hits:
            try:
                from .learner_model import (
                    apply_checkpoint_hits, propagate_checkpoints_to_ai,
                )
                apply_checkpoint_hits(
                    self.learner_models["user"], merged_hits,
                    stage=self.current_stage, turn=self.turn_count, source="user",
                )
                ai_levels_e = {
                    aid: self.config["personas"]["ai_students"][aid].get("level", "중")
                    for aid in self.AI_KEYS
                    if aid in self.config["personas"]["ai_students"]
                }
                self._recent_ai_checkpoint_gains = propagate_checkpoints_to_ai(
                    self.learner_models, merged_hits,
                    stage=self.current_stage, turn=self.turn_count,
                    ai_levels=ai_levels_e,
                )
                # 진척도 스냅샷
                user_prog = (
                    self.learner_models["user"].get("checkpoint_progress", {})
                    .get(str(self.current_stage), {})
                )
                hit_now = sorted(cid for cid, v in user_prog.items()
                                 if isinstance(v, dict) and v.get("hit"))
                print(f"       · [checkpoint stage{self.current_stage} progress] hit={hit_now}", flush=True)
            except Exception as e:
                print(f"       · [checkpoint] 적용 실패: {e}", flush=True)

        if "speaking_agents" not in decision or decision["speaking_agents"] is None:
            decision["speaking_agents"] = [
                aid for aid in self.AI_KEYS if decision.get(f"{aid}_directive")
            ]
        self._enforce_rotation_guard(decision)
        self._cap_single_speaker(decision)  # 기본 1명 발화 강제 (rotation 후 단계)
        # stage_complete 안전망: 위에서 progress 갱신 후 호출되므로 최신 상태 반영
        self._stage_complete_safety_net(decision, user_utterance)

        # --- 사용자 반복 답변 감지 → directive 강제 pivot ---
        # AI가 같은 질문을 또 만들어내는 루프 차단 (rotation/cap 이후, gate 이전)
        self._apply_loop_pivot(decision, user_utterance)

        # --- Stage 완료 2단계 게이트 (승인 확인) ---
        self._apply_stage_gate(decision, user_utterance)

        # --- 사용자 지목 오버라이드 (최종) ---
        # "민준아", "서연이한테" 등 명시적 지목 시 해당 AI 단독 발화로 강제.
        # rotation/cap/stage_gate 이후 마지막에 실행되어 사용자 의도가 이김.
        self._apply_user_addressed_override(decision, user_utterance)

        # --- CPS 태그 반영 (analyze_and_decide가 piggyback으로 태깅) ---
        cps_tags_raw = analysis.get("cps_tags") or merged.get("cps_tags") or []
        if cps_tags_raw:
            try:
                from .learner_model import apply_cps_tags
                gained = apply_cps_tags(
                    self.learner_models["user"],
                    {"tags": cps_tags_raw, "none": False},
                    stage=self.current_stage,
                    turn=self.turn_count,
                )
                if gained:
                    print(f"       · [cps] +{gained} 태그 반영 ({[t.get('dimension') for t in cps_tags_raw]})", flush=True)
            except Exception as e:
                print(f"       · [cps] 태그 반영 실패: {e}", flush=True)

        # --- self-efficacy 신호 반영 (발화 tone 기반 ±1) ---
        se_signals = (
            analysis.get("self_efficacy_delta")
            or merged.get("self_efficacy_delta")
            or []
        )
        if se_signals:
            try:
                from .learner_model import apply_self_efficacy_signal
                applied = apply_self_efficacy_signal(
                    self.learner_models["user"],
                    se_signals,
                    stage=self.current_stage,
                    turn=self.turn_count,
                )
                if applied:
                    deltas = [f"{s.get('item_id')}{'+' if s.get('delta',0)>0 else ''}{s.get('delta')}" for s in se_signals]
                    print(f"       · [se] {applied}건 반영 ({deltas})", flush=True)
            except Exception as e:
                print(f"       · [se] 신호 반영 실패: {e}", flush=True)

        # 체크포인트 적용은 위에서 이미 처리됨 (safety_net 이전).
        # _recent_ai_checkpoint_gains 가 채워졌으면 AI 발화 프롬프트에서 활용됨.

        self.last_tutor_decision = decision
        return {"analysis": analysis, "decision": decision}

    def user_turn_prep(self, user_utterance):
        print(f"[user_turn_prep] v1.37 시작 — utt={user_utterance[:40]!r}")
        silence_before = self.seconds_since_user_spoke()
        self.last_user_utterance_ts = time.time()
        self.current_user_mode = detect_user_mode(user_utterance)

        self.conversation.append({
            "speaker": "사용자",
            "content": user_utterance,
            "stage": self.current_stage,
            "mode": self.current_user_mode,
        })
        self.turn_count += 1
        self.stage_turn_count += 1

        print(f"  [mode] user_mode={self.current_user_mode} · 직전 침묵={silence_before:.0f}s", flush=True)
        print("  [1/2] 🔍🎓 통합 분석/결정 중 (1회 왕복)...", flush=True)
        combined = self.analyze_and_decide(user_utterance)
        analysis = combined["analysis"]
        decision = combined["decision"]
        if analysis.get("observation_summary"):
            print(f"       · 관찰: {analysis['observation_summary']}", flush=True)
        print(f"       · 전략: {decision.get('strategy', '')}", flush=True)
        print(f"       · speaking_agents: {decision.get('speaking_agents', [])}", flush=True)

        return {
            "analysis": analysis,
            "decision": decision,
            "user_mode": self.current_user_mode,
        }

    def stream_ai_turns(self, user_utterance, decision, silence_trigger=False,
                        user_silence_seconds=0.0):
        ai_students = self.config["personas"]["ai_students"]
        speaking = decision.get("speaking_agents", []) or []

        targets = [
            aid for aid in self.AI_KEYS
            if aid in speaking and decision.get(f"{aid}_directive")
        ]
        if not targets:
            return

        print(f"  [3/3] 💬 {len(targets)}명 병렬 발화 생성 (완료순 스트리밍)...", flush=True)

        def _call(aid):
            directive = decision.get(f"{aid}_directive")
            try:
                text = self.generate_ai_utterance(
                    aid, directive, user_utterance,
                    user_mode=self.current_user_mode,
                    silence_trigger=silence_trigger,
                    user_silence_seconds=user_silence_seconds,
                )
            except Exception as e:
                text = f"(발화 생성 실패: {e})"
            return aid, text

        with ThreadPoolExecutor(max_workers=max(1, len(targets))) as ex:
            futures = {ex.submit(_call, aid): aid for aid in targets}
            for fut in as_completed(futures):
                aid, text = fut.result()
                self.conversation.append({
                    "speaker": ai_students[aid]["name"],
                    "content": text,
                    "stage": self.current_stage,
                    "agent_id": aid,
                })
                yield aid, text

    def stream_ai_turns_tokens(self, user_utterance, decision, silence_trigger=False,
                               user_silence_seconds=0.0):
        """발화할 AI들을 병렬로 스트리밍 호출하고, 토큰이 도착할 때마다 이벤트 yield."""
        ai_students = self.config["personas"]["ai_students"]
        speaking = decision.get("speaking_agents", []) or []
        targets = [
            aid for aid in self.AI_KEYS
            if aid in speaking and decision.get(f"{aid}_directive")
        ]
        if not targets:
            return

        print(f"  [3/3] 💬 {len(targets)}명 병렬 토큰 스트리밍...", flush=True)

        q = queue.Queue()

        def _worker(aid):
            directive = decision.get(f"{aid}_directive")
            prompt = self._build_ai_prompt(
                aid, directive, user_utterance,
                user_mode=self.current_user_mode,
                silence_trigger=silence_trigger,
                user_silence_seconds=user_silence_seconds,
            )
            started = False
            buf = []
            try:
                # v1.44: temp=1.0 통일 (사용자 요청)
                stream = self.api.call(
                    prompt, max_tokens=500, temperature=1.0, stream=True,
                )
                for chunk in stream:
                    if not chunk:
                        continue
                    if not started:
                        started = True
                        q.put(("start", aid, ""))
                    buf.append(chunk)
                    q.put(("update", aid, chunk))
                full = sanitize_ai_output("".join(buf))
                print(f"       · [ai_stream {aid}] raw chunks={len(buf)} "
                      f"text={full[:80]!r}")
                # v1.36: 스트리밍 빈 응답 시 non-streaming으로 한 번 재시도
                if not full or len(full.strip()) < 3:
                    print(f"       · [ai_stream {aid}] 빈 스트림 — non-streaming 재시도", flush=True)
                    try:
                        raw_ns = self.api.call(prompt, max_tokens=500,
                                                temperature=1.0, stream=False)
                        full = sanitize_ai_output(raw_ns)
                        print(f"       · [ai_stream {aid}] non-stream raw len="
                              f"{len(raw_ns or '')} text={full[:80]!r}")
                    except Exception as e:
                        print(f"       · [ai_stream {aid}] non-stream 재시도 실패: {e}", flush=True)
                # 그래도 빈 응답이면 짧은 generic
                if not full or len(full.strip()) < 3:
                    persona_generic = {
                        "ai_1": "어디가 막혔어?",
                        "ai_2": "지금까지 얘기한 거 같이 정리해볼까?",
                        "ai_3": "어떻게 생각해?",
                    }
                    full = persona_generic.get(aid, "어떻게 생각해?")
                if not started:
                    q.put(("start", aid, ""))
                q.put(("done", aid, full))
            except Exception as e:
                err = f"(발화 생성 실패: {e})"
                if not started:
                    q.put(("start", aid, ""))
                q.put(("update", aid, err))
                q.put(("done", aid, err))
            finally:
                q.put(("__worker_end__", aid, None))

        threads = []
        for aid in targets:
            t = threading.Thread(target=_worker, args=(aid,), daemon=True)
            t.start()
            threads.append(t)

        remaining = len(targets)
        while remaining > 0:
            ev, aid, payload = q.get()
            if ev == "__worker_end__":
                remaining -= 1
                continue
            if ev == "done":
                self.conversation.append({
                    "speaker": ai_students[aid]["name"],
                    "content": payload,
                    "stage": self.current_stage,
                    "agent_id": aid,
                })
            yield (ev, aid, payload)

        for t in threads:
            t.join(timeout=0.1)

    def user_turn(self, user_utterance):
        prep = self.user_turn_prep(user_utterance)
        decision = prep["decision"]
        outputs = {"ai_1": None, "ai_2": None, "ai_3": None}
        for aid, text in self.stream_ai_turns(
            user_utterance, decision,
            silence_trigger=decision.get("silence_trigger", False),
        ):
            outputs[aid] = text
        return {
            "analysis": prep["analysis"],
            "decision": decision,
            "user_mode": prep["user_mode"],
            **outputs,
        }

    def nudge_on_silence(self, min_seconds: float = SILENCE_THRESHOLD_SECONDS):
        silence = self.seconds_since_user_spoke()
        if silence < min_seconds:
            return None

        print(f"  [silence] 사용자 침묵 {silence:.0f}s 감지 → AI 선제 발화", flush=True)
        decision = self.tutor_decision(user_silence_seconds=silence,
                                       user_mode=self.current_user_mode)

        speaking = decision.get("speaking_agents") or []
        aid = None
        for s in speaking:
            if s in self.AI_KEYS and decision.get(f"{s}_directive"):
                aid = s
                break
        if aid is None:
            aid = self._pick_silence_agent()
            directive = {
                "role": "침묵 유도자",
                "speech_goal": "사용자가 오래 말하지 않아 부드럽게 대화를 연다",
                "must_include": "짧은 질문 하나",
                "must_avoid": "정답 제시, 추궁조",
            }
            decision.setdefault("silence_trigger", True)
            decision[f"{aid}_directive"] = directive
            decision["speaking_agents"] = [aid]

        directive = decision[f"{aid}_directive"]
        text = self.generate_ai_utterance(
            aid, directive, user_utterance="",
            user_mode=self.current_user_mode,
            silence_trigger=True,
            user_silence_seconds=silence,
        )
        ai_students = self.config["personas"]["ai_students"]
        self.conversation.append({
            "speaker": ai_students[aid]["name"],
            "content": text,
            "stage": self.current_stage,
            "agent_id": aid,
            "silence_nudge": True,
            "silence_seconds": round(silence, 1),
        })
        self.last_silence_agent = aid
        # AI가 선제 발화한 시점을 "사용자 발화"로 간주해 타이머를 완전 리셋.
        self.last_user_utterance_ts = time.time()
        return {"agent_id": aid, "text": text, "decision": decision}

    def advance_stage(self):
        total_stages = len(self.task["stages"])
        if self.current_stage < total_stages:
            self.current_stage += 1
            self.stage_turn_count = 0
            return True
        return False

    def inspect_ai_turn(self, user_utterance, agent_id="ai_1",
                        run_analyze=True, dump=True):
        """디버그용: 발화 1회의 prompt와 LLM raw 응답을 dict로 반환 + 출력.

        사용 예시 (Colab):
            from lib import CollaborativeSession
            s = CollaborativeSession(ctx['config'], ctx['prompts'],
                                      ctx['learner_models'], ctx['api'])
            r = s.inspect_ai_turn("소수가 뭐야?", agent_id="ai_1")
            # dump=True면 자동 출력. False면 r['prompt'], r['raw'] 직접 확인.

        Args:
            user_utterance: 시뮬할 사용자 발화
            agent_id: 분석할 AI ("ai_1"=민준, "ai_2"=서연, "ai_3"=연우)
            run_analyze: True면 analyze_and_decide LLM 호출 (directive 생성).
                         False면 빈 directive로 발화만.
            dump: True면 콘솔에 보기 좋게 출력.

        Returns:
            dict {
                user_utterance, agent_id,
                analysis, decision_strategy, decision_speakers, directive,
                prompt, prompt_len,
                raw, raw_len,
                sanitized,
            }
        """
        analysis = None
        decision = {}
        directive = {}

        if run_analyze:
            try:
                prep = self.user_turn_prep(user_utterance)
                analysis = prep.get("analysis")
                decision = prep.get("decision") or {}
                directive = decision.get(f"{agent_id}_directive") or {}
            except Exception as e:
                print(f"[inspect] analyze 실패: {e}", flush=True)

        # AI 발화 prompt 빌드
        prompt = self._build_ai_prompt(
            agent_id, directive, user_utterance,
            user_mode=self.current_user_mode,
        )

        # LLM 호출 (non-stream, 발화와 동일한 파라미터)
        try:
            raw = self.api.call(prompt, max_tokens=500, temperature=1.0)
        except Exception as e:
            raw = f"(LLM 호출 실패: {e})"

        sanitized = sanitize_ai_output(raw)

        result = {
            "user_utterance": user_utterance,
            "agent_id": agent_id,
            "analysis": analysis,
            "decision_strategy": decision.get("strategy"),
            "decision_speakers": decision.get("speaking_agents"),
            "directive": directive,
            "prompt": prompt,
            "prompt_len": len(prompt),
            "raw": raw,
            "raw_len": len(raw or ""),
            "sanitized": sanitized,
        }

        if dump:
            sep = "=" * 70
            print(sep)
            print(f"[inspect_ai_turn] agent={agent_id}  user_utt={user_utterance!r}")
            print(sep)
            print(f"DECISION  strategy={result['decision_strategy']!r}  "
                  f"speakers={result['decision_speakers']}")
            print(f"DIRECTIVE:")
            for k, v in (directive or {}).items():
                print(f"  {k}: {v}")
            print(sep)
            print(f"PROMPT ({result['prompt_len']}자) — LLM에 보낸 내용:")
            print(prompt)
            print(sep)
            print(f"RAW LLM RESPONSE ({result['raw_len']}자):")
            print(raw)
            print(sep)
            print(f"SANITIZED (사용자에게 보일 발화):")
            print(sanitized)
            print(sep)

        return result

    def dump_stage_state(self):
        """Stage 진행 상태를 진단용으로 dump.

        Colab thread print가 안 보이는 환경에서 사용자가 새 셀에 직접 호출:
            ctx['config']['_session'].dump_stage_state()
        """
        import json
        stage = self.current_stage_info()
        sk = str(self.current_stage)
        user_prog = (
            (self.learner_models.get("user", {}).get("checkpoint_progress") or {})
            .get(sk) or {}
        )
        required = stage.get("stage_complete_required") or []
        hit_ids = sorted(cid for cid, v in user_prog.items()
                          if isinstance(v, dict) and v.get("hit"))
        missing = [cid for cid in required if cid not in hit_ids]

        print(f"== current_stage: {self.current_stage} ==")
        print(f"   turn_count: {self.turn_count}")
        print(f"   pending_stage_complete: {self.pending_stage_complete}")
        print(f"   pending_since_turn: {self.pending_stage_complete_since_turn}")
        print(f"   stage_complete_required: {required}")
        print(f"   hit_ids: {hit_ids}")
        print(f"   missing: {missing}")
        print()
        print("== last_tutor_decision ==")
        last = getattr(self, "last_tutor_decision", None) or {}
        print(json.dumps(last, ensure_ascii=False, indent=2)[:2000])
        print()
        print("== conversation 마지막 6개 ==")
        for m in self.conversation[-6:]:
            spk = m.get("speaker", "")
            content = (m.get("content") or "")[:100]
            print(f"   [{spk}] {content}")
        print()
        print("== stage 1 completion_quiz 정의 ==")
        quiz = stage.get("completion_quiz")
        if quiz:
            print(f"   question: {quiz.get('question', '')[:80]}")
            print(f"   correct keywords: {len(quiz.get('correct_answers_keywords', []))}개")
        else:
            print("   (없음 — legacy consent 방식)")
        return {
            "stage": self.current_stage,
            "pending_stage_complete": self.pending_stage_complete,
            "required": required,
            "hit_ids": hit_ids,
            "missing": missing,
            "last_decision": last,
        }

    def get_stage_intro_message(self):
        """현재 stage의 intro_message (시스템 정의 안내) 반환. 없으면 None.

        gradio_app가 stage 진입 시 별도 시스템 버블로 띄울 용도.
        학생 발화로는 사용하지 않는다.
        """
        stage = self.current_stage_info()
        return stage.get("intro_message")

    def stage_intro_utterance(self, opener_key="ai_1"):
        # opener: 기본은 민준(ai_1). gradio_app에서는 Stage 2부터 서연(ai_2)을 명시 지정함.
        # 의도적으로 opener를 바꾸면 각 Stage 시작 지점에서 발화자가 다양해진다.
        persona = self.config["personas"]["ai_students"][opener_key]
        stage = self.current_stage_info()

        # v1.42: intro_message는 학생 발화로 안 씀. gradio_app가 별도 시스템 버블로 띄움.
        # (get_stage_intro_message()로 호출자에게 노출)

        prompt = render_prompt(self.prompts["stage_intro"], {
            "opener_name": persona["name"],
            "stage_title": stage["title"],
            "stage_prompt": stage["prompt"],
            "core_question": stage["core_question"],
            "my_persona": persona,
            "domain_knowledge": self._domain_text(),
        })
        # stage_intro: max_tokens 넉넉하게 + 진단 로그로 truncation 원인 추적
        raw = self.api.call(prompt, max_tokens=600, temperature=0.8)
        print(f"       · [stage_intro raw len={len(raw)}] {raw[:120]!r}...", flush=True)
        text = sanitize_ai_output(raw)

        # 불완전(길이 짧음 or 문장 미종결)이면 재시도 (모듈 레벨 헬퍼 사용)
        _is_incomplete = _is_incomplete_utterance
        attempt = 1
        while _is_incomplete(text) and attempt <= 2:
            print(f"       · [stage_intro] 불완전({len(text)}자, '{text[-10:] if text else ''}') — 재시도 #{attempt}", flush=True)
            retry_prompt = render_prompt(self.prompts["stage_intro"], {
                "opener_name": persona["name"],
                # 첫 재시도: 제목 교체. 두 번째: 제목 더 일반화
                "stage_title": ("새로운 단계" if attempt == 1 else "다음 활동"),
                "stage_prompt": stage["prompt"],
                "core_question": stage["core_question"],
                "my_persona": persona,
                "domain_knowledge": "",
            })
            # 재시도마다 temperature 살짝 올려 다른 출력 유도
            raw = self.api.call(retry_prompt, max_tokens=600, temperature=0.9 + 0.05 * attempt)
            print(f"       · [stage_intro retry#{attempt} len={len(raw)}] {raw[:120]!r}...", flush=True)
            text = sanitize_ai_output(raw)
            attempt += 1

        # 그래도 불완전하면 하드코딩 fallback
        if _is_incomplete(text):
            stage_num = self.current_stage
            fallback_by_stage = {
                1: ("자, 이번엔 분류 문제야. 2부터 20까지의 수가 두 그룹으로 나눠져 있는데, "
                    "왼쪽: 2, 3, 5, 7, 11, 13, 17, 19 / 오른쪽: 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20. "
                    "이 두 그룹은 어떤 기준으로 나눠진 걸까? 다들 어떻게 생각해?"),
                2: ("이번엔 새로운 단계야. 20부터 30까지의 수 중에 소수를 모두 찾아보자. "
                    "어디서부터 시작해볼까?"),
                3: ("이번엔 참/거짓 판단이야. (1) '소수는 모두 홀수이다.' (2) '5의 배수는 모두 합성수이다.' "
                    "둘 다 참인지 거짓인지 같이 생각해보자."),
            }
            text = fallback_by_stage.get(stage_num,
                "자, 이번에는 새로운 걸 같이 얘기해볼까? 다들 어떻게 생각해?")
            print(f"       · [stage_intro] 하드코딩 fallback 적용 (stage {stage_num})", flush=True)

        print(f"       · [stage_intro final len={len(text)}] {text[:120]!r}...", flush=True)
        self.conversation.append({
            "speaker": persona["name"],
            "content": text,
            "stage": self.current_stage,
        })
        return text


# ============================================================
# 파일 끝 — mount sync 강제 플러시용 여분 주석
# 이 파일은 CollaborativeSession 클래스만 export한다.
# 수정 후 bash에서 AST parse 시 truncation이 보이면 이 구역을 한 줄 추가/삭제해
# Windows 쪽 flush를 유도하자.
# ============================================================
