"""협력학습 세션 파이프라인.

3단계 구조: 학습자 분석 → 교수자 의사결정 → AI 학생 발화.
전역 변수를 쓰지 않고 config/prompts/learner_models/api 를 명시적으로 주입 받는다.

확장 기능:
- AI 학생 3명 지원 (ai_1 민준·상, ai_2 서연·중, ai_3 연우·하)
- 침묵 유도: 사용자가 60초 이상 말하지 않으면 nudge_on_silence()로 한 명의 AI가 선제 발화
- 사용자 교수자 모드: 사용자가 설명자 모드가 되면 AI는 학습자 모드로 전환해 짧게 반응
"""

import time
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


def _is_incomplete_utterance(s: str) -> bool:
    """AI 발화가 중간에 잘렸는지 휴리스틱 판정.

    규칙:
      - 빈 문자열 또는 strip 후 20자 미만 → incomplete
      - 문장 부호(. ! ? …)로 끝나면 완결로 신뢰
      - 그 외에는 **다중 음절 종결패턴**만 완결로 인정 (단일 한글 음절은 모호해서 제외)
        · '어'는 '어때/어떨까/어떻게' 시작 / '자'는 '자기/자신' 시작 등, 단일 음절 종결은
          절단된 다음 음절과 구분 불가능해 false negative 빈발.
        · 예: "숫자 7은 어" → 완결 아님 (다음에 '어때?'가 올 수도).

    Gemini의 RECITATION 조기 종료로 인한 잘림 케이스 대응.
    """
    if not s:
        return True
    stripped = s.strip().rstrip('"\'」』')
    if len(stripped) < 20:
        return True
    # 1) 명시적 문장부호 종결 → 완결
    if stripped[-1] in ".!?…":
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

        # HARD-2: 서연이 최근 6턴 중 2회 이하 & 이번 턴 미포함 → 강제 추가
        if recent_all.count("ai_2") <= 2 and "ai_2" not in speakers:
            # 이미 두 명이 잡혔다면 덜 최근에 더 많이 말한 쪽을 서연으로 교체
            if len(speakers) >= 2:
                counts6 = {a: recent_all.count(a) for a in speakers}
                victim = max(counts6, key=counts6.get) if counts6 else speakers[-1]
                speakers.remove(victim)
                decision[f"{victim}_directive"] = None
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
            print(f"       · [rotation-guard] 교정: {speakers_in} → {deduped}")

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

        # main + comp 조합이면 2명 유지 허용
        if "main" in tagset and "comp" in tagset and len(speakers) == 2:
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
        """LLM이 stage_complete=false로 내려도, 학습자 발화(+최근 합산)가
        명백히 완료 기준을 만족하면 true로 덮어쓴다.

        현재 Stage의 번호에 따라 키워드 세트를 검사한다.
        - Stage 1: 경로 A(약수 개수) 또는 경로 B(나눗셈 표현) 판정
        - Stage 2: 31·37·41·43·47 중 4개 이상 등장 + 판정 근거 단서
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

        stage = self.current_stage
        hit_reason = None

        if stage == 1:
            # 관대한 통과 기준: 아래 세 경로 중 **하나만** 언급해도 통과.
            has_div = "약수" in combined

            # 경로 A: 약수가 2개 (소수 쪽 단서)
            path_a = has_div and any(k in combined for k in [
                "약수가 2개", "약수가 2 개", "약수가 두개", "약수가 두 개",
                "약수 2개", "약수 두개", "2개야", "두개야",
                "정확히 2개", "딱 2개", "딱 두개",
            ])
            if path_a:
                hit_reason = "경로A(약수 2개)"

            # 경로 B: 1과 자기 자신만 약수 (소수 정의)
            path_b = any(k in combined for k in [
                "1과 자기 자신", "1이랑 자기 자신", "1과 자신",
                "1, 자기 자신", "1이랑 자기자신", "1과 본인",
                "자기 자신만", "자기자신만",
            ])
            if path_b and not hit_reason:
                hit_reason = "경로B(1과 자기자신 정의)"

            # 경로 C: 약수가 3개 이상 (합성수 쪽 단서)
            path_c = has_div and any(k in combined for k in [
                "3개 이상", "3 개 이상", "세개 이상", "세 개 이상",
                "2개보다 많", "두개보다 많", "약수가 많",
                "4개", "5개", "6개",
            ])
            if path_c and not hit_reason:
                hit_reason = "경로C(약수 3개 이상)"

            # 경로 D: 1·자기자신 외 다른 수로도 나눠진다 (합성수 쪽 단서)
            path_d = any(k in combined for k in [
                "다른 수로", "다른 약수", "1과 자기자신 이외",
                "1과 자기 자신 이외", "본인 이외", "자기자신 이외",
                "2로도", "3으로도", "다른 수도",
            ])
            if path_d and not hit_reason:
                hit_reason = "경로D(1·자기자신 외 약수 존재)"

        elif stage == 2:
            # 범위 20~30으로 교체. 정답: 23, 29 (둘 다 필요)
            has_23 = "23" in combined
            has_29 = "29" in combined
            # 합성수 오답 포함 여부 — 하나라도 있으면 아직 미완료로 간주
            composites = ["20", "21", "22", "24", "25", "26", "27", "28", "30"]
            has_wrong = False
            for c in composites:
                # "23"을 "25"로 오판한 경우는 has_29 체크로 잡힘. 여기선
                # 명시적으로 "<c>는 소수" 류 표현을 추적.
                if f"{c}은 소수" in combined or f"{c}는 소수" in combined or f"{c}이 소수" in combined:
                    has_wrong = True
                    break
            if has_23 and has_29 and not has_wrong:
                hit_reason = "Stage2 소수 23·29 둘 다 정확"

        elif stage == 3:
            # Stage 3: 12 + 13 = 25. 정답 25가 나오거나 (12와 13) 모두 등장 → 완료.
            has_12 = "12" in combined
            has_13 = "13" in combined
            has_25 = any(k in combined for k in ["25", "이십오"])
            # "합은 25"나 "답은 25" 같은 명시적 계산도 체크
            explicit_sum = any(k in combined for k in [
                "합은 25", "합이 25", "답은 25", "답이 25",
                "12 + 13", "12+13", "12더하기 13", "12 더하기 13",
            ])
            if explicit_sum:
                hit_reason = "Stage3 12+13=25 명시적 계산"
            elif has_12 and has_13 and has_25:
                hit_reason = "Stage3 12·13·25 모두 등장"

        if hit_reason:
            decision["stage_complete"] = True
            print(f"       · [stage-guard] 완료 기준 충족 감지 → stage_complete=true "
                  f"(reason: {hit_reason})")

    def _apply_stage_gate(self, decision, user_utterance):
        """Stage 완료 2단계 게이트: 조건 충족 → 서연 확인 질문 → 사용자 승인.

        - pending 상태에서 사용자가 consent면 실제 advance, reject면 pending 해제
        - 조건이 이번 턴에 처음 충족되면 pending으로 전환하고 서연 단독 확인 질문 directive 주입
        - pending 2턴 경과하면 자동 해제 (stuck 방지)

        main flow·fallback 양쪽에서 호출되어야 함.
        """
        consent = _detect_stage_advance_consent(user_utterance)
        if self.pending_stage_complete:
            if consent == "consent":
                print(f"       · [stage-gate] 사용자 승인 감지 → stage_complete=True 확정")
                decision["stage_complete"] = True
                self.pending_stage_complete = False
                self.pending_stage_complete_since_turn = None
            elif consent == "reject":
                print(f"       · [stage-gate] 사용자 거부 감지 → pending 해제, Stage 유지")
                decision["stage_complete"] = False
                self.pending_stage_complete = False
                self.pending_stage_complete_since_turn = None
            else:
                held_turns = self.turn_count - (self.pending_stage_complete_since_turn or self.turn_count)
                if held_turns >= 2:
                    print(f"       · [stage-gate] pending 2턴 경과 → 해제, Stage 유지")
                    self.pending_stage_complete = False
                    self.pending_stage_complete_since_turn = None
                decision["stage_complete"] = False  # pending 중엔 advance 막음
        elif decision.get("stage_complete"):
            # 최초로 완료 조건 충족 감지 → 승인 질문 모드로 전환
            print(f"       · [stage-gate] Stage {self.current_stage} 완료 조건 충족 — 서연 확인 질문 삽입")
            self.pending_stage_complete = True
            self.pending_stage_complete_since_turn = self.turn_count
            decision["stage_complete"] = False  # 바로 advance 막기
            # 서연(ai_2) 혼자 발화하도록 강제 + 확인 directive 주입
            decision["speaking_agents"] = ["ai_2"]
            decision["ai_1_directive"] = None
            decision["ai_3_directive"] = None
            decision["ai_2_directive"] = {
                "role": "진행자 (Stage 완료 승인 확인)",
                "speech_goal": (
                    "지금까지 사용자가 한 설명을 한 줄로 짧게 요약해 재진술한 뒤, "
                    "'이렇게 이해하고 다음 단계로 넘어가도 될까?' 라고 분명하게 승인을 묻는다."
                ),
                "must_include": "사용자가 방금 말한 핵심 표현을 그대로 짧게 인용 + 승인 질문 (예: '~ 이렇게 정리해도 돼? 다음으로 넘어가도 될까?')",
                "must_avoid": "새로운 개념 도입, 긴 설명, 두 개 이상의 질문",
            }

    def current_stage_info(self):
        return self.task["stages"][str(self.current_stage)]

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
            print(f"  ⚠️ 학습자 분석 실패: {e}")
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
            print(f"  ⚠️ 교수자 의사결정 실패: {e}")
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
            "recent_dialogue": self.recent_dialogue(4),  # 8→4 턴
            "user_utterance": user_utterance,
        })

    def generate_ai_utterance(self, student_key, directive, user_utterance,
                              user_mode="collaborator", silence_trigger=False,
                              user_silence_seconds=0.0):
        prompt = self._build_ai_prompt(
            student_key, directive, user_utterance,
            user_mode=user_mode,
            silence_trigger=silence_trigger,
            user_silence_seconds=user_silence_seconds,
        )
        # max_tokens=480: 한글 2~4문장이 잘리지 않게 여유 있게 설정
        raw = self.api.call(prompt, max_tokens=480, temperature=0.9)
        text = sanitize_ai_output(raw)

        # 미완결이면 최대 2회 재시도
        if _is_incomplete_utterance(text):
            print(f"       · [ai_utterance {student_key}] 미완결({len(text)}자, '{text[-10:] if text else ''}') — 1차 재시도")
            retry_note_1 = (
                "\n\n---\n[재생성]: 직전 응답이 중단됐다. 아래를 엄수:\n"
                "- 따옴표로 수학 용어를 감싸지 말 것.\n"
                "- 교과서 문장을 그대로 인용하지 말고 학생 말투로 풀어쓸 것.\n"
                "- 문장 종결 어미(~야/어/지/까/래/해)로 끝맺을 것.\n"
            )
            text2 = ""
            try:
                raw2 = self.api.call(prompt + retry_note_1, max_tokens=480, temperature=1.0)
                text2 = sanitize_ai_output(raw2)
            except Exception as e:
                print(f"       · [ai_utterance {student_key}] 1차 실패: {e}")

            text3 = ""
            if _is_incomplete_utterance(text2):
                print(f"       · [ai_utterance {student_key}] 1차도 미완결 — 2차 재시도(용어 최소화)")
                retry_note_2 = (
                    "\n\n---\n[2차 재생성]: 수학 용어 최소화. 2문장 이내 짧게. "
                    "학생 말투로 되묻는 질문 하나로 끝내기.\n"
                )
                try:
                    raw3 = self.api.call(prompt + retry_note_2, max_tokens=300, temperature=1.2)
                    text3 = sanitize_ai_output(raw3)
                except Exception as e:
                    print(f"       · [ai_utterance {student_key}] 2차 실패: {e}")

            # 최선 선택
            best = None
            for cand in (text3, text2, text):
                if cand and not _is_incomplete_utterance(cand):
                    best = cand
                    break
            if best is None:
                partial = text3 or text2 or text or ""
                if partial:
                    if partial.rstrip()[-1:] in ".!?…":
                        best = partial.rstrip()
                    else:
                        best = partial.rstrip() + "..."
                else:
                    generic = {
                        "ai_1": "음, 잠깐 다시 볼까? 어디서 막혔어?",
                        "ai_2": "잠깐, 지금까지 얘기한 거 정리해볼래?",
                        "ai_3": "어… 나는 잘 모르겠는데, 같이 볼래?",
                    }
                    best = generic.get(student_key, "잠깐, 다시 볼까?")
            text = best
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
            "recent_dialogue": self.recent_dialogue(4),
            "user_learner_model": self._lm_summary(self.learner_models["user"]),
            "ai_1_learner_model": "(생략)",
            "ai_2_learner_model": "(생략)",
            "ai_3_learner_model": "(생략)",
            "stage_rubric": stage.get("assessment_rubric", "(해당 Stage 루브릭 없음)"),
            "stage_checklist": stage.get("ai_checklist", "(해당 Stage 체크리스트 없음)"),
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
            return self.api.call(prompt, max_tokens=400, temperature=0.4,
                                 model=fast_model, json_mode=use_json_mode)

        t0 = time.time()
        raw = None
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(_invoke)
                raw = fut.result(timeout=30)
            dt = time.time() - t0
            print(f"       · analyze_and_decide 응답 {dt:.1f}s (model={fast_model or self.api.model})")
            merged = extract_json(raw)
        except Exception as e:
            dt = time.time() - t0
            print(f"  ⚠️ 통합 분석/결정 실패({dt:.1f}s): {e} — 기본 결정으로 대체 (fallback 호출 생략)")
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
            self._stage_complete_safety_net(decision, user_utterance)
            # fallback 경로에서도 2단계 게이트 적용 (조건 충족 시 바로 advance 방지)
            self._apply_stage_gate(decision, user_utterance)
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
        for mis in mis_containers:
            if not isinstance(mis.get("value"), list):
                mis["value"] = []
            mis.setdefault("history", [])
            for add in analysis.get("misconception_changes", {}).get("added", []):
                if add not in mis["value"]:
                    mis["value"].append(add)
                    mis["history"].append({
                        "stage": self.current_stage, "turn": self.turn_count,
                        "event": "added", "item": add, "value": list(mis["value"]),
                        "evidence": "",
                    })
            for rem in analysis.get("misconception_changes", {}).get("removed", []):
                if rem in mis["value"]:
                    mis["value"].remove(rem)
                    mis["history"].append({
                        "stage": self.current_stage, "turn": self.turn_count,
                        "event": "removed", "item": rem, "value": list(mis["value"]),
                        "evidence": "",
                    })

        if "speaking_agents" not in decision or decision["speaking_agents"] is None:
            decision["speaking_agents"] = [
                aid for aid in self.AI_KEYS if decision.get(f"{aid}_directive")
            ]
        self._enforce_rotation_guard(decision)
        self._cap_single_speaker(decision)  # 기본 1명 발화 강제 (rotation 후 단계)
        # stage_complete 안전망: LLM이 false여도 명시적 완료 기준을 충족하면 true로 교정
        self._stage_complete_safety_net(decision, user_utterance)

        # --- Stage 완료 2단계 게이트 (승인 확인) ---
        self._apply_stage_gate(decision, user_utterance)

        # --- CPS 태그 반영 (analyze_and_decide가 piggyback으로 태깅) ---
        cps_tags_raw = analysis.get("cps_tags") or []
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
                    print(f"       · [cps] +{gained} 태그 반영 ({[t.get('dimension') for t in cps_tags_raw]})")
            except Exception as e:
                print(f"       · [cps] 태그 반영 실패: {e}")

        # --- self-efficacy 신호 반영 (발화 tone 기반 ±1) ---
        se_signals = analysis.get("self_efficacy_delta") or []
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
                    print(f"       · [se] {applied}건 반영 ({deltas})")
            except Exception as e:
                print(f"       · [se] 신호 반영 실패: {e}")

        # --- 체크포인트 적용 + AI 전파 ---
        checkpoint_hits = analysis.get("checkpoint_hits") or []
        if checkpoint_hits:
            try:
                from .learner_model import (
                    apply_checkpoint_hits, propagate_checkpoints_to_ai,
                )
                new_user = apply_checkpoint_hits(
                    self.learner_models["user"], checkpoint_hits,
                    stage=self.current_stage, turn=self.turn_count, source="user",
                )
                if new_user:
                    print(f"       · [checkpoint] user +{new_user}: {checkpoint_hits}")

                ai_levels = {
                    aid: self.config["personas"]["ai_students"][aid].get("level", "중")
                    for aid in self.AI_KEYS
                    if aid in self.config["personas"]["ai_students"]
                }
                newly_learned = propagate_checkpoints_to_ai(
                    self.learner_models, checkpoint_hits,
                    stage=self.current_stage, turn=self.turn_count,
                    ai_levels=ai_levels,
                )
                # 방금 학습한 AI가 있으면 전역 기록 + 로그 (다음 발화에서 활용)
                self._recent_ai_checkpoint_gains = newly_learned
                for aid, cps in newly_learned.items():
                    if cps:
                        name = self.config["personas"]["ai_students"][aid].get("name", aid)
                        print(f"       · [checkpoint→AI] {name}({ai_levels.get(aid)}) 학습: {cps}")
            except Exception as e:
                print(f"       · [checkpoint] 반영 실패: {e}")

        self.last_tutor_decision = decision
        return {"analysis": analysis, "decision": decision}

    def user_turn_prep(self, user_utterance):
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

        print(f"  [mode] user_mode={self.current_user_mode} · 직전 침묵={silence_before:.0f}s")
        print("  [1/2] 🔍🎓 통합 분석/결정 중 (1회 왕복)...")
        combined = self.analyze_and_decide(user_utterance)
        analysis = combined["analysis"]
        decision = combined["decision"]
        if analysis.get("observation_summary"):
            print(f"       · 관찰: {analysis['observation_summary']}")
        print(f"       · 전략: {decision.get('strategy', '')}")
        print(f"       · speaking_agents: {decision.get('speaking_agents', [])}")

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

        print(f"  [3/3] 💬 {len(targets)}명 병렬 발화 생성 (완료순 스트리밍)...")

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

        print(f"  [3/3] 💬 {len(targets)}명 병렬 토큰 스트리밍...")

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
                # max_tokens=480: 한글 2~4문장이 중간에서 잘리지 않도록 여유 부여
                stream = self.api.call(
                    prompt, max_tokens=480, temperature=0.9, stream=True,
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

                # 스트림이 도중 잘린 경우(RECITATION 등) — 최대 2회 재시도.
                # 1차: anti-quote/anti-교과서 지시. 2차: 완전 재구성 강제.
                if _is_incomplete_utterance(full):
                    print(f"       · [ai_stream {aid}] 미완결({len(full)}자, '{full[-10:] if full else ''}') — 1차 재시도")
                    retry_note_1 = (
                        "\n\n---\n[재생성]: 직전 응답이 중단됐다. 아래를 엄수:\n"
                        "- 따옴표로 수학 용어를 감싸지 말 것.\n"
                        "- 교과서 문장('약수가 2개인 수', '1과 자기 자신만' 등)을 그대로 인용하지 말고,\n"
                        "  학생 말투로 풀어쓸 것. 예: '약수가 몇 개냐면 두 개야' → '나누면 1이랑 자기만 되는 수야'.\n"
                        "- 문장 종결 어미(~야/어/지/까/래/해)로 끝맺을 것.\n"
                    )
                    full2 = ""
                    try:
                        raw2 = self.api.call(prompt + retry_note_1, max_tokens=480, temperature=1.0)
                        full2 = sanitize_ai_output(raw2)
                    except Exception as e:
                        print(f"       · [ai_stream {aid}] 1차 재시도 실패: {e}")

                    # 2차 재시도: 수학 용어 최소화 + 짧은 질문 중심
                    full3 = ""
                    if _is_incomplete_utterance(full2):
                        print(f"       · [ai_stream {aid}] 1차도 미완결 — 2차 재시도(용어 최소화)")
                        retry_note_2 = (
                            "\n\n---\n[2차 재생성]: 앞 시도 모두 중단됐다.\n"
                            "- 수학 용어 사용 최소화. 대신 학생 친구 말투로만.\n"
                            "- 2문장 이내 짧게. 친구한테 되묻는 질문 하나로 끝내기.\n"
                            "- 예: '음, 헷갈리는구나. 같이 천천히 볼까?' / '아 그렇게 생각했구나, 왜 그런 것 같아?'\n"
                        )
                        try:
                            raw3 = self.api.call(prompt + retry_note_2, max_tokens=300, temperature=1.2)
                            full3 = sanitize_ai_output(raw3)
                        except Exception as e:
                            print(f"       · [ai_stream {aid}] 2차 재시도 실패: {e}")

                    # 최선 선택: 완결인 것 중 가장 최근 것 우선
                    best = None
                    for candidate in (full3, full2, full):
                        if candidate and not _is_incomplete_utterance(candidate):
                            best = candidate
                            break

                    if best is None:
                        # 모두 불완결 — partial에 자연스러운 말줄임표로 종결 (UX 마커 제거)
                        partial = full3 or full2 or full or ""
                        if partial:
                            # 마지막이 이미 문장부호면 그대로, 아니면 "..." 추가
                            if partial.rstrip()[-1:] in ".!?…":
                                best = partial.rstrip()
                            else:
                                best = partial.rstrip() + "..."
                        else:
                            # 아예 빈 응답 — 페르소나별 generic 질문 fallback
                            generic_by_aid = {
                                "ai_1": "음, 잠깐 다시 볼까? 어디서 막혔어?",
                                "ai_2": "잠깐, 지금까지 얘기한 거 정리해보면 어떻게 될까?",
                                "ai_3": "어… 나는 잘 모르겠는데, 같이 다시 볼래?",
                            }
                            best = generic_by_aid.get(aid, "잠깐, 다시 볼까?")
                    full = best

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

        print(f"  [silence] 사용자 침묵 {silence:.0f}s 감지 → AI 선제 발화")
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

    def stage_intro_utterance(self, opener_key="ai_1"):
        # opener: 기본은 민준(ai_1). gradio_app에서는 Stage 2부터 서연(ai_2)을 명시 지정함.
        # 의도적으로 opener를 바꾸면 각 Stage 시작 지점에서 발화자가 다양해진다.
        persona = self.config["personas"]["ai_students"][opener_key]
        stage = self.current_stage_info()

        # 교사 주도 intro_message는 gradio_app이 별도 system 버블로 렌더링한다
        # (session.conversation에는 append하지 않음 — UI 전용 메시지).
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
        print(f"       · [stage_intro raw len={len(raw)}] {raw[:120]!r}...")
        text = sanitize_ai_output(raw)

        # 불완전(길이 짧음 or 문장 미종결)이면 재시도 (모듈 레벨 헬퍼 사용)
        _is_incomplete = _is_incomplete_utterance
        attempt = 1
        while _is_incomplete(text) and attempt <= 2:
            print(f"       · [stage_intro] 불완전({len(text)}자, '{text[-10:] if text else ''}') — 재시도 #{attempt}")
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
            print(f"       · [stage_intro retry#{attempt} len={len(raw)}] {raw[:120]!r}...")
            text = sanitize_ai_output(raw)
            attempt += 1

        # 그래도 불완전하면 하드코딩 fallback
        if _is_incomplete(text):
            stage_num = self.current_stage
            fallback_by_stage = {
                1: "자, 이번엔 소수랑 합성수가 뭔지 같이 얘기해 보자. 다들 어떻게 생각해?",
                2: "자, 이번엔 20부터 30까지의 수 중에 소수가 뭐가 있는지 한번 찾아보자. 누가 먼저 해볼래?",
                3: "자, 이번엔 진짜 문제야. 10보다 큰 최소 합성수랑 15보다 작은 최대 소수의 합을 구해보자. 어디서부터 시작해볼까?",
            }
            text = fallback_by_stage.get(stage_num,
                "자, 이번에는 새로운 걸 같이 얘기해볼까? 다들 어떻게 생각해?")
            print(f"       · [stage_intro] 하드코딩 fallback 적용 (stage {stage_num})")

        print(f"       · [stage_intro final len={len(text)}] {text[:120]!r}...")
        self.conversation.append({
            "speaker": persona["name"],
            "content": text,
            "stage": self.current_stage,
        })
        # **AI opener만 반환**한다. intro_message는 별도 system 버블로 이미
        # self.conversation에 append되어 있고, gradio_app이 별도 렌더링한다.
        return text

    def pop_pending_intro_message(self):
        """gradio_app에서 stage_intro_utterance 호출 직후 이 메서드로 system 버블을
        꺼내 별도 UI 버블로 렌더링한다. 반환: intro_message 문자열 또는 None.
        """
        stage = self.current_stage_info()
        return stage.get("intro_message")


# ============================================================
# 파일 끝 — mount sync 강제 플러시용 여분 주석
# 이 파일은 CollaborativeSession 클래스만 export한다.
# 수정 후 bash에서 AST parse 시 truncation이 보이면 이 구역을 한 줄 추가/삭제해
# Windows 쪽 flush를 유도하자.
# ============================================================
