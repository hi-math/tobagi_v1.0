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
from .llm_api import extract_json, render_prompt, DEFAULT_HAIKU, DEFAULT_SONNET

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
    """AI 발화 응답에서 메타-추론/헤더 프리픽스를 제거.

    Haiku 등 일부 모델은 프롬프트 제약을 어기고 다음 같은 누출을 함께 출력한다:
      ```
      민준
      AI 학생 발화 생성
      현재 상황 분석:
      * 침묵 트리거: true
      * user_mode: collaborator
      발화
      어디가 헷갈려?
      ```
    이 함수는 그런 누출 줄을 걷어내고 실제 대사만 남긴다.

    전략:
      1) 전체를 코드펜스로 감쌌으면 펜스 제거
      2) "발화" / "Utterance" / "응답" / "대사" 같은 라벨이 단독 줄로 나오면
         그 뒤 첫 비어있지 않은 블록만 채택
      3) 마크다운 헤더(`#`, `##`), 메타 bullet(`* key:`, `- key:`, `**굵게:**`),
         학생 이름 단독 줄 제거
      4) 앞뒤 공백·따옴표 trim
    """
    import re

    if not text:
        return ""

    t = text.strip()

    # (1) 전체 코드펜스 감쌈 벗기기
    if t.startswith("```") and t.endswith("```"):
        inner = t[3:-3].strip()
        # 첫 줄이 언어 지정이면 제거
        first_nl = inner.find("\n")
        if first_nl != -1 and " " not in inner[:first_nl] and "=" not in inner[:first_nl]:
            inner = inner[first_nl+1:].strip()
        t = inner

    # (2) "발화" 등 라벨이 단독 줄로 있으면 그 뒤만 채택
    label_pat = re.compile(
        r"^\s*(?:발화|발\s*화|Utterance|utterance|대사|응답|output|Output|OUTPUT|final|Final)\s*[:：]?\s*$",
        re.MULTILINE,
    )
    matches = list(label_pat.finditer(t))
    if matches:
        t = t[matches[-1].end():].strip()

    # (3) 줄 단위 필터링
    student_names = ("민준", "서연", "연우")
    kept = []
    for ln in t.split("\n"):
        stripped = ln.strip()
        if not stripped:
            kept.append(ln)
            continue
        # 마크다운 헤더
        if re.match(r"^#{1,6}\s", stripped):
            continue
        # 학생 이름만 단독으로 있는 줄
        if stripped in student_names:
            continue
        # 메타 bullet: `* key: value` 또는 `- key: value` (콜론이 앞쪽에 있음)
        m = re.match(r"^[\*\-\u2022]\s+\**([^:：\n]{1,25})[:：]", stripped)
        if m:
            continue
        # "현재 상황 분석:", "분석:", "추론:" 같은 단독 라벨 라인
        if re.match(r"^\**(현재\s*상황\s*분석|분석|reasoning|추론|상황)\**\s*[:：]\s*$", stripped):
            continue
        # "## xxx" 같은 굵게 섹션 헤더도 제거
        if re.match(r"^\*\*[^*]{1,30}\*\*\s*$", stripped):
            continue
        kept.append(ln)
    t = "\n".join(kept).strip()

    # (4) 라벨 재처리 (앞의 필터링 후 "발화" 등이 첫 줄로 드러날 수 있음)
    matches = list(label_pat.finditer(t))
    if matches:
        t = t[matches[-1].end():].strip()

    # (5) 앞/뒤 감싸는 큰따옴표 벗기기
    if len(t) >= 2 and t[0] in ('"', '"', "「") and t[-1] in ('"', '"', "」"):
        t = t[1:-1].strip()

    # (6) 여전히 비었다면 원본 반환 (안전장치)
    return t or text.strip()


def detect_user_mode(user_utterance: str) -> str:
    """사용자가 설명자(teacher) 모드인지 협력자(collaborator) 모드인지 감지.

    기준:
    - 문장이 3개 이상이면서
    - 설명/정당화 키워드(때문에·왜냐하면·정의 등)를 포함하면 teacher
    - 아니면 collaborator
    """
    if not user_utterance:
        return "collaborator"
    if _sentence_count(user_utterance) >= TEACHER_MODE_MIN_SENTENCES:
        if any(k in user_utterance for k in TEACHER_MODE_KEYWORDS):
            return "teacher"
    return "collaborator"


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

    def recent_dialogue(self, n=6):
        recent = self.conversation[-n:]
        return "\n".join(f"[{m['speaker']}]: {m['content']}" for m in recent) or "(아직 대화 없음)"

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
        try:
            raw = self.api.call(prompt, max_tokens=800, temperature=0.3,
                                model=DEFAULT_HAIKU)
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
            "current_stage_full": stage,
            "user_learner_model": self._lm_summary(self.learner_models["user"]),
            "ai_1_learner_model": self._lm_summary(self.learner_models["ai_1"]),
            "ai_2_learner_model": self._lm_summary(self.learner_models["ai_2"]),
            "ai_3_learner_model": self._lm_summary(self.learner_models["ai_3"]),
            "recent_dialogue": self.recent_dialogue(8),
            "user_silence_seconds": f"{user_silence_seconds:.0f}",
            "last_silence_trigger_agent": self.last_silence_agent or "(없음)",
            "user_mode_hint": user_mode,
            "tutor_principles": tm["pedagogical_principles"],
            "role_pool": tm["ai_student_role_assignment"]["role_pool"],
            "silence_prompt_policy": tm.get("silence_prompt_policy", {}),
            "user_as_teacher_policy": tm.get("user_as_teacher_policy", {}),
            "domain_knowledge": self._domain_text(),
        })
        try:
            raw = self.api.call(prompt, max_tokens=900, temperature=0.5)
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
        self.last_tutor_decision = decision
        return decision

    def _build_ai_prompt(self, student_key, directive, user_utterance,
                         user_mode="collaborator", silence_trigger=False,
                         user_silence_seconds=0.0):
        persona = self.config["personas"]["ai_students"][student_key]
        stage = self.current_stage_info()
        directive = directive or {}
        return render_prompt(self.prompts["ai_student"], {
            "student_name": persona["name"],
            "my_persona": persona,
            "my_learner_state": self._flatten_lm(self.learner_models[student_key]),
            "stage_title": stage["title"],
            "stage_prompt": stage["prompt"],
            "role": directive.get("role", ""),
            "speech_goal": directive.get("speech_goal", ""),
            "must_include": directive.get("must_include", ""),
            "must_avoid": directive.get("must_avoid", ""),
            "user_mode": user_mode,
            "silence_trigger": str(bool(silence_trigger)).lower(),
            "user_silence_seconds": f"{user_silence_seconds:.0f}",
            "recent_dialogue": self.recent_dialogue(8),
            "user_utterance": user_utterance,
            "domain_knowledge": self._domain_text(),
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
        raw = self.api.call(prompt, max_tokens=400, temperature=0.9)
        return sanitize_ai_output(raw)

    def analyze_and_decide(self, user_utterance):
        stage = self.current_stage_info()
        tm = self.config["tutor_model"]["tutor_model"]
        prompt = render_prompt(self.prompts["analyze_and_decide"], {
            "task_title": self.task["task_title"],
            "stage_title": stage["title"],
            "core_question": stage["core_question"],
            "current_stage_full": stage,
            "user_utterance": user_utterance,
            "recent_dialogue": self.recent_dialogue(8),
            "user_learner_model": self._lm_summary(self.learner_models["user"]),
            "ai_1_learner_model": self._lm_summary(self.learner_models["ai_1"]),
            "ai_2_learner_model": self._lm_summary(self.learner_models["ai_2"]),
            "ai_3_learner_model": self._lm_summary(self.learner_models["ai_3"]),
            "stage_rubric": stage.get("assessment_rubric", "(해당 Stage 루브릭 없음)"),
            "stage_checklist": stage.get("ai_checklist", "(해당 Stage 체크리스트 없음)"),
            "learning_objectives": self.task["learning_objectives"],
            "tutor_principles": tm["pedagogical_principles"],
            "role_pool": tm["ai_student_role_assignment"]["role_pool"],
            "silence_prompt_policy": tm.get("silence_prompt_policy", {}),
            "user_as_teacher_policy": tm.get("user_as_teacher_policy", {}),
            "user_silence_seconds": "0",
            "last_silence_trigger_agent": self.last_silence_agent or "(없음)",
            "user_mode_hint": self.current_user_mode,
            "domain_knowledge": self._domain_text(),
        })
        try:
            raw = self.api.call(prompt, max_tokens=1400, temperature=0.4)
            merged = extract_json(raw)
        except Exception as e:
            print(f"  ⚠️ 통합 분석/결정 실패: {e} — 폴백 분리 호출")
            analysis = self.analyze_user_utterance(user_utterance)
            decision = self.tutor_decision(user_silence_seconds=0.0,
                                           user_mode=self.current_user_mode)
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
        """발화할 AI들을 병렬로 스트리밍 호출하고, 토큰이 도착할 때마다 이벤트 yield.

        Yields:
            ("start",  aid, "")       — 첫 청크 직전
            ("update", aid, chunk)    — 토큰 델타 (누적 아닌 증분)
            ("done",   aid, full)     — 해당 AI 발화 완료 (conversation에 기록 직후)
        """
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
                stream = self.api.call(
                    prompt, max_tokens=400, temperature=0.9, stream=True,
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
        self.last_user_utterance_ts = time.time() - min_seconds + 30
        return {"agent_id": aid, "text": text, "decision": decision}

    def advance_stage(self):
        total_stages = len(self.task["stages"])
        if self.current_stage < total_stages:
            self.current_stage += 1
            self.stage_turn_count = 0
            return True
        return False

    def stage_intro_utterance(self, opener_key="ai_1"):
        persona = self.config["personas"]["ai_students"][opener_key]
        stage = self.current_stage_info()
        prompt = render_prompt(self.prompts["stage_intro"], {
            "opener_name": persona["name"],
            "stage_title": stage["title"],
            "stage_prompt": stage["prompt"],
            "core_question": stage["core_question"],
            "my_persona": persona,
            "domain_knowledge": self._domain_text(),
        })
        raw = self.api.call(prompt, max_tokens=300, temperature=0.8)
        text = sanitize_ai_output(raw)
        self.conversation.append({
            "speaker": persona["name"],
            "content": text,
            "stage": self.current_stage,
        })
        return text
