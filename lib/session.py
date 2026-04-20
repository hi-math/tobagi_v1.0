"""협력학습 세션 파이프라인.

3단계 구조: 학습자 분석 → 교수자 의사결정 → AI 학생 발화.
전역 변수를 쓰지 않고 config/prompts/learner_models/api 를 명시적으로 주입 받는다.

확장 기능:
- AI 학생 3명 지원 (ai_1 민준·상, ai_2 서연·중, ai_3 연우·하)
- 침묵 유도: 사용자가 60초 이상 말하지 않으면 nudge_on_silence()로 한 명의 AI가 선제 발화
- 사용자 교수자 모드: 사용자가 설명자 모드가 되면 AI는 학습자 모드로 전환해 짧게 반응
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from .llm_api import extract_json, render_prompt


# 침묵 유도 임계치 (초)
SILENCE_THRESHOLD_SECONDS = 60
SILENCE_ESCALATION_SECONDS = 120

# 사용자 설명자 모드 감지 기준
TEACHER_MODE_MIN_SENTENCES = 3
TEACHER_MODE_KEYWORDS = ("때문에", "왜냐하면", "이유는", "즉", "따라서", "약수", "정의", "만약", "반대로")


def _sentence_count(text: str) -> int:
    """대략적인 문장 수 계산 (., \!, ?, 줄바꿈 기준)."""
    import re
    if not text:
        return 0
    chunks = re.split(r"[\.\!?\n]+", text.strip())
    return len([c for c in chunks if c.strip()])


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
    """사용자 1명 + AI 학생 3명의 협력학습 세션.

    Attributes:
        config:         CONFIG dict
        prompts:        PROMPTS dict
        learner_models: {"user", "ai_1", "ai_2", "ai_3"} dict (상태가 in-place 업데이트됨)
        api:            ClaudeAPI 인스턴스
        task:           config["tasks"] 참조
        current_stage:  현재 stage 번호 (1부터)
        conversation:   [{speaker, content, stage}, ...]
        last_tutor_decision: 가장 최근 tutor_decision() 결과
        last_user_utterance_ts: 사용자 마지막 발화 시점(time.time())
        last_silence_agent:     마지막으로 침묵 유도를 담당한 AI id (로테이션용)
        current_user_mode:      "collaborator" | "teacher"
    """

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

        # 침묵 타이머 & 사용자 모드 상태
        self.last_user_utterance_ts = time.time()
        self.last_silence_agent = None
        self.current_user_mode = "collaborator"

    # ----- 유틸 -----
    def recent_dialogue(self, n=6):
        recent = self.conversation[-n:]
        return "\n".join(f"[{m['speaker']}]: {m['content']}" for m in recent) or "(아직 대화 없음)"

    def current_stage_info(self):
        return self.task["stages"][str(self.current_stage)]

    def _domain_text(self):
        """도메인 지식(교과서 등) 통합 텍스트. 없으면 안내 문구."""
        dk = self.config.get("domain_knowledge") or {}
        text = dk.get("combined_text") or ""
        return text if text else "(도메인 자료가 로드되지 않음)"

    def _flatten_lm(self, lm):
        """시각화·프롬프트용으로 학습자 모델 값만 평탄화.

        주의: self_efficacy 항목은 {pre, post, history} 구조라 'value' 키가 없다.
              → item별로 {"pre": v, "post": v}만 요약해 리턴한다.
        다른 모델(ordinal/counter/list/stage_categorical)은 dv["value"]를 사용.
        """
        out = {}
        for mk, mv in lm["models"].items():
            per_dim = {}
            for dk, dv in mv.items():
                if isinstance(dv, dict) and "value" in dv:
                    per_dim[dk] = dv["value"]
                elif isinstance(dv, dict) and ("pre" in dv or "post" in dv):
                    # self_efficacy 항목: pre/post만 노출 (history는 제외)
                    per_dim[dk] = {"pre": dv.get("pre"), "post": dv.get("post")}
                else:
                    per_dim[dk] = None
            out[mk] = per_dim
        return out

    # ----- 1단계: 학습자 분석 -----
    def analyze_user_utterance(self, user_utterance):
        stage = self.current_stage_info()
        prompt = render_prompt(self.prompts["learner_analysis"], {
            "task_title": self.task["task_title"],
            "stage_title": stage["title"],
            "core_question": stage["core_question"],
            "user_utterance": user_utterance,
            "recent_dialogue": self.recent_dialogue(6),
            "current_learner_model": self._flatten_lm(self.learner_models["user"]),
            "learner_model_schema": self.config["learner_model_schema"]["models"],
            "domain_knowledge": self._domain_text(),
            "stage_rubric": stage.get("assessment_rubric", "(해당 Stage 루브릭 없음)"),
            "stage_checklist": stage.get("ai_checklist", "(해당 Stage 체크리스트 없음)"),
        })
        try:
            raw = self.api.call(prompt, max_tokens=1200, temperature=0.3)
            result = extract_json(raw)
        except Exception as e:
            print(f"  ⚠️ 학습자 분석 실패: {e}")
            return {
                "updates": [],
                "misconception_changes": {"added": [], "removed": []},
                "observation_summary": "",
            }

        # 학습자 모델 업데이트 (사용자만)
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

        # 오개념 리스트 — v2 스키마는 task_achievement.misconceptions에 위치
        # (v1 호환: cognitive_state.misconceptions도 있으면 함께 업데이트)
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

    # ----- 침묵 타이머 -----
    def seconds_since_user_spoke(self) -> float:
        """사용자가 마지막으로 발화한 이후 경과된 초."""
        return time.time() - self.last_user_utterance_ts

    def _pick_silence_agent(self) -> str:
        """침묵 유도를 담당할 AI id 선택.

        규칙: 마지막 침묵 유도 담당자와 다른 AI를 선택.
        기본 우선순위: 분위기 해소는 ai_3(연우) 우선, 그 다음 ai_1(민준), ai_2(서연).
        """
        preference = ["ai_3", "ai_1", "ai_2"]
        # 120초 이상이면 ai_1(민준)을 우선 → 난이도 낮춘 확인 질문
        if self.seconds_since_user_spoke() >= SILENCE_ESCALATION_SECONDS:
            preference = ["ai_1", "ai_3", "ai_2"]
        for aid in preference:
            if aid != self.last_silence_agent:
                return aid
        return preference[0]

    # ----- 2단계: 교수자 모델 의사결정 -----
    def tutor_decision(self, user_silence_seconds: float = 0.0, user_mode: str = "collaborator"):
        stage = self.current_stage_info()
        silence_trigger = user_silence_seconds >= SILENCE_THRESHOLD_SECONDS
        tm = self.config["tutor_model"]["tutor_model"]
        prompt = render_prompt(self.prompts["tutor_decision"], {
            "learning_objectives": self.task["learning_objectives"],
            "current_stage_full": stage,
            "user_learner_model": self._flatten_lm(self.learner_models["user"]),
            "ai_1_learner_model": self._flatten_lm(self.learner_models["ai_1"]),
            "ai_2_learner_model": self._flatten_lm(self.learner_models["ai_2"]),
            "ai_3_learner_model": self._flatten_lm(self.learner_models["ai_3"]),
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
            raw = self.api.call(prompt, max_tokens=1500, temperature=0.5)
            decision = extract_json(raw)
        except Exception as e:
            print(f"  ⚠️ 교수자 의사결정 실패: {e}")
            # 폴백: 침묵 트리거 시 한 명만, 그 외 두 명 기본 배정
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
        # speaking_agents 기본값 보정
        if "speaking_agents" not in decision or decision["speaking_agents"] is None:
            decision["speaking_agents"] = [
                aid for aid in self.AI_KEYS
                if decision.get(f"{aid}_directive")
            ]
        self.last_tutor_decision = decision
        return decision

    # ----- 3단계: AI 학생 발화 -----
    def generate_ai_utterance(self, student_key, directive, user_utterance,
                              user_mode="collaborator", silence_trigger=False,
                              user_silence_seconds=0.0):
        persona = self.config["personas"]["ai_students"][student_key]
        stage = self.current_stage_info()
        directive = directive or {}
        prompt = render_prompt(self.prompts["ai_student"], {
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
        return self.api.call(prompt, max_tokens=400, temperature=0.9).strip()

    # ----- 턴 준비: analyze + tutor_decision (AI 발화 전까지) -----
    def user_turn_prep(self, user_utterance):
        """사용자 발화를 받아서 학습자 분석 + 교수자 의사결정까지 수행.

        AI 학생 발화는 생성하지 않는다. 발화는 stream_ai_turns()로 이어서
        병렬·완료순으로 스트리밍한다.
        """
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
        print("  [1/3] 🔍 학습자 분석 중...")
        analysis = self.analyze_user_utterance(user_utterance)
        if analysis.get("observation_summary"):
            print(f"       · 관찰: {analysis['observation_summary']}")

        print("  [2/3] 🎓 교수자 모델 의사결정 중...")
        decision = self.tutor_decision(user_silence_seconds=0.0,
                                       user_mode=self.current_user_mode)
        print(f"       · 전략: {decision.get('strategy', '')}")
        print(f"       · speaking_agents: {decision.get('speaking_agents', [])}")

        return {
            "analysis": analysis,
            "decision": decision,
            "user_mode": self.current_user_mode,
        }

    # ----- 병렬·완료순 AI 발화 스트리밍 -----
    def stream_ai_turns(self, user_utterance, decision, silence_trigger=False,
                        user_silence_seconds=0.0):
        """발화해야 하는 AI들의 API 호출을 병렬로 쏘고, 완료 순서대로 yield.

        "말할 내용이 먼저 생긴 사람부터 말한다"는 자연스러운 대화 리듬을 구현한다.
        짧게 말하는 AI(예: 질문자 연우)가 보통 먼저 끝나고 먼저 발화한다.

        Yields:
            (agent_id, text): API가 완료된 순서대로
        """
        ai_students = self.config["personas"]["ai_students"]
        speaking = decision.get("speaking_agents", []) or []

        # 발화 대상 수집 (directive 없는 agent는 건너뜀)
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

        # max_workers = 발화자 수 (1~3). 같은 ClaudeAPI 인스턴스를 공유해도 SDK는 thread-safe.
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

    # ----- 전체 턴 (기존 API 호환) -----
    def user_turn(self, user_utterance):
        """CLI 런너 등 기존 호출부 호환용: prep + stream을 한 번에 수행.

        스트리밍이 필요한 UI에서는 user_turn_prep + stream_ai_turns를 직접 쓰라.
        """
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

    # ----- 침묵 유도 턴 (사용자가 60초 이상 말하지 않을 때 호출) -----
    def nudge_on_silence(self, min_seconds: float = SILENCE_THRESHOLD_SECONDS):
        """사용자 침묵이 임계치를 넘었을 때 한 명의 AI가 대화를 유도한다.

        호출 예 (UI 측 1초 주기 폴링):
            if session.seconds_since_user_spoke() >= 60 and not just_nudged:
                session.nudge_on_silence()

        반환: {"agent_id", "text", "decision"} 또는 None(임계 미달)
        """
        silence = self.seconds_since_user_spoke()
        if silence < min_seconds:
            return None

        print(f"  [silence] 사용자 침묵 {silence:.0f}s 감지 → AI 선제 발화")
        decision = self.tutor_decision(user_silence_seconds=silence,
                                       user_mode=self.current_user_mode)

        # 교수자가 지정한 speaker가 있으면 그걸 쓰고, 없으면 로테이션 규칙으로 선택
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
        # 침묵 유도 후에도 last_user_utterance_ts는 갱신하지 않는다
        # (사용자가 실제 말할 때까지 계속 침묵 시간이 누적되어야 함)
        # 단, 연속 nudge 방지를 위해 약간 뒤로 밀어 재트리거 간격 확보
        self.last_user_utterance_ts = time.time() - min_seconds + 30
        return {"agent_id": aid, "text": text, "decision": decision}

    # ----- Stage 전환 -----
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
        text = self.api.call(prompt, max_tokens=300, temperature=0.8).strip()
        self.conversation.append({
            "speaker": persona["name"],
            "content": text,
            "stage": self.current_stage,
        })
        return text
