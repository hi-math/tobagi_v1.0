"""협력학습 세션 파이프라인.

3단계 구조: 학습자 분석 → 교수자 의사결정 → AI 학생 발화.
전역 변수를 쓰지 않고 config/prompts/learner_models/api 를 명시적으로 주입 받는다.
"""

from .llm_api import extract_json, render_prompt


class CollaborativeSession:
    """사용자 1명 + AI 학생 2명의 협력학습 세션.

    Attributes:
        config:         CONFIG dict
        prompts:        PROMPTS dict
        learner_models: {"user", "ai_1", "ai_2"} dict (상태가 in-place 업데이트됨)
        api:            ClaudeAPI 인스턴스
        task:           config["tasks"] 참조
        current_stage:  현재 stage 번호 (1부터)
        conversation:   [{speaker, content, stage}, ...]
        last_tutor_decision: 가장 최근 tutor_decision() 결과
    """

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

    # ----- 유틸 -----
    def recent_dialogue(self, n=6):
        recent = self.conversation[-n:]
        return "\n".join(f"[{m['speaker']}]: {m['content']}" for m in recent) or "(아직 대화 없음)"

    def current_stage_info(self):
        return self.task["stages"][str(self.current_stage)]

    def _flatten_lm(self, lm):
        """시각화·프롬프트용으로 학습자 모델 값만 평탄화."""
        return {
            mk: {dk: dv["value"] for dk, dv in mv.items()}
            for mk, mv in lm["models"].items()
        }

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

        # 오개념 리스트
        if "misconceptions" in user_lm["cognitive_state"]:
            mis = user_lm["cognitive_state"]["misconceptions"]
            if not isinstance(mis["value"], list):
                mis["value"] = []
            for add in result.get("misconception_changes", {}).get("added", []):
                if add not in mis["value"]:
                    mis["value"].append(add)
            for rem in result.get("misconception_changes", {}).get("removed", []):
                if rem in mis["value"]:
                    mis["value"].remove(rem)
        return result

    # ----- 2단계: 교수자 모델 의사결정 -----
    def tutor_decision(self):
        stage = self.current_stage_info()
        prompt = render_prompt(self.prompts["tutor_decision"], {
            "learning_objectives": self.task["learning_objectives"],
            "current_stage_full": stage,
            "user_learner_model": self._flatten_lm(self.learner_models["user"]),
            "ai_1_learner_model": self._flatten_lm(self.learner_models["ai_1"]),
            "ai_2_learner_model": self._flatten_lm(self.learner_models["ai_2"]),
            "recent_dialogue": self.recent_dialogue(6),
            "tutor_principles": self.config["tutor_model"]["tutor_model"]["pedagogical_principles"],
            "role_pool": self.config["tutor_model"]["tutor_model"]["ai_student_role_assignment"]["role_pool"],
        })
        try:
            raw = self.api.call(prompt, max_tokens=1200, temperature=0.5)
            decision = extract_json(raw)
        except Exception as e:
            print(f"  ⚠️ 교수자 의사결정 실패: {e}")
            decision = {
                "strategy": "기본 진행",
                "ai_1_directive": {
                    "role": "탐험가",
                    "speech_goal": "자연스럽게 대화 진행",
                    "must_include": "",
                    "must_avoid": "정답 직접 말하기",
                },
                "ai_2_directive": {
                    "role": "검증자",
                    "speech_goal": "민준의 의견 검토",
                    "must_include": "",
                    "must_avoid": "정답 직접 말하기",
                },
                "pedagogical_goal": "",
                "stage_complete": False,
            }
        self.last_tutor_decision = decision
        return decision

    # ----- 3단계: AI 학생 발화 -----
    def generate_ai_utterance(self, student_key, directive, user_utterance):
        persona = self.config["personas"]["ai_students"][student_key]
        stage = self.current_stage_info()
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
            "recent_dialogue": self.recent_dialogue(8),
            "user_utterance": user_utterance,
        })
        return self.api.call(prompt, max_tokens=400, temperature=0.9).strip()

    # ----- 전체 턴 -----
    def user_turn(self, user_utterance):
        self.conversation.append({
            "speaker": "사용자",
            "content": user_utterance,
            "stage": self.current_stage,
        })
        self.turn_count += 1
        self.stage_turn_count += 1

        print("  [1/3] 🔍 학습자 분석 중...")
        analysis = self.analyze_user_utterance(user_utterance)
        if analysis.get("observation_summary"):
            print(f"       · 관찰: {analysis['observation_summary']}")

        print("  [2/3] 🎓 교수자 모델 의사결정 중...")
        decision = self.tutor_decision()
        print(f"       · 전략: {decision.get('strategy', '')}")

        print("  [3/3] 💬 AI 학생 발화 생성 중...")
        ai_students = self.config["personas"]["ai_students"]
        ai1_text = self.generate_ai_utterance("ai_1", decision["ai_1_directive"], user_utterance)
        self.conversation.append({
            "speaker": ai_students["ai_1"]["name"],
            "content": ai1_text,
            "stage": self.current_stage,
        })

        ai2_text = self.generate_ai_utterance("ai_2", decision["ai_2_directive"], user_utterance)
        self.conversation.append({
            "speaker": ai_students["ai_2"]["name"],
            "content": ai2_text,
            "stage": self.current_stage,
        })

        return {
            "analysis": analysis,
            "decision": decision,
            "ai_1": ai1_text,
            "ai_2": ai2_text,
        }

    # ----- Stage 전환 -----
    def advance_stage(self):
        if self.current_stage < 4:
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
        })
        text = self.api.call(prompt, max_tokens=300, temperature=0.8).strip()
        self.conversation.append({
            "speaker": persona["name"],
            "content": text,
            "stage": self.current_stage,
        })
        return text
