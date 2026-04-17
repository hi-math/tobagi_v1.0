"""학습자 모델 인스턴스 생성.

각 학습자(사용자, ai_1, ai_2)는 동일 schema(learner_model.json)의
모델·차원을 갖고, 각 차원은 {value, history} 구조를 유지한다.
"""


def create_learner_model_instance(config, student_name, override_initial=None):
    """스키마를 기반으로 빈 학습자 모델 인스턴스 생성.

    Args:
        config:            CONFIG dict (learner_model_schema 포함)
        student_name:      표시용 이름
        override_initial:  {model_key: {dim_key: init_value}} — persona 의 초기치

    Returns:
        {student_name, models: {mk: {dk: {value, history}}}}
    """
    schema = config["learner_model_schema"]["models"]
    inst = {"student_name": student_name, "models": {}}
    for mk, mv in schema.items():
        inst["models"][mk] = {}
        for dk, dv in mv["dimensions"].items():
            init_val = dv.get("default", 3)
            if (override_initial
                    and mk in override_initial
                    and dk in override_initial[mk]):
                init_val = override_initial[mk][dk]
            inst["models"][mk][dk] = {
                "value": init_val,
                "history": [{"stage": 0, "value": init_val, "evidence": "초기값"}],
            }
    return inst


def init_learners(config):
    """사용자 1 + AI 학생 2명의 학습자 모델을 dict로 반환.

    반환 키: "user", "ai_1", "ai_2"
    """
    ai_students = config["personas"]["ai_students"]
    return {
        "user": create_learner_model_instance(config, "사용자"),
        "ai_1": create_learner_model_instance(
            config,
            ai_students["ai_1"]["name"],
            ai_students["ai_1"]["initial_learner_state"],
        ),
        "ai_2": create_learner_model_instance(
            config,
            ai_students["ai_2"]["name"],
            ai_students["ai_2"]["initial_learner_state"],
        ),
    }
