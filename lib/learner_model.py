"""학습자 모델 인스턴스 생성 및 업데이트 로직 (v2.0).

스키마: config/learner_model.json (schema_version >= 2.0)
  - 인지: task_achievement, math_communication, math_reasoning
  - 정의: cps, self_efficacy
  - extension_slot: +α placeholder

인스턴스 구조:
  {
    "student_name": str,
    "models": {
      "<model_key>": {
        "<dim_key or item_id>": {
          "value": <type별 값> | ("pre", "post" for self_efficacy),
          "history": [{stage, value, evidence, ...}],
          "per_stage_values": {stage: value}  # stage_categorical 전용
        }
      }
    }
  }

차원 타입별 초기값/업데이트 규약:
  - ordinal          : int 1~5, default=3, update: ±1 권장 (클리핑은 호출 측 책임)
  - list             : [], misconceptions 등, update: add/remove 세트 연산
  - counter          : 0, update: += 1 (CPS 태그 1건당)
  - stage_categorical: None, per_stage_values={stage: 'A'..'E'}
  - likert_self_report: items별 {pre, post, history}, update: pre/post 덮어쓰기

공개 API:
  - create_learner_model_instance(config, student_name, override_initial=None,
                                   override_self_efficacy=None)
  - init_learners(config)  →  {"user", "ai_1", "ai_2", "ai_3"}
  - apply_analysis_updates(inst, analysis_result, stage, turn=None, schema=None)
        01_learner_analysis.md 프롬프트 응답 JSON을 인스턴스에 반영
  - apply_cps_tags(inst, tags_result, stage, turn=None)
        08_cps_tagging.md 프롬프트 응답 JSON을 인스턴스에 반영 (카운터 가산)
  - apply_self_efficacy_responses(inst, responses, timestamp=None)
        09_self_efficacy_survey.md 프롬프트 응답 JSON(pre/post)을 인스턴스에 반영
  - counter_to_level(count, thresholds), stage_grade_to_score(grade)  # 시각화 재사용
"""

from copy import deepcopy


# ----- 상수 -----
AI_KEYS = ("ai_1", "ai_2", "ai_3")

# 타입별 기본값 (스키마에 default가 없을 때)
_TYPE_DEFAULTS = {
    "ordinal":             3,
    "list":                [],
    "counter":             0,
    "stage_categorical":   None,
}


# ----- 초기화 -----

def _init_cell(dim_def):
    """한 차원의 초기 인스턴스 셀을 생성.

    반환: {"value", "history": [...]} 그리고 stage_categorical인 경우 "per_stage_values".
    """
    dim_type = dim_def.get("type", "ordinal")
    if "default" in dim_def:
        init_val = dim_def["default"]
    else:
        init_val = _TYPE_DEFAULTS.get(dim_type)
    init_val = deepcopy(init_val) if isinstance(init_val, (list, dict)) else init_val

    cell = {
        "value": init_val,
        "history": [{
            "stage": 0,
            "value": deepcopy(init_val) if isinstance(init_val, (list, dict)) else init_val,
            "evidence": "초기값",
        }],
    }
    if dim_type == "stage_categorical" and dim_def.get("per_stage_tracking"):
        cell["per_stage_values"] = {}
    return cell


def _init_self_efficacy_items(model_def, override_self_efficacy=None):
    """self_efficacy 모델의 items 기반 인스턴스를 생성.

    반환: {item_id: {"pre", "post", "history": []}}
    """
    override_self_efficacy = override_self_efficacy or {}
    out = {}
    for item in model_def.get("items", []):
        iid = item["id"]
        seed = override_self_efficacy.get(iid, {}) or {}
        out[iid] = {
            "pre":  seed.get("pre"),
            "post": seed.get("post"),
            "history": [],
        }
    return out


def create_learner_model_instance(config, student_name,
                                  override_initial=None,
                                  override_self_efficacy=None):
    """스키마(learner_model.json v2.0)에 따라 빈 학습자 모델 인스턴스 생성.

    Args:
        config:                CONFIG dict (learner_model_schema 포함)
        student_name:          표시용 이름
        override_initial:      {model_key: {dim_key: value}} — dimension 기반 모델의 초기치 덮어쓰기
        override_self_efficacy:{item_id: {"pre": int, "post": int}} — self_efficacy 시드

    Returns:
        {student_name, models: {...}}
    """
    schema = config["learner_model_schema"]["models"]
    inst = {"student_name": student_name, "models": {}}
    override_initial = override_initial or {}

    for mk, mv in schema.items():
        # self_efficacy: items-based (dimensions가 아닌 items 구조)
        if mv.get("measurement_method") == "likert_self_report" and "items" in mv:
            inst["models"][mk] = _init_self_efficacy_items(mv, override_self_efficacy)
            continue

        # dimensions 기반 모델
        model_cells = {}
        dims = mv.get("dimensions") or {}
        for dk, dv in dims.items():
            cell = _init_cell(dv)

            # override 적용 (value + 초기 history 양쪽)
            if mk in override_initial and dk in override_initial[mk]:
                ov = override_initial[mk][dk]
                cell["value"] = deepcopy(ov) if isinstance(ov, (list, dict)) else ov
                cell["history"] = [{
                    "stage": 0,
                    "value": deepcopy(cell["value"])
                             if isinstance(cell["value"], (list, dict))
                             else cell["value"],
                    "evidence": "페르소나 초기값",
                }]
                # stage_categorical override가 {stage: level} dict이면
                # per_stage_values에 반영하고 value는 최신 stage 진입 시점에 세팅
                if dv.get("type") == "stage_categorical" and isinstance(ov, dict):
                    cell["per_stage_values"] = deepcopy(ov)
                    cell["value"] = None

            model_cells[dk] = cell

        inst["models"][mk] = model_cells

    return inst


def init_learners(config):
    """사용자 1명 + AI 학생 전원의 학습자 모델 dict 반환.

    반환 키: "user" + personas.json에 정의된 모든 ai_* (ai_1, ai_2, ai_3, ...)
    각 페르소나의 initial_learner_state / initial_self_efficacy를 override로 반영.
    """
    ai_students = config["personas"]["ai_students"]
    learners = {"user": create_learner_model_instance(config, "사용자")}
    for aid, info in ai_students.items():
        learners[aid] = create_learner_model_instance(
            config,
            info.get("name", aid),
            override_initial=info.get("initial_learner_state"),
            override_self_efficacy=info.get("initial_self_efficacy"),
        )
    return learners


# ----- 업데이트 헬퍼 -----

def _clip_ordinal(value, dim_def):
    """ordinal scale 범위 밖 값을 clip. scale 미지정이면 1~5."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None
    scale = dim_def.get("scale", [1, 5])
    lo, hi = int(scale[0]), int(scale[1])
    return max(lo, min(hi, v))


def apply_analysis_updates(inst, analysis_result, stage, turn=None, schema=None):
    """01_learner_analysis.md 응답 JSON을 인스턴스에 반영.

    analysis_result 형식:
      {
        "updates": [{"model", "dimension", "new_value", "evidence"}, ...],
        "misconception_changes": {"added": [...], "removed": [...]},
        "stage_level_estimate": "A"~"E" | null,
        "observation_summary": "..."
      }

    schema: (선택) learner_model.json의 models 딕셔너리. 주면 ordinal clipping 수행.
    """
    models = inst["models"]

    # 1) 일반 updates
    for u in analysis_result.get("updates", []) or []:
        mk, dk = u.get("model"), u.get("dimension")
        nv = u.get("new_value")
        if mk not in models or dk not in models[mk]:
            continue
        cell = models[mk][dk]

        dim_def = (schema or {}).get(mk, {}).get("dimensions", {}).get(dk, {})
        dim_type = dim_def.get("type", "ordinal")
        if dim_type == "ordinal" and dim_def:
            clipped = _clip_ordinal(nv, dim_def)
            if clipped is None:
                continue
            cell["value"] = clipped
        elif dim_type == "stage_categorical":
            cell["value"] = nv
            cell.setdefault("per_stage_values", {})[str(stage)] = nv
        else:
            cell["value"] = nv

        cell.setdefault("history", []).append({
            "stage": stage,
            "turn": turn,
            "value": cell["value"],
            "evidence": u.get("evidence", ""),
        })

    # 2) misconceptions (task_achievement.misconceptions 위치; v1 cognitive_state도 호환)
    mc = analysis_result.get("misconception_changes") or {}
    added = mc.get("added") or []
    removed = mc.get("removed") or []

    mis_targets = []
    if "task_achievement" in models and "misconceptions" in models["task_achievement"]:
        mis_targets.append(models["task_achievement"]["misconceptions"])
    if "cognitive_state" in models and "misconceptions" in models["cognitive_state"]:
        mis_targets.append(models["cognitive_state"]["misconceptions"])

    for mis in mis_targets:
        if not isinstance(mis.get("value"), list):
            mis["value"] = []
        mis.setdefault("history", [])
        for a in added:
            if a not in mis["value"]:
                mis["value"].append(a)
                mis["history"].append({
                    "stage": stage, "turn": turn, "event": "added",
                    "item": a, "value": list(mis["value"]),
                })
        for r in removed:
            if r in mis["value"]:
                mis["value"].remove(r)
                mis["history"].append({
                    "stage": stage, "turn": turn, "event": "removed",
                    "item": r, "value": list(mis["value"]),
                })

    # 3) stage_level_estimate (task_achievement.stage_level에 덮어쓰기)
    sle = analysis_result.get("stage_level_estimate")
    if sle and "task_achievement" in models and "stage_level" in models["task_achievement"]:
        cell = models["task_achievement"]["stage_level"]
        cell["value"] = sle
        cell.setdefault("per_stage_values", {})[str(stage)] = sle
        cell.setdefault("history", []).append({
            "stage": stage, "turn": turn, "value": sle,
            "evidence": analysis_result.get("observation_summary", ""),
            "source": "stage_level_estimate",
        })


def apply_cps_tags(inst, tags_result, stage, turn=None, min_confidence=0.6):
    """08_cps_tagging.md 응답 JSON을 인스턴스에 반영(카운터 +1 가산).

    tags_result 형식:
      {
        "tags": [{"dimension", "evidence_feature", "quote", "confidence"}, ...],
        "none": bool,
        "note": "..."
      }
    반환: 실제로 가산된 태그 수
    """
    cps = inst["models"].get("cps")
    if not cps:
        return 0

    gained = 0
    for tag in tags_result.get("tags") or []:
        dim = tag.get("dimension")
        if dim not in cps:
            continue
        try:
            conf = float(tag.get("confidence", 1.0))
        except (TypeError, ValueError):
            conf = 1.0
        if conf < min_confidence:
            continue
        cell = cps[dim]
        if not isinstance(cell.get("value"), (int, float)):
            cell["value"] = 0
        cell["value"] = int(cell["value"]) + 1
        cell.setdefault("history", []).append({
            "stage": stage,
            "turn": turn,
            "value": cell["value"],
            "evidence_feature": tag.get("evidence_feature", ""),
            "quote": tag.get("quote", ""),
            "confidence": conf,
        })
        gained += 1
    return gained


def apply_self_efficacy_responses(inst, responses, timestamp=None):
    """09_self_efficacy_survey.md 응답 JSON을 인스턴스에 반영.

    responses 형식:
      {
        "phase": "pre" | "post",
        "stage": int,
        "timestamp": "ISO8601 (선택)",
        "responses": {"se_01": 3, "se_02": 2, ...}
      }
    반환: 기록된 응답 수
    """
    se = inst["models"].get("self_efficacy")
    if not se:
        return 0

    phase = responses.get("phase")
    if phase not in ("pre", "post"):
        return 0
    stage = responses.get("stage")
    ts = responses.get("timestamp") or timestamp

    written = 0
    for iid, val in (responses.get("responses") or {}).items():
        if iid not in se:
            continue
        try:
            v = int(val)
        except (TypeError, ValueError):
            continue
        if not (1 <= v <= 4):
            continue
        se[iid][phase] = v
        se[iid].setdefault("history", []).append({
            "stage": stage, "phase": phase, "value": v, "timestamp": ts,
        })
        written += 1
    return written


# ----- 파생값 유틸 (시각화에서 재사용) -----

def counter_to_level(count, thresholds):
    """CPS 하위구인 누적 카운트를 1~5 파생 레벨로 변환.

    thresholds 예: {"1":"0", "2":"1-2", "3":"3-4", "4":"5-7", "5":"8+"}
    """
    if not isinstance(count, (int, float)):
        return None
    try:
        c = int(count)
    except (TypeError, ValueError):
        return None
    for lv in ("5", "4", "3", "2", "1"):
        spec = thresholds.get(lv)
        if spec is None:
            continue
        s = str(spec).strip()
        if s.endswith("+"):
            if c >= int(s[:-1]):
                return int(lv)
        elif "-" in s:
            lo, hi = s.split("-")
            if int(lo) <= c <= int(hi):
                return int(lv)
        else:
            if c == int(s):
                return int(lv)
    return 1


def stage_grade_to_score(grade):
    """A~E 등급 → 1~5 점수."""
    mapping = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}
    if isinstance(grade, str) and grade.upper() in mapping:
        return mapping[grade.upper()]
    return None
