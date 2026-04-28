"""학습자 모델 인스턴스 생성 및 업데이트 로직 (v2.1).

스키마: config/learner_model.json (schema_version >= 2.1)
  - 인지: task_achievement, math_communication
  - 정의: cps, self_efficacy
  - extension_slot: +α placeholder

v2.1 변경: math_reasoning 모델 제거 (추론 구인은 task_achievement 루브릭에 흡수)

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

    # 체크포인트 진행도 초기화 (Stage별 빈 dict)
    inst["checkpoint_progress"] = {"1": {}, "2": {}, "3": {}}
    # 하위 수준 AI의 '지연 학습'을 위한 관찰 카운터
    inst["_checkpoint_obs_count"] = {"1": {}, "2": {}, "3": {}}

    return inst


# AI 학생의 level별 초기 체크포인트 지식 시딩.
# 상(민준): 거의 전부 알고 있음 — 개념 설명자 역할
# 중(서연): 필수만 — 진행자, 학습자와 비슷한 수준에서 정리·연결
# 하(연우): 어휘(s1-1)만 — 질문자, 학습 과정에서 성장
_AI_INITIAL_CHECKPOINTS_BY_LEVEL = {
    "상": {
        "1": ["s1-1", "s1-2", "s1-3", "s1-4", "s1-5", "s1-6"],
        "2": ["s2-1", "s2-2", "s2-3", "s2-4", "s2-5", "s2-6", "s2-7", "s2-8"],
        "3": ["s3-1", "s3-2", "s3-3", "s3-4", "s3-5", "s3-6", "s3-7", "s3-8"],
    },
    "중": {
        "1": ["s1-1", "s1-2", "s1-3"],
        "2": ["s2-1", "s2-2", "s2-3", "s2-4"],
        "3": ["s3-1", "s3-2", "s3-3"],
    },
    "하": {
        "1": ["s1-1"],
        "2": [],
        "3": [],
    },
}


def _seed_ai_checkpoints(inst, level):
    """AI 학생의 학습자 모델에 level별 초기 체크포인트 지식 설정."""
    prior = _AI_INITIAL_CHECKPOINTS_BY_LEVEL.get(level, _AI_INITIAL_CHECKPOINTS_BY_LEVEL["중"])
    prog = inst.setdefault("checkpoint_progress", {"1": {}, "2": {}, "3": {}})
    for stage_key, cp_ids in prior.items():
        stage_prog = prog.setdefault(stage_key, {})
        for cid in cp_ids:
            stage_prog[cid] = {
                "hit": True,
                "first_seen_turn": 0,
                "source": "prior",   # 초기 사전지식
            }


def init_learners(config):
    """사용자 1명 + AI 학생 전원의 학습자 모델 dict 반환.

    반환 키: "user" + personas.json에 정의된 모든 ai_* (ai_1, ai_2, ai_3, ...)
    각 페르소나의 initial_learner_state / initial_self_efficacy를 override로 반영.
    """
    ai_students = config["personas"]["ai_students"]
    learners = {"user": create_learner_model_instance(config, "또바기")}
    for aid, info in ai_students.items():
        learners[aid] = create_learner_model_instance(
            config,
            info.get("name", aid),
            override_initial=info.get("initial_learner_state"),
            override_self_efficacy=info.get("initial_self_efficacy"),
        )
        # AI 체크포인트 사전지식 시딩
        _seed_ai_checkpoints(learners[aid], info.get("level", "중"))
    return learners


# ----- 체크포인트 적용·전파 -----

def apply_checkpoint_hits(inst, hit_ids, stage, turn=None, source="user"):
    """사용자의 이번 발화에서 탐지된 체크포인트 id들을 학습자 모델에 기록.
    중복은 무시 (idempotent). 처음 포착된 것만 카운트해 반환.
    """
    if not hit_ids:
        return 0
    prog = inst.setdefault("checkpoint_progress", {"1": {}, "2": {}, "3": {}})
    stage_prog = prog.setdefault(str(stage), {})
    added = 0
    for cid in hit_ids:
        if cid not in stage_prog:
            stage_prog[cid] = {
                "hit": True,
                "first_seen_turn": turn,
                "source": source,
            }
            added += 1
    return added


def propagate_checkpoints_to_ai(learner_models, hit_ids, stage, turn, ai_levels):
    """사용자가 이번 턴에 hit한 체크포인트를 AI 학생들에게 level별 규칙으로 전파.

    규칙 (peer learning simulation):
      - 상(민준): 이미 알고 있음 (source="prior"로 seed됨) → 추가 기록 없음
      - 중(서연): 첫 관찰 즉시 학습 (source="observed")
      - 하(연우): 누적 관찰 2회 이상이면 학습 (source="learned")

    반환: {aid: [새로 hit된 cp_id, ...]} — 방금 학습한 체크포인트 목록
    """
    newly_learned = {aid: [] for aid in ai_levels.keys()}
    if not hit_ids:
        return newly_learned

    stage_key = str(stage)

    for aid, level in ai_levels.items():
        inst = learner_models.get(aid)
        if not inst:
            continue
        prog = inst.setdefault("checkpoint_progress", {"1": {}, "2": {}, "3": {}})
        stage_prog = prog.setdefault(stage_key, {})
        obs_root = inst.setdefault("_checkpoint_obs_count", {"1": {}, "2": {}, "3": {}})
        obs_stage = obs_root.setdefault(stage_key, {})

        for cid in hit_ids:
            if cid in stage_prog:
                continue  # 이미 알고 있음

            if level == "상":
                # 개념 설명자는 기본적으로 알고 있으나, 시딩에서 누락된 경우를 대비해 즉시 기록.
                stage_prog[cid] = {"hit": True, "first_seen_turn": turn, "source": "observed"}
                newly_learned[aid].append(cid)
            elif level == "중":
                stage_prog[cid] = {"hit": True, "first_seen_turn": turn, "source": "observed"}
                newly_learned[aid].append(cid)
            elif level == "하":
                obs_stage[cid] = obs_stage.get(cid, 0) + 1
                if obs_stage[cid] >= 2:
                    stage_prog[cid] = {
                        "hit": True, "first_seen_turn": turn, "source": "learned",
                        "obs_count": obs_stage[cid],
                    }
                    newly_learned[aid].append(cid)
    return newly_learned


def known_checkpoint_ids(inst, stage):
    """stage 기준으로 이 학습자가 '알고 있다고 볼 수 있는' cp_id 리스트."""
    prog = (inst.get("checkpoint_progress") or {}).get(str(stage)) or {}
    return [cid for cid, v in prog.items() if isinstance(v, dict) and v.get("hit")]


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


def apply_cps_tags(inst, tags_result, stage, turn=None, min_confidence=0.4):
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


def apply_self_efficacy_signal(inst, signal_list, stage, turn=None):
    """Live 세션용: analyze_and_decide가 뽑아낸 self_efficacy_delta 리스트를
    반영한다. Bandura 권고상 자기보고가 원칙이지만 데모 파이프라인에서 설문 UI가
    없을 때 LLM이 발화 tone에서 추론한 신호로 pre 값을 움직여 시각적 변화를 낸다.

    signal_list 형식:
      [{"item_id": "se_02", "delta": +1, "reason": "..."}]

    적용 규칙:
      - delta는 -1 / +1만 허용 (그 외는 clip).
      - pre 값이 null이면 기본 2(약간 자신 없음)에서 출발해 적용.
      - 최종값은 1~4 범위로 clip.
      - history에 {stage, phase='live', value, delta, reason, turn} append.
    반환: 적용된 signal 수.
    """
    se = inst["models"].get("self_efficacy")
    if not se or not isinstance(signal_list, list):
        return 0

    applied = 0
    for sig in signal_list:
        iid = sig.get("item_id")
        if iid not in se:
            continue
        try:
            delta = int(sig.get("delta", 0))
        except (TypeError, ValueError):
            continue
        if delta == 0:
            continue
        delta = max(-1, min(1, delta))  # ±1로 clip

        cur = se[iid].get("pre")
        if cur is None:
            cur = 2
        new_val = max(1, min(4, int(cur) + delta))
        se[iid]["pre"] = new_val  # pre 값을 현재값으로 업데이트 (live 추적)
        se[iid].setdefault("history", []).append({
            "stage": stage,
            "phase": "live",
            "value": new_val,
            "delta": delta,
            "reason": sig.get("reason", ""),
            "turn": turn,
        })
        applied += 1
    return applied


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
