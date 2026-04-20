# 학습자 모델 설계 문서 (v2.0)

> 수학교육평가론 과제 — Human 1명 + AI 2명 협력학습 시스템의 **학습자 모델** 설계.
> 인지적 요소 3개 + 정의적 요소 2개(+α) 구조로 재구성됨. 교수자 모델은 별도 문서.

---

## 1. 설계 개요

학습자 모델은 협력학습 세션에서 **사용자(실제 학생)의 상태**를 교수자 모델이 참조할 수 있도록 구조화하여 저장한다. v2.0에서는 과제에서 요구한 대로 **인지적 요소 / 정의적 요소** 2개 범주로 명시 분리하였다.

| 범주 | 모델 키 | 한글 이름 | 측정 방식 |
|---|---|---|---|
| 인지 | `task_achievement` | 과제 수행도달 | LLM 루브릭 판단 (Stage별 A~E) |
| 인지 | `math_communication` | 수학적 의사소통 | LLM 루브릭 판단 (1~5) |
| 인지 | `math_reasoning` | 수학적 추론 | LLM 루브릭 판단 (1~5) |
| 정의 | `cps` | 협력적 문제해결 | **Claude API 발화 다중 태깅** (누적 카운트) |
| 정의 | `self_efficacy` | 수학 자기효능감 | **학습자 자기보고** (Likert 4점) |
| +α | `extension_slot` | 추가 정의적 요소 | placeholder (math_anxiety, interest, attribution 후보) |

측정 방식이 세 가지로 다르다는 점이 v2.0의 핵심 변화다. CPS는 자기보고 태도 설문이 아닌 행동 관찰, self_efficacy는 발화 추정이 아닌 학습자 직접 응답으로 수집된다. 이는 각각 OECD(2017a)의 CPS 프레임워크와 Bandura(2006)의 domain-specificity 원칙을 따른 것이다.

---

## 2. 인지적 요소

### 2.1 `task_achievement` — 과제 수행도달
각 Stage에 부여된 `assessment_rubric`(A~E)을 기준으로 현재 도달 수준을 판단한다. v1.0의 `conceptual_understanding` + `procedural_fluency` + `misconceptions`를 통합해, **Stage 루브릭 중심**으로 단일화했다. 이유는 두 가지:

1. 과제의 성취기준 [9수01-01]은 Stage별 루브릭(A~E)로 서술되어 있어, 1~5 ordinal보다 이 A~E 등급이 직접적으로 대응된다.
2. 개념/절차를 분리 판단하면 동일한 발화로 두 값이 상충하는 문제가 잦았다(v1.0 운용 경험).

`misconceptions`는 `task_achievement` 하위 차원으로 유지해, 도달 등급을 설명하는 **질적 근거** 역할을 한다.

### 2.2 `math_communication` — 수학적 의사소통
표현의 명확성(`expression_clarity`)과 수학 용어 사용(`math_vocabulary_use`) 두 차원만 남겼다. v1.0의 `reasoning_quality`(논증의 수준)는 "의사소통"이 아니라 "추론" 구인에 속하므로 `math_reasoning`으로 분리.

### 2.3 `math_reasoning` — 수학적 추론
v1.0에는 없던 독립 모델. 세 차원:

- `justification_quality` — 주장에 대한 근거·정당화 수준
- `counterexample_handling` — 반례의 제시·이해·활용 수준
- `generalization` — 사례에서 규칙 추출·전이

이 세 차원은 이번 수업 주제(소수·합성수)에서 특히 관찰 가능성이 높다. 예: "홀수는 다 소수"에 대한 반례(9, 15) 활용, "약수가 2개인 수" 일반화 등.

---

## 3. 정의적 요소

### 3.1 `cps` — 협력적 문제해결 역량

**정의** (OECD, 2017a)
> 둘 이상의 행위자가 문제 해결에 필요한 이해와 노력을 공유하고, 지식·기술·노력을 모아 해결에 도달하는 과정에 효과적으로 참여하는 능력.

PISA 2015 프레임워크의 3개 협력 차원에 **repair moves** 를 추가해 4개 하위구인으로 조작화했다:

| 하위구인 | 행동 feature |
|---|---|
| `shared_understanding` | 친구 발화 재진술 / 기준 합의 / 문제 재정의 |
| `action_taking` | 전략 제안 / 예시 제시 / 반례 제시·검증 요청 |
| `team_organisation` | 역할 제안 / 순서 조정 / 조율 발화 |
| `repair_moves` | 오해 정정 / 기준 재합의 / 틀린 추론의 협력적 수정 |

**측정 방식**: 대화 로그 기반 Claude API 태깅 (`prompts/08_cps_tagging.md`). PISA CPS의 원형이 "자기보고식 태도 설문이 아닌 컴퓨터 기반 상호작용 과제"(OECD, 2017b)라는 점을 반영해, "친하게 협력했는가"가 아니라 **어떤 인지적·사회적 행동이 나타났는가**를 구조적으로 카운트한다.

- 매 사용자 발화 → Claude가 0개 이상의 태그 부여 → 해당 하위구인의 `value`(카운터) +1
- 각 태그는 `quote`, `evidence_feature`, `confidence`를 함께 저장 → 정성 분석에 활용
- 누적 카운트는 `derived_level_thresholds`로 1~5 등급으로도 환산(레이더 차트 시각화용)

**왜 단순 대화 길이가 아닌가?** 발화 빈도만 세면 "많이 말한 학생"이 과대평가된다. Dimension별 태깅은 *어떤 종류의* 협력 행동을 했는지를 분리한다.

### 3.2 `self_efficacy` — 수학 자기효능감

**정의** (Bandura, 2006)
> 특정 수학 과제를 성공적으로 수행할 수 있다는 자신의 능력에 대한 신념.

**원칙**:
- **Domain-specificity**: 일반적 자아존중감이 아닌, 이번 수업 맥락에 밀착된 문항.
- **Task-specificity** (Pajares & Miller, 1995): "나는 수학을 잘한다"가 아닌 "나는 이 문제를 풀 수 있다".
- **개념 숙련도와 분리**: 수행 결과(`task_achievement`)와 자기 신념(`self_efficacy`)은 **다른 construct**. 두 값이 엇갈리는 경우(예: 실제 수행은 C인데 self_efficacy는 높음) 자체가 의미 있는 교수적 정보.

**문항** (8개, `prompts/09_self_efficacy_survey.md`에 전문)
- Stage 1 관련 3문항 (소수/합성수 판단과 설명, 1의 특수성)
- Stage 2 관련 3문항 (약수쌍·배열·반례)
- Stage 3 관련 1문항 (자기검토)
- 전 Stage 공통 1문항 (동료와의 수학적 소통)

**척도**: Likert 4점 (전혀 자신 없음 / 별로 자신 있음 / 자신 있음 / 매우 자신 있음).
**수집 시점**: 각 Stage 시작 직후(pre) + 종료 시점(post).

### 3.3 `extension_slot` — +α
추후 추가될 정의적 요소. 현재 `placeholder` 상태이며 후보는:
- `math_anxiety` (수학 불안) — 자기효능감과 분리된 정서적 부담
- `interest_engagement` (흥미·몰입) — v1.0 `emotional_state.engagement`의 확장
- `attribution_style` (귀인 양식) — 성공/실패 원인 귀인 패턴

---

## 4. 업데이트 로직

### 4.1 흐름 요약
```
user_turn(utterance)
  │
  ├─ [1] analyze_user_utterance(utt)        # prompts/01_learner_analysis.md
  │      → task_achievement, math_communication, math_reasoning 업데이트
  │      → misconceptions 추가/제거
  │
  ├─ [2] tag_cps_features(utt)              # prompts/08_cps_tagging.md
  │      → cps.<dim>.value 카운터 가산
  │
  ├─ [3] (optional) ask_self_efficacy()     # prompts/09_self_efficacy_survey.md
  │      → stage 전/후 시점에만 호출
  │      → self_efficacy.items.<id>.pre|post 기록
  │
  ├─ [4] tutor_decision()                   # 교수자 모델이 위 모델을 읽고 결정
  │
  └─ [5] generate_ai_utterance x2            # 민준, 서연 발화
```

### 4.2 타입별 업데이트 규약

| type | 업데이트 방식 |
|---|---|
| `ordinal` (1~5) | LLM이 `new_value` 제안 → ±1 이내로 clip → `history` append |
| `list` (misconceptions) | `added`는 중복 제거 후 concat, `removed`는 set difference |
| `counter` (cps) | 태그 1개당 +1. 감산 없음. 세션 단위 누적 |
| `stage_categorical` (stage_level) | 현재 Stage의 값만 갱신. `per_stage_values[stage]`에 누적 |
| `likert_self_report` (self_efficacy) | `pre`/`post` 필드에 덮어쓰기, `history`는 append-only |

### 4.3 변경이 없는 경우의 처리
- `01_learner_analysis.md`: 근거가 없는 차원은 `updates` 배열에서 생략.
- `08_cps_tagging.md`: 중립 발화는 `"tags": [], "none": true`. 카운터 가산 없음.
- `09_self_efficacy_survey.md`: 응답 누락은 null 유지, 평균 계산에서 제외.

---

## 5. v1.0 → v2.0 마이그레이션 노트

| v1.0 위치 | v2.0 위치 | 비고 |
|---|---|---|
| `cognitive_state.conceptual_understanding` | `task_achievement.stage_level` | 1~5 → A~E로 루브릭 전환 |
| `cognitive_state.procedural_fluency` | `task_achievement.stage_level`에 통합 | 분리 판단 상충 문제 해소 |
| `cognitive_state.misconceptions` | `task_achievement.misconceptions` | 위치만 이동 |
| `math_communication.expression_clarity` | 동일 | 유지 |
| `math_communication.reasoning_quality` | `math_reasoning.justification_quality` | 구인 이동 |
| `math_communication.math_vocabulary_use` | 동일 | 유지 |
| `emotional_state.*` | `extension_slot.candidates` | 추후 복귀 가능 |
| `collaboration.*` | `cps.*` (재설계) | 1~5 ordinal → 카운터 + 파생 레벨 |
| (없음) | `self_efficacy` | 신설 |
| (없음) | `math_reasoning.counterexample_handling` | 신설 |
| (없음) | `math_reasoning.generalization` | 신설 |

**하위호환 주의**: v1.0 인스턴스를 직접 로드하면 `KeyError`. 기존 세션 스냅샷이 있다면 변환 스크립트 필요 (현재 MVP 범위에서는 해당 스냅샷 없음으로 생략).

---

## 6. 시각화 지침 (후속 작업 연결)

- **레이더 차트**: 5개 모델을 방사형으로. 각 모델은 하위 차원 평균 또는 파생 레벨로 정규화해 1~5 스케일로 표시.
  - `task_achievement.stage_level`: A~E → 5~1로 수치화
  - `math_communication` / `math_reasoning`: 하위 차원 평균
  - `cps`: 4 하위 카운터를 각각 `derived_level_thresholds`로 1~5 변환 후 평균
  - `self_efficacy`: 전체 문항 평균 (1~4를 1~5로 선형 매핑 또는 그대로 4점 표시)

- **Stage별 타임라인**: `task_achievement.stage_level.per_stage_values` 를 막대 또는 라인으로.
- **CPS 행동 분포**: 4 하위구인 카운터를 스택/바 차트로. 시간대별 누적도 가능.
- **자기효능감 pre/post 비교**: 문항별 slope chart 또는 Stage별 평균 비교.

---

## 7. 참고문헌

- Bandura, A. (2006). Guide for constructing self-efficacy scales. In F. Pajares & T. Urdan (Eds.), *Self-efficacy beliefs of adolescents* (pp. 307-337). Information Age Publishing.
- OECD (2017a). *PISA 2015 assessment and analytical framework: Science, reading, mathematic, financial literacy and collaborative problem solving*. OECD Publishing.
- OECD (2017b). *PISA 2015 results (Volume V): Collaborative problem solving*. OECD Publishing.
- OECD (2022). *PISA 2022 technical report*.
- Pajares, F., & Miller, M. D. (1995). Mathematics self-efficacy and mathematics performances: The need for specificity of assessment. *Journal of Counseling Psychology*, 42(2), 190-198.
