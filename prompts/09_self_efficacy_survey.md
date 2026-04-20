# 수학 자기효능감 자기보고 설문 (v1.0)

Bandura(2006)의 과제-특이적 자기효능감 측정 원칙과 domain-specificity 권고(Pajares & Miller, 1995)에 따라, 이번 수업(소수·합성수)에 밀착된 8개 문항의 Likert 4점 척도 자기보고 도구. **발화 분석이나 LLM 추정으로 대체하지 않는다.** 반드시 학습자의 직접 응답으로만 수집한다.

## 측정 시점
- **pre**: 각 Stage 시작 직후 (해당 Stage와 관련된 문항만 제시)
- **post**: 각 Stage 종료 시점 (동일 문항)
- 세션 종료 후 전체 8문항 일괄 수집도 가능(옵션)

## Likert 척도
| 값 | 라벨 |
|---|---|
| 1 | 전혀 자신 없음 |
| 2 | 별로 자신 있음 |
| 3 | 자신 있음 |
| 4 | 매우 자신 있음 |

> 주의: Bandura 권고에 따라 "나는 수학을 잘한다" 같은 포괄적 문항이 아닌, **"나는 이 과제를 할 수 있다"** 형태의 구체적 수행 문항으로 구성했다.

## 문항 (Stage 매핑 포함)

| ID | 문항 | Stage |
|---|---|---|
| se_01 | 어떤 수가 소수인지 합성수인지 판단할 수 있다. | 1 |
| se_02 | 어떤 수가 소수인지 합성수인지 이유와 함께 설명할 수 있다. | 1 |
| se_03 | 1이 왜 소수가 아닌지 설명할 수 있다. | 1 |
| se_04 | 어떤 수를 약수쌍으로 나타내어 소수인지 합성수인지 판단할 수 있다. | 2 |
| se_05 | 어떤 수를 배열(직사각형 만들기)로 생각해 소수인지 합성수인지 판단할 수 있다. | 2 |
| se_06 | 친구가 "홀수는 다 소수다"라고 말했을 때, 반례를 들어 설명할 수 있다. | 2 |
| se_07 | 내 생각이 맞는지 확인하기 위해 다른 방법으로 다시 검토할 수 있다. | 3 |
| se_08 | 민준이나 서연이와 이야기하면서 내 생각을 수학적으로 분명하게 설명할 수 있다. | all |

## 학습자에게 제시되는 안내 문구

```
잠깐! 지금 이 순간, 아래 각 문장에 대해 얼마나 자신 있는지 골라 줄래?
맞았는지 틀렸는지 평가하는 게 아니라, **네가 지금 얼마나 자신 있는지** 를 알려주는 거야.
정답은 없어. 네 솔직한 느낌을 골라줘.

각 문장을 읽고 4개 중 하나를 선택:
(1) 전혀 자신 없음   (2) 별로 자신 있음   (3) 자신 있음   (4) 매우 자신 있음
```

## 응답 수집 형식 (시스템 입력 → 저장 구조)

```json
{
  "phase": "pre",           // "pre" | "post"
  "stage": 1,
  "timestamp": "2026-04-20T14:30:00+09:00",
  "responses": {
    "se_01": 3,
    "se_02": 2,
    "se_03": 1
  }
}
```

## 저장 스키마 (learner_model.user.models.self_efficacy)

```
self_efficacy:
  items:
    se_01:
      pre:  3        # 최신 pre 응답
      post: 4        # 최신 post 응답
      history:
        - {stage: 1, phase: "pre",  value: 3, timestamp: "..."}
        - {stage: 1, phase: "post", value: 4, timestamp: "..."}
    se_02:
      ...
  aggregate:
    stage_1_pre_mean:  2.33
    stage_1_post_mean: 3.67
    overall_pre_mean:  2.5
    overall_post_mean: 3.5
```

## 분석·시각화 가이드

1. **pre/post 비교**: 각 문항의 post−pre 차이로 자기효능감 변동 추정. 양수면 신념 상승.
2. **개념 숙련도와의 분리**: Bandura(2006)·Pajares & Miller(1995) 권고대로 self_efficacy는 `task_achievement`(실제 수행 결과)와 **분리된 feature**로 보고한다. 두 값이 엇갈리는 경우(예: task_achievement=C지만 se_mean=3.8) 자체가 의미 있는 지표.
3. **Stage별 문항 제시**: stage 1 진행 중에는 se_01~se_03만 제시. se_08(소통)은 매 stage 또는 세션 종료 시점에 제시.
4. **결측 처리**: 응답 누락 시 해당 문항의 pre/post는 null 유지. 평균 계산에서 제외.

## 참고문헌
- Bandura, A. (2006). Guide for constructing self-efficacy scales. In F. Pajares & T. Urdan (Eds.), *Self-efficacy beliefs of adolescents* (pp. 307-337). Information Age Publishing.
- Pajares, F., & Miller, M. D. (1995). Mathematics self-efficacy and mathematics performances: The need for specificity of assessment. *Journal of Counseling Psychology*, 42(2), 190-198.
- OECD (2022). *PISA 2022 technical report*.
