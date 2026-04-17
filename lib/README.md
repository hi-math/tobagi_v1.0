# team4 — 수학교육평가론 협력학습 MVP

사용자(Human) 1명 + AI 학생 2명이 함께 수업을 설계하는 협력학습 세션을 Claude API로 구동.
학습자 모델과 교수자 모델을 명시적으로 구현하고 Gradio 로 시각화한다.

## 폴더 구조

```
team4/
├── config/                 # 인물·학습자모델·교수자모델·과제 JSON
│   ├── personas.json
│   ├── learner_model.json
│   ├── tutor_model.json
│   └── tasks.json
├── prompts/                # Claude 프롬프트 템플릿 (MD)
│   ├── 01_learner_analysis.md
│   ├── 02_tutor_decision.md
│   ├── 03_ai_student_utterance.md
│   ├── 04_stage_intro.md
│   ├── 05_stage_closure.md
│   ├── 06_misconception_challenge.md
│   └── 07_encouragement.md
├── fonts/                  # 한글 폰트 (OFL 1.1 라이선스)
│   ├── NanumGothic.ttf
│   └── OFL.txt
├── __init__.py             # 패키지 진입점 + bootstrap()
├── config_loader.py        # JSON/MD 로더
├── learner_model.py        # 학습자 모델 인스턴스 초기화
├── llm_api.py              # Claude API 래퍼 + 프롬프트 유틸
├── session.py              # CollaborativeSession (3-stage)
├── visualize.py            # Matplotlib 시각화 + 한글 폰트
├── gradio_app.py           # Gradio Blocks UI (launch_ui)
└── cli_runner.py           # 레거시 CLI (run_session)
```

## Colab 실행 (권장)

```python
# 1. 의존성 + 리포 클론 (폰트는 리포에 동봉되어 apt 설치 불필요)
!pip install anthropic gradio -q
!git clone -q https://github.com/hi-math/team4.git   # 최초 1회

# 2. 경로 설정 (리포지토리 루트)
BASE_PATH = "team4"
MODEL     = "claude-sonnet-4-20250514"

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(BASE_PATH)) or ".")

# 3. 부트스트랩 + UI 기동
from google.colab import userdata
from team4 import bootstrap, launch_ui

ctx = bootstrap(base_path=BASE_PATH,
                api_key=userdata.get("CLAUDE_API_KEY"),
                model=MODEL,
                setup_fonts=True)     # team4/fonts/NanumGothic.ttf 자동 등록
launch_ui(**ctx, share=True)
```

## 한글 폰트

`team4/fonts/NanumGothic.ttf` (OFL 1.1) 가 패키지에 포함되어 있어 `apt-get install
fonts-nanum` 같은 시스템 설치가 필요 없습니다. 로컬(맥·윈도우)에서도 그대로 동작.

`setup_korean_font()` 우선순위:
1. 명시 `font_path` → 2. `team4/fonts/NanumGothic.ttf` (번들) → 3. 시스템 경로

## 모듈별 역할

| 모듈 | 주요 API |
|---|---|
| `config_loader` | `load_config(base_path)` → `(CONFIG, PROMPTS)` |
| `learner_model` | `init_learners(config)` → `{user, ai_1, ai_2}` |
| `llm_api` | `ClaudeAPI(client, model).call(prompt, ...)`, `extract_json`, `render_prompt` |
| `session` | `CollaborativeSession(config, prompts, learner_models, api)` |
| `visualize` | `radar_figure`, `history_figure`, `user_model_markdown`, `setup_korean_font`, `plot_radar_all`, `plot_user_history`, `print_user_model` |
| `gradio_app` | `launch_ui(*, config, prompts, learner_models, api, share=True)` |
| `cli_runner` | `run_session(*, config, prompts, learner_models, api)` |

## 세션 파이프라인 (CollaborativeSession)

```
user_turn(utt)
  ├─ [1/3] analyze_user_utterance(utt)   # 학습자 모델 업데이트
  ├─ [2/3] tutor_decision()              # AI 두 명의 directive 결정
  └─ [3/3] generate_ai_utterance x2      # AI 학생 발화 생성
```

`decision["stage_complete"] == True` 이면 `advance_stage()` → 다음 stage 인트로.
