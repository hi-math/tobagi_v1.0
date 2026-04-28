"""team4 — 수학교육평가론 협력학습 MVP 패키지.

모듈 구성:
    config_loader : JSON/MD 설정·프롬프트 로더
    learner_model : 학습자 모델 인스턴스 초기화
    llm_api       : LLM API 래퍼 (Gemini / Claude) + 프롬프트 유틸
    session       : CollaborativeSession (3-stage 파이프라인)
    visualize     : 한글 폰트 + 레이더/히스토리/모델 표시
    gradio_app    : Gradio 채팅 인터페이스
    cli_runner    : 레거시 CLI 런너 (디버깅용)

노트북에서의 일반적인 사용 패턴:

    from team4 import bootstrap, launch_ui
    # OpenAI (기본 권장 — RECITATION 필터 없음)
    ctx = bootstrap(base_path="team4",
                    api_key=userdata.get("OPENAI_API_KEY"),
                    provider="openai",
                    model="gpt-4o-mini")

    # Gemini를 쓰려면:
    # ctx = bootstrap(..., provider="gemini",  api_key=userdata.get("GEMINI_API_KEY"))
    # Claude를 쓰려면:
    # ctx = bootstrap(..., provider="anthropic", api_key=userdata.get("CLAUDE_API_KEY"))

    launch_ui(**ctx, share=True)
"""

# ============================================================
# 버전 — 매 커밋마다 +0.01 수동 증가 (단일 source of truth)
# 이 값이 Gradio UI 상단에 자동 표시된다.
# ============================================================
__version__ = "v1.81"

from .config_loader import load_config, load_json, load_md
from .learner_model import create_learner_model_instance, init_learners
from .llm_api import (
    ClaudeAPI, GeminiAPI, OpenAIAPI,
    extract_json, render_prompt,
    DEFAULT_HAIKU, DEFAULT_SONNET,
    DEFAULT_GEMINI_FLASH, DEFAULT_GEMINI_FLASH_LITE, DEFAULT_GEMINI_25,
    DEFAULT_OPENAI_MINI, DEFAULT_OPENAI_FULL,
)
from .session import CollaborativeSession
from .visualize import (
    setup_korean_font,
    print_user_model,
    plot_radar_all,
    plot_user_history,
    radar_figure,
    history_figure,
    user_model_markdown,
)
from .gradio_app import launch_ui
from .cli_runner import run_session

__all__ = [
    "bootstrap", "__version__",
    # config
    "load_config", "load_json", "load_md",
    # learner model
    "create_learner_model_instance", "init_learners",
    # llm
    "ClaudeAPI", "GeminiAPI", "OpenAIAPI",
    "extract_json", "render_prompt",
    "DEFAULT_HAIKU", "DEFAULT_SONNET",
    "DEFAULT_GEMINI_FLASH", "DEFAULT_GEMINI_FLASH_LITE", "DEFAULT_GEMINI_25",
    "DEFAULT_OPENAI_MINI", "DEFAULT_OPENAI_FULL",
    # session
    "CollaborativeSession",
    # visualize
    "setup_korean_font", "print_user_model",
    "plot_radar_all", "plot_user_history",
    "radar_figure", "history_figure", "user_model_markdown",
    # runners
    "launch_ui", "run_session",
]


def bootstrap(base_path, api_key, model=None, provider="openai",
              setup_fonts=False):
    """한 번의 호출로 config/prompts/학습자모델/API 객체를 모두 생성.

    Args:
        base_path:  리포지토리 루트 경로 (config/, prompts/ 하위 포함)
        api_key:    LLM 공급자 API 키 (provider에 맞게)
        model:      모델 식별자. None이면 provider에 맞는 기본값 사용
                      openai 기본:    DEFAULT_OPENAI_MINI ("gpt-4o-mini")
                      gemini 기본:    DEFAULT_GEMINI_FLASH ("gemini-2.5-flash")
                      anthropic 기본: DEFAULT_HAIKU ("claude-haiku-4-5-...")
        provider:   "openai"(기본, RECITATION 없음) / "gemini" / "anthropic"
        setup_fonts: True면 NanumGothic 폰트 등록까지 실행

    Returns:
        dict with keys: config, prompts, learner_models, api
        (launch_ui / run_session 에 **ctx 로 언패킹해서 전달 가능)
    """
    config, prompts = load_config(base_path)
    learner_models = init_learners(config)

    provider = (provider or "openai").lower()
    if provider == "openai":
        api = OpenAIAPI(model=model or DEFAULT_OPENAI_MINI, api_key=api_key)
    elif provider == "gemini":
        # 신규 SDK `google-genai` 사용. Client 생성자에 직접 api_key를 주입.
        api = GeminiAPI(model=model or DEFAULT_GEMINI_FLASH, api_key=api_key)
    elif provider in ("anthropic", "claude"):
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        api = ClaudeAPI(client, model=model or DEFAULT_HAIKU)
    else:
        raise ValueError(
            f"Unknown provider: {provider!r} (openai | gemini | anthropic)"
        )

    if setup_fonts:
        import os as _os
        setup_korean_font(_os.path.join(base_path, "fonts", "NanumGothic.ttf"))

    print(f"[bootstrap] provider={provider}, model={api.model}")
    return {
        "config": config,
        "prompts": prompts,
        "learner_models": learner_models,
        "api": api,
    }
