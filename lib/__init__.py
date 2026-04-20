"""team4 — 수학교육평가론 협력학습 MVP 패키지.

모듈 구성:
    config_loader : JSON/MD 설정·프롬프트 로더
    learner_model : 학습자 모델 인스턴스 초기화
    llm_api       : Claude API 래퍼 + 프롬프트 유틸
    session       : CollaborativeSession (3-stage 파이프라인)
    visualize     : 한글 폰트 + 레이더/히스토리/모델 표시
    gradio_app    : Gradio 채팅 인터페이스
    cli_runner    : 레거시 CLI 런너 (디버깅용)

노트북에서의 일반적인 사용 패턴:

    from team4 import bootstrap, launch_ui
    ctx = bootstrap(base_path="team4",
                    api_key=userdata.get("CLAUDE_API_KEY"))
    launch_ui(**ctx, share=True)
"""

from .config_loader import load_config, load_json, load_md
from .learner_model import create_learner_model_instance, init_learners
from .llm_api import ClaudeAPI, extract_json, render_prompt, DEFAULT_HAIKU, DEFAULT_SONNET
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
    "bootstrap",
    # config
    "load_config", "load_json", "load_md",
    # learner model
    "create_learner_model_instance", "init_learners",
    # llm
    "ClaudeAPI", "extract_json", "render_prompt",
    "DEFAULT_HAIKU", "DEFAULT_SONNET",
    # session
    "CollaborativeSession",
    # visualize
    "setup_korean_font", "print_user_model",
    "plot_radar_all", "plot_user_history",
    "radar_figure", "history_figure", "user_model_markdown",
    # runners
    "launch_ui", "run_session",
]


def bootstrap(base_path, api_key, model=DEFAULT_HAIKU,
              setup_fonts=False):
    """한 번의 호출로 config/prompts/학습자모델/API 객체를 모두 생성.

    Args:
        base_path:  team4 리포지토리 루트 경로 (config/, prompts/ 하위 포함)
        api_key:    Anthropic API 키
        model:      Claude 모델 식별자 (기본: Haiku 4.5 — 빠른 응답)
                    옵션:
                      - "claude-haiku-4-5-20251001"   (DEFAULT_HAIKU, 빠름)
                      - "claude-sonnet-4-20250514"    (DEFAULT_SONNET, 품질)
                      - 그 외 Anthropic 모델 식별자
        setup_fonts: True면 NanumGothic 폰트 등록까지 실행

    Returns:
        dict with keys: config, prompts, learner_models, api
        (launch_ui / run_session 에 **ctx 로 언패킹해서 전달 가능)
    """
    import anthropic

    config, prompts = load_config(base_path)
    learner_models = init_learners(config)
    client = anthropic.Anthropic(api_key=api_key)
    api = ClaudeAPI(client, model=model)
    if setup_fonts:
        import os as _os
        setup_korean_font(_os.path.join(base_path, "fonts", "NanumGothic.ttf"))
    return {
        "config": config,
        "prompts": prompts,
        "learner_models": learner_models,
        "api": api,
    }
