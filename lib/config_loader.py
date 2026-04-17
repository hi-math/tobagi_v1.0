"""config/ 및 prompts/ 하위의 JSON·MD 파일을 읽어 CONFIG, PROMPTS dict 반환."""

import json
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_md(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_config(base_path="team4"):
    """base_path/config, base_path/prompts 에서 모든 설정과 프롬프트를 로드.

    Args:
        base_path: team4 루트 경로 (문자열 또는 Path)

    Returns:
        (CONFIG, PROMPTS) tuple
            CONFIG  — personas, learner_model_schema, tutor_model, tasks
            PROMPTS — 7개 프롬프트 템플릿 문자열
    """
    base = Path(base_path)

    config = {
        "personas":              load_json(base / "config" / "personas.json"),
        "learner_model_schema":  load_json(base / "config" / "learner_model.json"),
        "tutor_model":           load_json(base / "config" / "tutor_model.json"),
        "tasks":                 load_json(base / "config" / "tasks.json"),
    }

    prompts = {
        "learner_analysis":  load_md(base / "prompts" / "01_learner_analysis.md"),
        "tutor_decision":    load_md(base / "prompts" / "02_tutor_decision.md"),
        "ai_student":        load_md(base / "prompts" / "03_ai_student_utterance.md"),
        "stage_intro":       load_md(base / "prompts" / "04_stage_intro.md"),
        "stage_closure":     load_md(base / "prompts" / "05_stage_closure.md"),
        "misconception":     load_md(base / "prompts" / "06_misconception_challenge.md"),
        "encouragement":     load_md(base / "prompts" / "07_encouragement.md"),
    }

    return config, prompts
