"""config/, prompts/, domain/ 하위의 JSON·MD·PDF 파일을 읽어 CONFIG, PROMPTS dict 반환."""

import json
from pathlib import Path

from .domain_loader import load_domain_knowledge


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_md(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_config(base_path="team4", domain_folder="domain", max_domain_chars=8000):
    base = Path(base_path)

    config = {
        "personas":              load_json(base / "config" / "personas.json"),
        "learner_model_schema":  load_json(base / "config" / "learner_model.json"),
        "tutor_model":           load_json(base / "config" / "tutor_model.json"),
        "tasks":                 load_json(base / "config" / "tasks.json"),
        "domain_knowledge":      load_domain_knowledge(
            base_path=base,
            folder=domain_folder,
            max_chars_per_doc=max_domain_chars,
        ),
    }

    prompts = {
        "analyze_and_decide": load_md(base / "prompts" / "00_analyze_and_decide.md"),
        "learner_analysis":   load_md(base / "prompts" / "01_learner_analysis.md"),
        "tutor_decision":     load_md(base / "prompts" / "02_tutor_decision.md"),
        "ai_student":         load_md(base / "prompts" / "03_ai_student_utterance.md"),
        "stage_intro":        load_md(base / "prompts" / "04_stage_intro.md"),
        "stage_closure":      load_md(base / "prompts" / "05_stage_closure.md"),
        "misconception":      load_md(base / "prompts" / "06_misconception_challenge.md"),
        "encouragement":      load_md(base / "prompts" / "07_encouragement.md"),
        "cps_tagging":        load_md(base / "prompts" / "08_cps_tagging.md"),
        "self_efficacy":      load_md(base / "prompts" / "09_self_efficacy_survey.md"),
        "completion_summary": load_md(base / "prompts" / "10_completion_summary.md"),
    }

    return config, prompts
