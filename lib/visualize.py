"""한글 폰트 설정 + 학습자 모델 시각화 (Matplotlib).

Gradio 에서는 `*_figure()` (Figure 반환), CLI 에서는 `plot_*()` (plt.show) 를 사용한다.
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# --------------------------------------------------------------------
# 한글 폰트
# --------------------------------------------------------------------

# team4/fonts/NanumGothic.ttf — 패키지에 동봉된 기본 폰트 (OFL 1.1)
_BUNDLED_FONT = os.path.join(os.path.dirname(__file__), "fonts", "NanumGothic.ttf")

# 시스템 폰트 탐색 경로 (리눅스/맥/윈도우 순)
_SYSTEM_FALLBACKS = (
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/Library/Fonts/NanumGothic.ttf",
    "C:/Windows/Fonts/NanumGothic.ttf",
)


def setup_korean_font(font_path=None, verbose=True):
    """Matplotlib 한글 폰트(NanumGothic) 등록.

    우선순위:
        1. 명시된 `font_path`
        2. `team4/fonts/NanumGothic.ttf` (패키지 번들)
        3. 시스템 설치본 (apt `fonts-nanum`, 맥/윈도우 기본 경로)

    모두 실패하면 경고만 출력하고 조용히 넘어간다.

    Returns:
        실제로 등록된 폰트 파일 경로 (str) 또는 실패 시 None
    """
    candidates = []
    if font_path:
        candidates.append(font_path)
    candidates.append(_BUNDLED_FONT)
    candidates.extend(_SYSTEM_FALLBACKS)

    for path in candidates:
        if not path or not os.path.exists(path):
            continue
        try:
            fm.fontManager.addfont(path)
            matplotlib.rcParams["font.family"] = "NanumGothic"
            matplotlib.rcParams["axes.unicode_minus"] = False
            if verbose:
                print(f"✅ 한글 폰트 등록: {path}")
            return path
        except Exception as e:
            if verbose:
                print(f"⚠️ {path} 로드 실패: {e}")

    if verbose:
        print("⚠️ 한글 폰트를 찾지 못했습니다. 차트의 한글이 □ 로 표시될 수 있습니다.")
    return None


# --------------------------------------------------------------------
# 콘솔 출력
# --------------------------------------------------------------------

def print_user_model(config, learner_models):
    m = learner_models["user"]
    print(f"\n{'═'*55}\n📊 {m['student_name']}의 학습자 모델\n{'═'*55}")
    for mk, mv in config["learner_model_schema"]["models"].items():
        print(f"\n  ▸ {mv['name']}")
        for dk, dv in mv["dimensions"].items():
            val = m["models"][mk][dk]["value"]
            if isinstance(val, (int, float)):
                bar = "█" * int(val) + "░" * (5 - int(val))
                print(f"    [{bar}] {dv['name']}: {val}/5")
            elif isinstance(val, list) and val:
                print(f"    {dv['name']}: {val}")


# --------------------------------------------------------------------
# 레이더 차트
# --------------------------------------------------------------------

def radar_figure(config, learner_models):
    """사용자·AI 두 명의 학습자 모델을 레이더로 비교. Figure 반환."""
    labels, vals = [], {"user": [], "ai_1": [], "ai_2": []}
    for mk, mv in config["learner_model_schema"]["models"].items():
        for dk, dv in mv["dimensions"].items():
            sample = learner_models["user"]["models"][mk][dk]["value"]
            if isinstance(sample, (int, float)):
                labels.append(dv["name"])
                for sk in ("user", "ai_1", "ai_2"):
                    vals[sk].append(learner_models[sk]["models"][mk][dk]["value"])

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    colors = {"user": "#E74C3C", "ai_1": "#3498DB", "ai_2": "#2ECC71"}
    names = {
        "user": "사용자",
        "ai_1": config["personas"]["ai_students"]["ai_1"]["name"],
        "ai_2": config["personas"]["ai_students"]["ai_2"]["name"],
    }
    for sk in ("user", "ai_1", "ai_2"):
        v = vals[sk] + vals[sk][:1]
        ax.plot(angles, v, "o-", linewidth=2, label=names[sk], color=colors[sk])
        ax.fill(angles, v, alpha=0.15, color=colors[sk])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=9)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    ax.set_title("학습자 모델 비교 (Radar)", size=13, pad=20, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_radar_all(config, learner_models):
    """CLI/노트북 표시용: Figure 생성 후 plt.show()."""
    fig = radar_figure(config, learner_models)
    plt.show()
    return fig


# --------------------------------------------------------------------
# 사용자 변화 추이
# --------------------------------------------------------------------

def history_figure(config, learner_models):
    """사용자의 영역별(인지·의사소통·감정·협력) 변화 추이. Figure 반환."""
    schema = config["learner_model_schema"]["models"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    for ax, (mk, mv) in zip(axes, schema.items()):
        for dk, dv in mv["dimensions"].items():
            hist = learner_models["user"]["models"][mk][dk]["history"]
            if hist and isinstance(hist[0]["value"], (int, float)):
                xs = list(range(len(hist)))
                ys = [h["value"] for h in hist]
                ax.plot(xs, ys, "o-", label=dv["name"], linewidth=2, markersize=6)
        ax.set_title(mv["name"], fontsize=12, fontweight="bold")
        ax.set_ylim(0, 5.5)
        ax.set_xlabel("업데이트 회차")
        ax.set_ylabel("수준")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("사용자 학습자 모델 — 단계별 변화", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_user_history(config, learner_models):
    fig = history_figure(config, learner_models)
    plt.show()
    return fig


# --------------------------------------------------------------------
# Markdown 요약 (Gradio 탭 용)
# --------------------------------------------------------------------

def user_model_markdown(config, learner_models):
    """사용자 학습자 모델을 Markdown 문자열로 반환."""
    m = learner_models["user"]
    lines = [f"### 📊 {m['student_name']}의 학습자 모델"]
    for mk, mv in config["learner_model_schema"]["models"].items():
        lines.append(f"\n**▸ {mv['name']}**")
        for dk, dv in mv["dimensions"].items():
            val = m["models"][mk][dk]["value"]
            if isinstance(val, (int, float)):
                bar = "█" * int(val) + "░" * (5 - int(val))
                lines.append(f"- `[{bar}]` {dv['name']}: **{val}/5**")
            elif isinstance(val, list) and val:
                lines.append(f"- {dv['name']}: {', '.join(map(str, val))}")
    return "\n".join(lines)
