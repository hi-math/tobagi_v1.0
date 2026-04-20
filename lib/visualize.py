"""한글 폰트 설정 + 학습자 모델 시각화 (Matplotlib).

Gradio 에서는 `*_figure()` (Figure 반환), CLI 에서는 `plot_*()` (plt.show) 를 사용한다.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches


# --------------------------------------------------------------------
# 한글 폰트
# --------------------------------------------------------------------

def setup_korean_font(font_path="/usr/share/fonts/truetype/nanum/NanumGothic.ttf"):
    """Matplotlib 한글 폰트(NanumGothic) 등록.

    Colab 에서는 먼저 `\!apt-get -qq install fonts-nanum` 을 실행해야 파일이 존재한다.
    실패하면 경고만 출력하고 조용히 넘어간다.
    """
    try:
        fm.fontManager.addfont(font_path)
        matplotlib.rcParams["font.family"] = "NanumGothic"
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception as e:
        print(f"⚠️ 한글 폰트 로드 실패 ({font_path}): {e}")


# --------------------------------------------------------------------
# 콘솔 출력
# --------------------------------------------------------------------

def print_user_model(config, learner_models):
    m = learner_models["user"]
    print(f"\n{'═'*55}\n📊 {m['student_name']}의 학습자 모델\n{'═'*55}")
    for mk, mv in config["learner_model_schema"]["models"].items():
        print(f"\n  ▸ {mv['name']}")
        for dk, dv in (mv.get("dimensions") or {}).items():
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
    """사용자의 학습자 모델을 레이더로 표시. Figure 반환."""
    labels, vals = [], []
    for mk, mv in config["learner_model_schema"]["models"].items():
        for dk, dv in (mv.get("dimensions") or {}).items():
            v = learner_models["user"]["models"][mk][dk]["value"]
            if isinstance(v, (int, float)):
                labels.append(dv["name"])
                vals.append(v)

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    vals_plot = vals + vals[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles, vals_plot, "o-", linewidth=2, color="#E74C3C")
    ax.fill(angles, vals_plot, alpha=0.2, color="#E74C3C")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=9)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_title("사용자 학습자 모델 (Radar)", size=13, pad=20, fontweight="bold")
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
        for dk, dv in (mv.get("dimensions") or {}).items():
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

def _counter_to_level(count, thresholds):
    """CPS 하위구인 등 counter 값 → 파생 레벨(1~5)."""
    if not isinstance(count, (int, float)):
        return None
    try:
        c = int(count)
    except (TypeError, ValueError):
        return None
    # thresholds: {"1": "0", "2": "1-2", "3": "3-4", ...}
    for lv in ("5", "4", "3", "2", "1"):
        spec = thresholds.get(lv)
        if spec is None:
            continue
        spec = str(spec).strip()
        if spec.endswith("+"):
            if c >= int(spec[:-1]):
                return int(lv)
        elif "-" in spec:
            lo, hi = spec.split("-")
            if int(lo) <= c <= int(hi):
                return int(lo and hi and lv) if lv else None
            if int(lo) <= c <= int(hi):
                return int(lv)
        else:
            if c == int(spec):
                return int(lv)
    return 1


def _stage_grade_to_score(grade):
    """task_achievement의 A~E 등급 → 1~5 점수."""
    mapping = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}
    if isinstance(grade, str) and grade.upper() in mapping:
        return mapping[grade.upper()]
    return None


def _score_bar(value, scale=5):
    """정수 점수를 █ 막대로."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        return "—"
    v = max(0, min(scale, v))
    return "█" * v + "░" * (scale - v)


def user_model_markdown(config, learner_models):
    """사용자 학습자 모델을 Markdown 문자열로 반환.

    표시 구조:
        # 사용자 학습자 모델
        ## 인지적 요소
          ▸ 과제 수행도달 (stage별 A~E → 점수 변환)
          ▸ 수학적 의사소통 (하위요소 2개 점수)
          ▸ 수학적 추론 (하위요소 3개 점수)
        ## 정의적 요소
          ▸ 협력적 문제해결 (CPS 하위구인 4개: 카운터 + 파생 레벨)
          ▸ 수학 자기효능감 (문항별 pre/post 또는 평균)
        ## 관찰된 오개념 (리스트)
    """
    schema = config["learner_model_schema"]
    m = learner_models["user"]
    models = m["models"]

    lines = [f"### 📊 {m['student_name']}의 학습자 모델"]

    categories = schema.get("categories", {})
    cat_of = {}
    for cat_key, cat_info in categories.items():
        for mk in cat_info.get("models", []):
            cat_of[mk] = cat_key

    # 카테고리 순서: cognitive → affective → (기타)
    ordered_cats = ["cognitive", "affective"] + [
        c for c in categories.keys() if c not in ("cognitive", "affective")
    ]

    for cat_key in ordered_cats:
        if cat_key not in categories:
            continue
        cat_name = categories[cat_key]["name"]
        cat_models = [mk for mk in schema["models"].keys() if cat_of.get(mk) == cat_key]
        if not cat_models:
            continue
        lines.append(f"\n#### 🧩 {cat_name}")

        for mk in cat_models:
            mv = schema["models"][mk]
            if mk not in models:
                continue
            lines.append(f"\n**▸ {mv['name']}** · _{mv.get('name_en','')}_")

            for dk, dv in (mv.get("dimensions") or {}).items():
                if dk not in models[mk]:
                    continue
                entry = models[mk][dk]
                val = entry.get("value")
                dim_type = dv.get("type", "")

                # 1) Stage 등급 (A~E)
                if dim_type == "stage_categorical":
                    per_stage = entry.get("per_stage_values") or {}
                    if per_stage:
                        parts = []
                        for s, g in sorted(per_stage.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else 99):
                            sc = _stage_grade_to_score(g)
                            bar = _score_bar(sc) if sc else "—"
                            parts.append(f"Stage {s}: `[{bar}]` **{g}** ({sc}/5)")
                        lines.append(f"- {dv['name']}")
                        for p in parts:
                            lines.append(f"    - {p}")
                    else:
                        sc = _stage_grade_to_score(val)
                        if sc:
                            lines.append(f"- `[{_score_bar(sc)}]` {dv['name']}: **{val}** ({sc}/5)")
                        else:
                            lines.append(f"- {dv['name']}: 아직 기록 없음")

                # 2) 오개념 리스트
                elif dim_type == "list":
                    if isinstance(val, list) and val:
                        lines.append(f"- {dv['name']}: {', '.join(map(str, val))}")
                    else:
                        lines.append(f"- {dv['name']}: _(없음)_")

                # 3) CPS 카운터 → 파생 레벨
                elif dim_type == "counter":
                    thresholds = dv.get("derived_level_thresholds") or {}
                    level = _counter_to_level(val, thresholds) if thresholds else None
                    count = val if isinstance(val, (int, float)) else 0
                    if level is not None:
                        lines.append(
                            f"- `[{_score_bar(level)}]` {dv['name']}: "
                            f"**Lv {level}/5** · 누적 {count}회"
                        )
                    else:
                        lines.append(f"- {dv['name']}: 누적 {count}회")

                # 4) 순서척도 1~5
                elif dim_type == "ordinal" and isinstance(val, (int, float)):
                    rubric = dv.get("rubric") or {}
                    rubric_hint = rubric.get(str(int(val)), "")
                    hint_suffix = f" — _{rubric_hint}_" if rubric_hint else ""
                    lines.append(
                        f"- `[{_score_bar(val)}]` {dv['name']}: **{int(val)}/5**{hint_suffix}"
                    )

                # 5) 기타 (숫자면 바, 리스트면 나열)
                elif isinstance(val, (int, float)):
                    lines.append(f"- `[{_score_bar(val)}]` {dv['name']}: **{val}/5**")
                elif isinstance(val, list) and val:
                    lines.append(f"- {dv['name']}: {', '.join(map(str, val))}")

            # self_efficacy의 Likert 문항별 pre/post 요약
            if mk == "self_efficacy":
                items = mv.get("items", [])
                pre_vals, post_vals = [], []
                for item in items:
                    iid = item["id"]
                    rec = models[mk].get(iid) or {}
                    pre = rec.get("pre")
                    post = rec.get("post")
                    if isinstance(pre, (int, float)):
                        pre_vals.append(pre)
                    if isinstance(post, (int, float)):
                        post_vals.append(post)
                if pre_vals or post_vals:
                    pre_mean = sum(pre_vals) / len(pre_vals) if pre_vals else None
                    post_mean = sum(post_vals) / len(post_vals) if post_vals else None
                    pre_s = f"{pre_mean:.2f}/4" if pre_mean is not None else "—"
                    post_s = f"{post_mean:.2f}/4" if post_mean is not None else "—"
                    delta_s = ""
                    if pre_mean is not None and post_mean is not None:
                        d = post_mean - pre_mean
                        sign = "▲" if d > 0 else ("▼" if d < 0 else "■")
                        delta_s = f" · {sign} {d:+.2f}"
                    lines.append(f"- 📋 자기효능감 평균 — pre {pre_s} → post {post_s}{delta_s}")
                else:
                    lines.append(f"- 📋 자기효능감 평균 — _(아직 응답 없음)_")

    lines.append(
        f"\n> _1~5 점수는 루브릭 해석 수준을 의미합니다. "
        f"A~E 등급은 A=5 … E=1 로 환산. CPS는 누적 카운트 기반 파생 레벨._"
    )
    return "\n".join(lines)


# --------------------------------------------------------------------
# 오개념 타임라인 (간트 차트)
# --------------------------------------------------------------------

def _collect_misconception_history(learner_models):
    """user의 misconceptions history를 스키마 이동(v1/v2) 양쪽 모두에서 수집."""
    user_models = learner_models["user"]["models"]
    candidates = []
    for parent in ("task_achievement", "cognitive_state"):
        branch = user_models.get(parent) or {}
        mis = branch.get("misconceptions")
        if mis and mis.get("history"):
            # '초기값' 같은 add/remove 아닌 엔트리는 걸러낸다
            events = [e for e in mis["history"] if e.get("event") in ("added", "removed")]
            if events:
                candidates.extend(events)
    # turn 순으로 정렬
    candidates.sort(key=lambda e: (e.get("turn", 0), 0 if e.get("event") == "added" else 1))
    return candidates


def misconception_timeline_figure(config, learner_models):
    """오개념 등장/해소 타임라인 (간트 차트).

    x축: 턴(turn) — 업데이트 회차
    y축: 개별 오개념 항목
    각 오개념은 등장(added)한 턴부터 해소(removed)된 턴까지 가로 막대로 표시.
    미해소 오개념은 현재 턴까지 연장되며 다른 색으로 표시된다.
    """
    events = _collect_misconception_history(learner_models)
    fig, ax = plt.subplots(figsize=(11, max(3.5, 0.5 * max(3, len(set(e["item"] for e in events))))))

    if not events:
        ax.axis("off")
        ax.text(0.5, 0.5,
                "아직 기록된 오개념이 없습니다.\n\n"
                "사용자가 발화하면 교수자 모델이 분석하여\n"
                "오개념 등장/해소 이벤트를 여기에 타임라인으로 표시합니다.",
                ha="center", va="center", fontsize=11, color="#666")
        fig.tight_layout()
        return fig

    # 각 오개념 항목별로 (start_turn, duration) 구간들 계산
    item_order = []
    current_open = {}   # item -> start_turn
    spans = {}          # item -> list of (start, end, resolved_bool)

    last_turn = max(e.get("turn", 0) for e in events)

    for e in events:
        item = e["item"]
        if item not in spans:
            spans[item] = []
            item_order.append(item)
        t = e.get("turn", 0)
        if e["event"] == "added":
            # 이미 열린 구간이 있으면 무시(중복 add)
            if item not in current_open:
                current_open[item] = t
        elif e["event"] == "removed":
            if item in current_open:
                spans[item].append((current_open[item], t, True))
                del current_open[item]

    # 아직 안 닫힌 오개념 → 현재 턴까지 연장 (미해소)
    for item, start_t in current_open.items():
        spans[item].append((start_t, last_turn + 0.5, False))

    # 타임라인 그리기
    y_positions = {item: i for i, item in enumerate(reversed(item_order))}
    ax.set_ylim(-0.6, len(item_order) - 0.4)
    ax.set_yticks(range(len(item_order)))
    ax.set_yticklabels(list(reversed(item_order)), fontsize=10)

    for item, ranges in spans.items():
        y = y_positions[item]
        for start, end, resolved in ranges:
            width = max(0.4, end - start)
            color = "#7BC67E" if resolved else "#E88B8B"  # 해소=초록, 미해소=붉은분홍
            ax.broken_barh([(start, width)], (y - 0.3, 0.6),
                           facecolors=color, edgecolor="#444", linewidth=0.8, alpha=0.85)
            # 시작점 마커
            ax.plot([start], [y], marker="o", color="#CC3333", markersize=6, zorder=3)
            if resolved:
                ax.plot([end], [y], marker="s", color="#2E7D32", markersize=6, zorder=3)

    ax.set_xlim(0, last_turn + 1.5)
    ax.set_xlabel("턴(turn) — 업데이트 회차", fontsize=11)
    ax.set_title("🧭 오개념 타임라인 — 언제 등장했고 언제 해소됐는가",
                 fontsize=13, fontweight="bold", pad=14)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.invert_yaxis()  # 첫 등장 순서를 위에서 아래로

    # 범례
    legend_handles = [
        mpatches.Patch(facecolor="#7BC67E", edgecolor="#444", label="해소된 오개념"),
        mpatches.Patch(facecolor="#E88B8B", edgecolor="#444", label="미해소 (현재 지속)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    return fig


def plot_misconception_timeline(config, learner_models):
    fig = misconception_timeline_figure(config, learner_models)
    plt.show()
    return fig


# --------------------------------------------------------------------
# CPS 히트맵 (Stage × CPS 하위구인)
# --------------------------------------------------------------------

def _compute_cps_stage_matrix(config, learner_models):
    """CPS 하위구인별로 Stage당 증가량 매트릭스를 계산.

    반환:
        stages: [1, 2, 3, ...]
        dims:   [("shared_understanding","공동 이해 형성·유지"), ...]
        matrix: np.ndarray shape (len(dims), len(stages))
                각 셀 = 해당 stage에서의 증가량 (counter delta)
        totals: 각 dimension의 현재 누적 (value)
    """
    schema = config["learner_model_schema"]
    task = config.get("tasks") or {}
    stages_list = list(task.get("stages", {}).keys()) or ["1"]
    stages = [int(s) for s in stages_list]
    stages.sort()

    user_models = learner_models["user"]["models"]
    cps_model = schema["models"].get("cps") or {}
    cps_branch = user_models.get("cps") or {}

    dim_keys = list((cps_model.get("dimensions") or {}).keys())
    dim_labels = [
        cps_model["dimensions"][dk]["name"] for dk in dim_keys
    ]

    matrix = np.zeros((len(dim_keys), len(stages)), dtype=float)
    totals = []

    for i, dk in enumerate(dim_keys):
        entry = cps_branch.get(dk) or {}
        totals.append(entry.get("value", 0))
        history = entry.get("history") or []
        # history 각 entry는 {stage, value(=cumulative), evidence} 로 가정
        # 연속 엔트리 간 차이를 증가량으로 환산
        prev_val = 0
        for h in history:
            cur_val = h.get("value", 0)
            if not isinstance(cur_val, (int, float)):
                continue
            delta = max(0, cur_val - prev_val)
            stage = h.get("stage", stages[0])
            if stage in stages:
                col = stages.index(stage)
                matrix[i, col] += delta
            prev_val = cur_val

    return stages, list(zip(dim_keys, dim_labels)), matrix, totals


def cps_heatmap_figure(config, learner_models):
    """Stage × CPS 하위구인 히트맵.

    - 빈 데이터 상태에서도 정상 렌더(안내 메시지 포함)
    - 셀에는 해당 stage 증가량을 숫자로 표기
    - 우측에 각 하위구인의 현재 누적 총합 표시
    """
    stages, dim_pairs, matrix, totals = _compute_cps_stage_matrix(config, learner_models)
    dim_labels = [d[1] for d in dim_pairs]

    fig, ax = plt.subplots(figsize=(10, 4.5))

    if matrix.size == 0 or matrix.sum() == 0:
        # CPS 태깅 파이프라인이 아직 데이터를 생성하지 않은 상태
        im = ax.imshow(matrix if matrix.size else np.zeros((4, max(1, len(stages)))),
                       aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels([f"Stage {s}" for s in stages], fontsize=10)
        ax.set_yticks(range(len(dim_labels) or 4))
        ax.set_yticklabels(dim_labels or ["—"] * 4, fontsize=10)
        ax.set_title("🤝 CPS 히트맵 — Stage별 협력 행동 분포",
                     fontsize=13, fontweight="bold", pad=14)
        ax.text(0.5, -0.35,
                "아직 CPS 태깅 데이터가 없습니다. "
                "prompts/08_cps_tagging.md 와 세션 파이프라인을 연결하면 "
                "각 stage에서 관찰된 협력 행동이 여기에 표시됩니다.",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=9, color="#666", wrap=True)
        fig.tight_layout()
        return fig

    vmax = max(1.0, float(matrix.max()))
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0, vmax=vmax)

    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels([f"Stage {s}" for s in stages], fontsize=10)
    ax.set_yticks(range(len(dim_labels)))
    ax.set_yticklabels(dim_labels, fontsize=10)

    # 셀 값 주석
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            color = "white" if v > vmax * 0.55 else "#222"
            ax.text(j, i, f"{int(v)}" if v == int(v) else f"{v:.1f}",
                    ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    # 우측 누적 합계 (twinx)
    ax2 = ax.twinx()
    ax2.set_yticks(range(len(dim_labels)))
    ax2.set_yticklabels([f"총 {int(t)}회" for t in totals], fontsize=9, color="#333")
    ax2.set_ylim(ax.get_ylim())
    ax2.tick_params(axis="y", pad=4)

    ax.set_title("🤝 CPS 히트맵 — Stage별 협력 행동 증가량",
                 fontsize=13, fontweight="bold", pad=14)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.12)
    cbar.set_label("해당 Stage의 관찰 횟수", fontsize=9)

    fig.tight_layout()
    return fig


def plot_cps_heatmap(config, learner_models):
    fig = cps_heatmap_figure(config, learner_models)
    plt.show()
   