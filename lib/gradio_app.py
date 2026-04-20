"""Gradio 채팅 인터페이스.

왼쪽 패널: Chatbot(발화자별 컬러 버블) + 입력 / 오른쪽 탭: 레이더 · 변화 추이 · 학습자모델 · 교수자 Decision

* gradio 는 launch_ui() 실행 시점에 지연 임포트한다. 덕분에 gradio 미설치
  환경에서도 team4 패키지 전체는 정상 임포트된다 (CLI 런너나 config 로더만
  쓰는 경우).
"""

import json

from .session import CollaborativeSession
from .visualize import (
    radar_figure,
    history_figure,
    user_model_markdown,
    misconception_timeline_figure,
    cps_heatmap_figure,
)


def launch_ui(*, config, prompts, learner_models, api, share=True):
    """협력학습 세션을 Gradio Blocks UI로 기동.

    Args:
        config, prompts, learner_models, api: bootstrap() 결과를 그대로 주입.
        share: Colab 이면 True (공용 링크 72시간), 로컬 단독 실행이면 False 가능.
    """
    import gradio as gr  # 지연 임포트

    session = CollaborativeSession(config, prompts, learner_models, api)
    n1 = config["personas"]["ai_students"]["ai_1"]["name"]
    n2 = config["personas"]["ai_students"]["ai_2"]["name"]
    n3 = config["personas"]["ai_students"]["ai_3"]["name"]
    AI_NAME_BY_ID = {"ai_1": n1, "ai_2": n2, "ai_3": n3}

    # -------- 발화자별 버블 스타일 --------
    # 각 AI와 시스템 메시지에 고유 색상 테두리/배경을 준다.
    # 🧑‍🎓 아이콘은 사용하지 않고, 좌측 컬러바가 발화자를 구분한다.
    BUBBLE_STYLES = {
        "ai_1":   {"border": "#2563eb", "bg": "#eff6ff"},  # 민준 — 파랑
        "ai_2":   {"border": "#db2777", "bg": "#fdf2f8"},  # 서연 — 분홍
        "ai_3":   {"border": "#16a34a", "bg": "#f0fdf4"},  # 연우 — 초록
        "system": {"border": "#6b7280", "bg": "#f9fafb"},  # 시스템 — 회색
    }

    def _bubble(speaker_id: str, name: str, text: str) -> str:
        """발화자별 색상 버블 HTML (markdown 호환).

        블록 내부 공백 라인을 넣어 안쪽 텍스트가 markdown으로 계속 렌더되게 한다.
        """
        s = BUBBLE_STYLES.get(speaker_id, BUBBLE_STYLES["system"])
        return (
            f'<div style="border-left:4px solid {s["border"]};'
            f' background:{s["bg"]}; padding:8px 12px; border-radius:6px;">\n\n'
            f'<span style="color:{s["border"]}; font-weight:700;">{name}</span>\n\n'
            f'{text}\n\n'
            f'</div>'
        )

    def _system_bubble(text: str) -> str:
        return (
            f'<div style="border-left:4px solid {BUBBLE_STYLES["system"]["border"]};'
            f' background:{BUBBLE_STYLES["system"]["bg"]};'
            f' padding:8px 12px; border-radius:6px; font-size:0.95em;">\n\n'
            f'{text}\n\n'
            f'</div>'
        )

    # -------- 시각화 래퍼 (현재 session 이 참조하는 learner_models 사용) --------
    def _radar():
        return radar_figure(config, learner_models)

    def _history():
        return history_figure(config, learner_models)

    def _model_md():
        return user_model_markdown(config, learner_models)

    def _decision_json():
        d = getattr(session, "last_tutor_decision", None) or {}
        return json.dumps(d, ensure_ascii=False, indent=2)

    def _misconception_timeline():
        return misconception_timeline_figure(config, learner_models)

    def _cps_heatmap():
        return cps_heatmap_figure(config, learner_models)

    # -------- 대화 초기 상태: stage 안내 + ai_1 인트로 --------
    def _initial_history():
        s = session.current_stage_info()
        intro = session.stage_intro_utterance("ai_1")
        return [
            {"role": "assistant",
             "content": _system_bubble(f"**Stage {session.current_stage}: {s['title']}**\n\n{s['prompt']}")},
            {"role": "assistant",
             "content": _bubble("ai_1", n1, intro)},
        ]

    # -------- 이벤트 콜백 --------
    def _refresh_bundle(history, clear_msg=True):
        """chatbot/msg + 6개 visualization output을 한 번에 묶어 반환.

        on_submit 처럼 매 턴 전체 대시보드를 자동 갱신하기 위해 사용한다.
        outputs 순서: [chatbot, msg, radar, hist_plot, model_md, dec_code, misc_plot, cps_plot]
        """
        return (
            history,
            "" if clear_msg else gr.update(),
            _radar(), _history(), _model_md(),
            _decision_json(), _misconception_timeline(), _cps_heatmap(),
        )

    def on_submit(msg, history):
        """제너레이터 기반 스트리밍 on_submit.

        흐름:
          1) 사용자 버블 즉시 표시 (yield)
          2) '생각하는 중…' indicator 표시 (yield)
          3) analyze + tutor_decision 실행 (두 번의 순차 LLM 호출)
          4) indicator 제거 + speaking_agents를 병렬로 호출
          5) 먼저 끝난 AI부터 순서대로 버블 추가 (yield마다 UI 갱신)
          6) stage_complete면 전환 처리까지 포함

        AI 발화 순서는 API 완료 순(= 말 짧게 하는 사람이 먼저)으로 자연스러운
        인간 대화 리듬을 만든다.
        """
        msg = (msg or "").strip()
        if not msg:
            yield _refresh_bundle(history)
            return

        # 1) 사용자 버블 즉시 반영 + 입력창 클리어
        history = history + [{"role": "user", "content": msg}]
        yield _refresh_bundle(history, clear_msg=True)

        # 2) '생각하는 중' indicator
        thinking_idx = len(history)
        history = history + [{"role": "assistant",
                              "content": _system_bubble("_학생들이 생각하는 중…_")}]
        yield _refresh_bundle(history, clear_msg=False)

        # 3) analyze + tutor_decision (이 두 호출이 병목의 대부분)
        try:
            prep = session.user_turn_prep(msg)
        except Exception as e:
            history[thinking_idx] = {"role": "assistant", "content": f"⚠️ 오류: {e}"}
            yield _refresh_bundle(history, clear_msg=False)
            return

        decision = prep["decision"]

        # 4) indicator 제거, 사용자 모드가 teacher면 시스템 버블 하나 추가
        history = history[:thinking_idx]
        if prep["user_mode"] == "teacher":
            history.append({"role": "assistant",
                            "content": _system_bubble("_사용자가 설명자(교수자) 모드로 감지되었습니다. AI 학생들은 학습자 모드로 짧게 반응합니다._")})
        yield _refresh_bundle(history, clear_msg=False)

        # 5) 병렬 발화: 먼저 끝난 사람부터 버블이 나타난다 (자연스러운 대화 리듬)
        for aid, text in session.stream_ai_turns(
            msg, decision,
            silence_trigger=decision.get("silence_trigger", False),
        ):
            history.append({"role": "assistant",
                            "content": _bubble(aid, AI_NAME_BY_ID[aid], text)})
            yield _refresh_bundle(history, clear_msg=False)

        # 6) stage 전환
        if decision.get("stage_complete"):
            history.append({"role": "assistant",
                            "content": _system_bubble(f"Stage {session.current_stage} 완료")})
            if session.advance_stage():
                s = session.current_stage_info()
                history.append({"role": "assistant",
                                "content": _system_bubble(f"**Stage {session.current_stage}: {s['title']}**\n\n{s['prompt']}")})
                yield _refresh_bundle(history, clear_msg=False)
                intro = session.stage_intro_utterance("ai_2")
                history.append({"role": "assistant",
                                "content": _bubble("ai_2", n2, intro)})
            else:
                history.append({"role": "assistant",
                                "content": _system_bubble("모든 Stage 완료! 오른쪽 탭이 자동으로 최신 상태입니다.")})
            yield _refresh_bundle(history, clear_msg=False)

    def on_silence_tick(history):
        """1초마다 호출되는 침묵 감지 폴러.

        사용자가 60초 이상 말하지 않으면 한 명의 AI가 대화를 유도한다.
        AI 응답이 추가되면 새 history를, 아니면 기존 history를 그대로 반환.
        """
        try:
            result = session.nudge_on_silence()
        except Exception as e:
            history = history + [{"role": "assistant",
                                   "content": _system_bubble(f"침묵 유도 중 오류: {e}")}]
            return history
        if not result:
            return history
        aid = result["agent_id"]
        text = result["text"]
        # 침묵 감지 안내 문구 없이 일반 발화처럼 버블만 추가한다.
        history = history + [{
            "role": "assistant",
            "content": _bubble(aid, AI_NAME_BY_ID[aid], text),
        }]
        return history

    def on_next_stage(history):
        if session.advance_stage():
            s = session.current_stage_info()
            intro = session.stage_intro_utterance("ai_2")
            history = history + [
                {"role": "assistant",
                 "content": _system_bubble(f"**Stage {session.current_stage}: {s['title']}**\n\n{s['prompt']}")},
                {"role": "assistant",
                 "content": _bubble("ai_2", n2, intro)},
            ]
        else:
            history = history + [{"role": "assistant",
                                  "content": _system_bubble("모든 stage가 이미 완료되었습니다.")}]
        return history

    # -------- Blocks 레이아웃 --------
    with gr.Blocks(title="협력학습 세션", theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"# 🎓 {session.task['task_title']}")
        with gr.Row():
            # ---- 왼쪽: 채팅 ----
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    value=_initial_history(),
                    type="messages",
                    height=560,
                    label="세션 대화",
                    show_copy_button=True,
                )
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="질문이나 답변을 입력하고 Enter (Shift+Enter로 줄바꿈)",
                        scale=6, show_label=False, lines=1, autofocus=True,
                    )
                    send = gr.Button("보내기", variant="primary", scale=1)
                with gr.Row():
                    next_btn = gr.Button("▶ 다음 Stage")
                    refresh_all_btn = gr.Button("🔄 대시보드 전체 갱신")

            # ---- 오른쪽: 대시보드 ----
            # 탭 순서: ① 레이더 → ② 학습자 모델 → ③ 변화 추이
            #         → ④ 교수자 디시젼 → ⑤ 오개념 타임라인 → ⑥ CPS 히트맵
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("① 🕸️ 레이더"):
                        gr.Markdown(
                            "_학습자 모델의 주요 요소를 방사형 그래프로 한눈에 조감합니다._"
                        )
                        radar = gr.Plot(label="사용자 레이더")
                        gr.Button("🔄 갱신").click(_radar, outputs=radar)
                    with gr.Tab("② 🧠 학습자 모델"):
                        gr.Markdown(
                            "_인지/정의 카테고리별 하위요소 점수와 루브릭 해석입니다._"
                        )
                        model_md = gr.Markdown()
                        gr.Button("🔄 갱신").click(_model_md, outputs=model_md)
                    with gr.Tab("③ 📈 변화 추이"):
                        gr.Markdown(
                            "_업데이트 회차에 따른 각 하위요소의 변화 추이입니다._"
                        )
                        hist_plot = gr.Plot(label="사용자 단계별 변화")
                        gr.Button("🔄 갱신").click(_history, outputs=hist_plot)
                    with gr.Tab("④ 🧭 교수자 디시젼"):
                        gr.Markdown(
                            "_가장 최근 턴의 교수자 모델이 내린 의사결정(JSON)입니다._"
                        )
                        dec_code = gr.Code(language="json", label="last_tutor_decision")
                        gr.Button("🔄 갱신").click(_decision_json, outputs=dec_code)
                    with gr.Tab("⑤ ⏱️ 오개념 타임라인"):
                        gr.Markdown(
                            "_관찰된 오개념이 언제 등장해서 언제 해소됐는지 간트 차트로 표시합니다. "
                            "🟢 = 해소, 🔴 = 현재 지속._"
                        )
                        misc_plot = gr.Plot(label="오개념 타임라인")
                        gr.Button("🔄 갱신").click(_misconception_timeline, outputs=misc_plot)
                    with gr.Tab("⑥ 🤝 CPS 히트맵"):
                        gr.Markdown(
                            "_Stage × CPS 하위구인(공동이해·행동·조직화·수정) 매트릭스. "
                            "각 stage에서 어떤 협력 행동이 많이 관찰됐는지 색 진하기로 표시._"
                        )
                        cps_plot = gr.Plot(label="CPS 히트맵")
                        gr.Button("🔄 갱신").click(_cps_heatmap, outputs=cps_plot)

        # 이벤트 와이어링
        # on_submit은 턴마다 [chatbot, msg]와 6개 시각화를 한 번에 갱신한다.
        auto_outputs = [chatbot, msg, radar, hist_plot, model_md, dec_code, misc_plot, cps_plot]
        send.click(on_submit, [msg, chatbot], auto_outputs)
        msg.submit(on_submit, [msg, chatbot], auto_outputs)
        next_btn.click(on_next_stage, chatbot, chatbot)
        refresh_all_btn.click(
            lambda: (
                _radar(), _history(), _model_md(), _decision_json(),
                _misconception_timeline(), _cps_heatmap(),
            ),
            outputs=[radar, hist_plot, model_md, dec_code, misc_plot, cps_plot],
        )

        # 침묵 유도 타이머: 15초 간격으로 침묵 여부 폴링 (실제 트리거는 60초 누적부터)
        silence_timer = gr.Timer(value=15.0, active=True)
        silence_timer.tick(on_silence_tick, inputs=chatbot, outputs=chatbot)

        # 최초 로드 시 대시보드 초기화
        demo.load(_radar, outputs=radar)
        demo.load(_history, outputs=hist_plot)
        demo.load(_model_md, outputs=model_md)
        demo.load(_misconception_timeline, outputs=misc_plot)
        demo.load(_cps_heatmap, outputs=cps_plot)

    demo.launch(share=share, debug=False, quiet=True)
    return demo
