"""Gradio 채팅 인터페이스.

왼쪽 패널: Chatbot + 입력 / 오른쪽 탭: 레이더 · 변화 추이 · 학습자모델 · 교수자 Decision

* gradio 는 launch_ui() 실행 시점에 지연 임포트한다. 덕분에 gradio 미설치
  환경에서도 team4 패키지 전체는 정상 임포트된다 (CLI 런너나 config 로더만
  쓰는 경우).
"""

import json

from .session import CollaborativeSession
from .visualize import radar_figure, history_figure, user_model_markdown


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

    # -------- 대화 초기 상태: stage 안내 + ai_1 인트로 --------
    def _initial_history():
        s = session.current_stage_info()
        intro = session.stage_intro_utterance("ai_1")
        return [
            {"role": "assistant",
             "content": f"📝 **Stage {session.current_stage}: {s['title']}**\n\n{s['prompt']}"},
            {"role": "assistant",
             "content": f"🧑‍🎓 **{n1}**  \n{intro}"},
        ]

    # -------- 이벤트 콜백 --------
    def on_submit(msg, history):
        msg = (msg or "").strip()
        if not msg:
            return history, ""
        history = history + [{"role": "user", "content": msg}]
        try:
            result = session.user_turn(msg)
        except Exception as e:
            history.append({"role": "assistant", "content": f"⚠️ 오류: {e}"})
            return history, ""

        history.append({"role": "assistant",
                        "content": f"🧑‍🎓 **{n1}**  \n{result['ai_1']}"})
        history.append({"role": "assistant",
                        "content": f"🧑‍🎓 **{n2}**  \n{result['ai_2']}"})

        if result["decision"].get("stage_complete"):
            history.append({"role": "assistant",
                            "content": f"✅ Stage {session.current_stage} 완료 신호"})
            if session.advance_stage():
                s = session.current_stage_info()
                history.append({"role": "assistant",
                                "content": f"▶ **Stage {session.current_stage}: {s['title']}**\n\n{s['prompt']}"})
                intro = session.stage_intro_utterance("ai_2")
                history.append({"role": "assistant",
                                "content": f"🧑‍🎓 **{n2}**  \n{intro}"})
            else:
                history.append({"role": "assistant",
                                "content": "🎉 모든 Stage 완료! 오른쪽 탭의 🔄 갱신으로 최종 모델을 확인해 보세요."})
        return history, ""

    def on_next_stage(history):
        if session.advance_stage():
            s = session.current_stage_info()
            intro = session.stage_intro_utterance("ai_2")
            history = history + [
                {"role": "assistant",
                 "content": f"▶ **Stage {session.current_stage}: {s['title']}**\n\n{s['prompt']}"},
                {"role": "assistant",
                 "content": f"🧑‍🎓 **{n2}**  \n{intro}"},
            ]
        else:
            history = history + [{"role": "assistant",
                                  "content": "모든 stage가 이미 완료되었습니다."}]
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
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("🕸️ 레이더"):
                        radar = gr.Plot(label="사용자 레이더")
                        gr.Button("🔄 갱신").click(_radar, outputs=radar)
                    with gr.Tab("📈 변화 추이"):
                        hist_plot = gr.Plot(label="사용자 단계별 변화")
                        gr.Button("🔄 갱신").click(_history, outputs=hist_plot)
                    with gr.Tab("🧠 학습자모델"):
                        model_md = gr.Markdown()
                        gr.Button("🔄 갱신").click(_model_md, outputs=model_md)
                    with gr.Tab("🧭 교수자 Decision"):
                        dec_code = gr.Code(language="json", label="last_tutor_decision")
                        gr.Button("🔄 갱신").click(_decision_json, outputs=dec_code)

        # 이벤트 와이어링
        send.click(on_submit, [msg, chatbot], [chatbot, msg])
        msg.submit(on_submit, [msg, chatbot], [chatbot, msg])
        next_btn.click(on_next_stage, chatbot, chatbot)
        refresh_all_btn.click(
            lambda: (_radar(), _history(), _model_md(), _decision_json()),
            outputs=[radar, hist_plot, model_md, dec_code],
        )

        # 최초 로드 시 대시보드 초기화
        demo.load(_radar, outputs=radar)
        demo.load(_history, outputs=hist_plot)
        demo.load(_model_md, outputs=model_md)

    demo.launch(share=share, debug=False, quiet=True)
    return demo
