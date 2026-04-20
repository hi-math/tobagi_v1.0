"""Gradio 채팅 인터페이스.

왼쪽 패널: Chatbot + 입력 / 오른쪽 탭: 레이더 · 변화 추이 · 학습자모델 · 교수자 Decision

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

        if result.get("user_mode") == "teacher":
            history.append({"role": "assistant",
                            "content": "🎓 _사용자가 설명자(교수자) 모드로 감지되었습니다. AI 학생들은 학습자 모드로 짧게 반응합니다._"})

        # 발화한 AI만 출력 (speaking_agents 기반)
        for aid in ("ai_1", "ai_2", "ai_3"):
            text = result.get(aid)
            if text:
                history.append({"role": "assistant",
                                "content": f"🧑‍🎓 **{AI_NAME_BY_ID[aid]}**  \n{text}"})

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

    def on_silence_tick(history):
        """1초마다 호출되는 침묵 감지 폴러.

        사용자가 60초 이상 말하지 않으면 한 명의 AI가 대화를 유도한다.
        AI 응답이 추가되면 새 history를, 아니면 기존 history를 그대로 반환.
        """
        try:
            result = session.nudge_on_silence()
        except Exception as e:
            history = history + [{"role": "assistant",
                                   "content": f"⚠️ 침묵 유도 중 오류: {e}"}]
            return history
        if not result:
            return history
        aid = result["agent_id"]
        text = result["text"]
        history = history + [{
            "role": "assistant",
            "content": f"🕰️ _{int(session.seconds_since_user_spoke())}초 침묵 감지 — {AI_NAME_BY_ID[aid]}이(가) 먼저 말합니다._\n\n🧑‍🎓 **{AI_NAME_BY_ID[aid]}**  \n{text}",
        }]
        return history

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
        send.click(on_submit, [msg, chatbot], [chatbot, msg])
        msg.submit(on_submit, [msg, chatbot], [chatbot, msg])
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
