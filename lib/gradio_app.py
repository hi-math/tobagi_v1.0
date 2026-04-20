"""Gradio 채팅 인터페이스 (V4 + 토큰 스트리밍 + 새로고침 싱크).

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
    """협력학습 세션을 Gradio Blocks UI로 기동."""
    import gradio as gr

    session = CollaborativeSession(config, prompts, learner_models, api)
    n1 = config["personas"]["ai_students"]["ai_1"]["name"]
    n2 = config["personas"]["ai_students"]["ai_2"]["name"]
    n3 = config["personas"]["ai_students"]["ai_3"]["name"]
    AI_NAME_BY_ID = {"ai_1": n1, "ai_2": n2, "ai_3": n3}

    BUBBLE_STYLES = {
        "ai_1":   {"border": "#2563eb", "bg": "#eff6ff"},
        "ai_2":   {"border": "#db2777", "bg": "#fdf2f8"},
        "ai_3":   {"border": "#16a34a", "bg": "#f0fdf4"},
        "system": {"border": "#6b7280", "bg": "#f9fafb"},
    }

    def _bubble(speaker_id, name, text, show_name=True):
        s = BUBBLE_STYLES.get(speaker_id, BUBBLE_STYLES["system"])
        name_html = (
            f'<span style="color:{s["border"]}; font-weight:700;">{name}</span>\n\n'
            if show_name else ''
        )
        return (
            f'<div style="border-left:4px solid {s["border"]};'
            f' background:{s["bg"]}; padding:8px 12px; border-radius:6px;">\n\n'
            f'{name_html}'
            f'{text}\n\n'
            f'</div>'
        )

    def _split_paragraphs(text):
        """AI 발화를 빈 줄(\\n\\n) 기준으로 단락 리스트로 분할."""
        parts = [p.strip() for p in (text or "").split("\n\n") if p.strip()]
        return parts if parts else [(text or "").strip()]

    def _bubble_messages(aid, name, text):
        """한 AI 발화를 단락별 Chatbot 메시지 리스트로 변환.

        첫 번째 버블만 이름 헤더를 표시하고, 이어지는 단락들은
        동일 색상의 버블에 이름 없이 본문만 담아 시각적으로 '연속 발화'처럼 보이게 한다.
        """
        parts = _split_paragraphs(text)
        out = []
        for i, p in enumerate(parts):
            out.append({
                "role": "assistant",
                "content": _bubble(aid, name, p, show_name=(i == 0)),
            })
        return out

    def _system_bubble(text):
        return (
            f'<div style="border-left:4px solid {BUBBLE_STYLES["system"]["border"]};'
            f' background:{BUBBLE_STYLES["system"]["bg"]};'
            f' padding:8px 12px; border-radius:6px; font-size:0.95em;">\n\n'
            f'{text}\n\n'
            f'</div>'
        )

    def _stage_card(stage_num, stage_info):
        title = stage_info.get("title", "")
        core_q = stage_info.get("core_question", "")
        prompt = stage_info.get("prompt", "")
        accent = "#4f46e5"
        accent_dark = "#4338ca"
        body_bg = "#eef2ff"
        prompt_html = prompt.replace("\n", "<br>")
        core_q_html = core_q.replace("\n", "<br>")
        return (
            f'<div style="border:2px solid {accent}; border-radius:10px; '
            f'overflow:hidden; margin:6px 0; box-shadow:0 2px 6px rgba(79,70,229,0.15);">'
            f'<div style="background:{accent}; color:white; padding:10px 16px; '
            f'font-weight:700; letter-spacing:0.5px; font-size:1.05em;">'
            f'📋 STAGE {stage_num}'
            f'</div>'
            f'<div style="background:{body_bg}; padding:14px 16px; color:#1e1b4b;">'
            f'<div style="font-size:1.15em; font-weight:700; color:{accent_dark}; '
            f'margin-bottom:10px;">{title}</div>'
            + (
                f'<div style="font-size:0.85em; font-weight:600; color:{accent_dark}; '
                f'text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px;">'
                f'핵심 질문</div>'
                f'<div style="margin-bottom:12px;">{core_q_html}</div>'
                if core_q else ''
            )
            + (
                f'<div style="font-size:0.85em; font-weight:600; color:{accent_dark}; '
                f'text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px;">'
                f'과제</div>'
                f'<div>{prompt_html}</div>'
                if prompt else ''
            )
            + '</div></div>'
        )

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

    def _initial_history():
        s = session.current_stage_info()
        intro = session.stage_intro_utterance("ai_1")
        return [
            {"role": "assistant",
             "content": _stage_card(session.current_stage, s)},
        ] + _bubble_messages("ai_1", n1, intro)

    def _rebuild_chat_from_session():
        """페이지 새로고침 시 session.conversation 기반으로 채팅을 재구성."""
        if not session.conversation:
            return _initial_history()

        history = []
        current_stage_seen = None
        for m in session.conversation:
            stage_num = m.get("stage", session.current_stage)
            if stage_num != current_stage_seen:
                try:
                    s = session.task["stages"][str(stage_num)]
                    history.append({"role": "assistant",
                                    "content": _stage_card(stage_num, s)})
                except KeyError:
                    pass
                current_stage_seen = stage_num

            speaker = m.get("speaker", "")
            content = m.get("content", "")
            if speaker == "사용자":
                history.append({"role": "user", "content": content})
                continue

            aid = m.get("agent_id")
            if not aid:
                for k, nm in AI_NAME_BY_ID.items():
                    if nm == speaker:
                        aid = k
                        break
            aid = aid or "ai_1"
            history.extend(_bubble_messages(aid, AI_NAME_BY_ID.get(aid, speaker), content))
        return history

    def _refresh_bundle(history, clear_msg=True):
        return (
            history,
            "" if clear_msg else gr.update(),
            _radar(), _history(), _model_md(),
            _decision_json(), _misconception_timeline(), _cps_heatmap(),
        )

    def _chat_only(history, clear_msg=False):
        return (
            history,
            "" if clear_msg else gr.update(),
            gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(),
        )

    def on_submit(msg, history):
        msg = (msg or "").strip()
        if not msg:
            yield _refresh_bundle(history)
            return

        history = history + [{"role": "user", "content": msg}]
        yield _chat_only(history, clear_msg=True)

        thinking_idx = len(history)
        history = history + [{"role": "assistant",
                              "content": _system_bubble("_학생들이 생각하는 중…_")}]
        yield _chat_only(history, clear_msg=False)

        try:
            prep = session.user_turn_prep(msg)
        except Exception as e:
            history[thinking_idx] = {"role": "assistant", "content": f"⚠️ 오류: {e}"}
            yield _refresh_bundle(history, clear_msg=False)
            return

        decision = prep["decision"]

        history = history[:thinking_idx]
        if prep["user_mode"] == "teacher":
            history.append({"role": "assistant",
                            "content": _system_bubble("_사용자가 설명자(교수자) 모드로 감지되었습니다. AI 학생들은 학습자 모드로 짧게 반응합니다._")})
        yield _chat_only(history, clear_msg=False)

        slot = {}
        buf = {}
        done_payloads = {}
        CURSOR = "▍"

        for ev, aid, payload in session.stream_ai_turns_tokens(
            msg, decision,
            silence_trigger=decision.get("silence_trigger", False),
        ):
            if ev == "start":
                buf[aid] = ""
                slot[aid] = len(history)
                history.append({"role": "assistant",
                                "content": _bubble(aid, AI_NAME_BY_ID[aid], CURSOR)})
                yield _chat_only(history, clear_msg=False)
            elif ev == "update":
                buf[aid] = buf.get(aid, "") + payload
                history[slot[aid]] = {
                    "role": "assistant",
                    "content": _bubble(aid, AI_NAME_BY_ID[aid], buf[aid] + CURSOR),
                }
                yield _chat_only(history, clear_msg=False)
            elif ev == "done":
                done_payloads[aid] = payload
                # 스트리밍 끝난 시점엔 일단 단일 버블로 고정 (분할은 뒤에서 일괄 처리).
                history[slot[aid]] = {
                    "role": "assistant",
                    "content": _bubble(aid, AI_NAME_BY_ID[aid], payload),
                }
                yield _refresh_bundle(history, clear_msg=False)

        # 모든 AI 스트리밍이 끝난 뒤 단락(\n\n) 단위로 버블 분할.
        # slot 인덱스가 큰 것부터 처리해야 뒤쪽 insert가 앞쪽 slot 위치를 깨뜨리지 않는다.
        for aid in sorted(slot.keys(), key=lambda a: slot[a], reverse=True):
            text = done_payloads.get(aid, "")
            msgs = _bubble_messages(aid, AI_NAME_BY_ID[aid], text)
            if len(msgs) <= 1:
                continue
            history[slot[aid]] = msgs[0]
            for m in reversed(msgs[1:]):
                history.insert(slot[aid] + 1, m)
        if done_payloads:
            yield _chat_only(history, clear_msg=False)

        if decision.get("stage_complete"):
            history.append({"role": "assistant",
                            "content": _system_bubble(f"Stage {session.current_stage} 완료")})
            if session.advance_stage():
                s = session.current_stage_info()
                history.append({"role": "assistant",
                                "content": _stage_card(session.current_stage, s)})
                yield _chat_only(history, clear_msg=False)
                intro = session.stage_intro_utterance("ai_2")
                history.extend(_bubble_messages("ai_2", n2, intro))
            else:
                history.append({"role": "assistant",
                                "content": _system_bubble("모든 Stage 완료. 오른쪽 탭이 자동으로 최신 상태입니다.")})
            yield _refresh_bundle(history, clear_msg=False)

    def on_silence_tick(history):
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
        history = history + _bubble_messages(aid, AI_NAME_BY_ID[aid], text)
        return history

    def on_next_stage(history):
        if session.advance_stage():
            s = session.current_stage_info()
            intro = session.stage_intro_utterance("ai_2")
            history = history + [
                {"role": "assistant",
                 "content": _stage_card(session.current_stage, s)},
            ] + _bubble_messages("ai_2", n2, intro)
        else:
            history = history + [{"role": "assistant",
                                  "content": _system_bubble("모든 stage가 이미 완료되었습니다.")}]
        return history

    with gr.Blocks(title="협력학습 세션", theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"# 🎓 {session.task['task_title']}")
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    value=[],
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

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("① 🕸️ 레이더"):
                        gr.Markdown("_학습자 모델의 주요 요소를 방사형 그래프로 한눈에 조감합니다._")
                        radar = gr.Plot(label="사용자 레이더")
                        gr.Button("🔄 갱신").click(_radar, outputs=radar)
                    with gr.Tab("② 🧠 학습자 모델"):
                        gr.Markdown("_인지/정의 카테고리별 하위요소 점수와 루브릭 해석입니다._")
                        model_md = gr.Markdown()
                        gr.Button("🔄 갱신").click(_model_md, outputs=model_md)
                    with gr.Tab("③ 📈 변화 추이"):
                        gr.Markdown("_업데이트 회차에 따른 각 하위요소의 변화 추이입니다._")
                        hist_plot = gr.Plot(label="사용자 단계별 변화")
                        gr.Button("🔄 갱신").click(_history, outputs=hist_plot)
                    with gr.Tab("④ 🧭 교수자 디시젼"):
                        gr.Markdown("_가장 최근 턴의 교수자 모델이 내린 의사결정(JSON)입니다._")
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

        silence_timer = gr.Timer(value=15.0, active=True)
        silence_timer.tick(on_silence_tick, inputs=chatbot, outputs=chatbot)

        demo.load(_rebuild_chat_from_session, outputs=chatbot)
        demo.load(_radar, outputs=radar)
        demo.load(_history, outputs=hist_plot)
        demo.load(_model_md, outputs=model_md)
        demo.load(_decision_json, outputs=dec_code)
        demo.load(_misconception_timeline, outputs=misc_plot)
        demo.load(_cps_heatmap, outputs=cps_plot)

    demo.launch(share=share, debug=False, quiet=True)
    return demo
