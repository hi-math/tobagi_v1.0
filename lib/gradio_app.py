"""Gradio 채팅 인터페이스 (V5, 1 발화 = 1 버블).

왼쪽 패널: Chatbot(발화자별 컬러 버블) + 입력 / 오른쪽 탭: 레이더 · 학습자모델 · 변화 추이 · 교수자 Decision · 오개념 · CPS
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
    import gradio as gr

    session = CollaborativeSession(config, prompts, learner_models, api)

    _streaming_flag = [False]
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

    def _bubble_messages(aid, name, text):
        """한 AI의 1회 발화를 1개 버블로 렌더 (1 발화 = 1 버블 원칙)."""
        text = (text or "").strip()
        return [{"role": "assistant", "content": _bubble(aid, name, text, show_name=True)}]

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
            f'STAGE {stage_num}'
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
        import traceback
        try:
            s = session.current_stage_info()
            intro = session.stage_intro_utterance("ai_1")
            return [
                {"role": "assistant",
                 "content": _stage_card(session.current_stage, s)},
            ] + _bubble_messages("ai_1", n1, intro)
        except Exception as e:
            tb = traceback.format_exc()
            print("[initial_history 오류]\n" + tb)
            err_html = (
                f'<div style="border-left:4px solid #dc2626; background:#fef2f2;'
                f' padding:10px 14px; border-radius:6px; font-family:ui-monospace,monospace;'
                f' font-size:0.85em; white-space:pre-wrap;">'
                f'<b>초기 발화 생성 중 오류</b>\n{type(e).__name__}: {e}\n\n{tb}</div>'
            )
            return [{"role": "assistant", "content": err_html}]

    def _rebuild_chat_from_session():
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

    def echo_user(msg, history):
        msg = (msg or "").strip()
        if not msg:
            return "", history
        return "", history + [{"role": "user", "content": msg}]

    def stream_ai(history):
        if not history or history[-1].get("role") != "user":
            yield _refresh_bundle(history, clear_msg=False)
            return
        msg = history[-1]["content"]

        _streaming_flag[0] = True
        try:
            # 레이턴시 완충: 분석 API가 돌아가는 동안 사용자가 화면을 내내
            # 멍하게 보지 않도록 "분석 중..." placeholder를 즉시 띄운다.
            # prep 완료 후 이 placeholder는 제거되고 실제 AI 발화가 그 자리에서 시작된다.
            thinking_slot = len(history)
            history.append({"role": "assistant",
                            "content": _system_bubble("_AI 학생들이 대화를 곱씹고 있어요..._")})
            yield _chat_only(history, clear_msg=False)

            try:
                prep = session.user_turn_prep(msg)
            except Exception as e:
                history[thinking_slot] = {"role": "assistant", "content": f"오류: {e}"}
                yield _refresh_bundle(history, clear_msg=False)
                return

            # placeholder 제거 (분석 완료 → 실제 발화로 교체 시작)
            history.pop(thinking_slot)

            decision = prep["decision"]

            if prep["user_mode"] == "teacher":
                history = history + [{"role": "assistant",
                                      "content": _system_bubble("_사용자가 설명자(교수자) 모드로 감지되었습니다. AI 학생들은 학습자 모드로 짧게 반응합니다._")}]
                yield _chat_only(history, clear_msg=False)

            slot = {}
            buf = {}
            done_payloads = {}
            CURSOR = "\u258d"

            THROTTLE_CHARS = 6
            last_yield_len = {}

            for ev, aid, payload in session.stream_ai_turns_tokens(
                msg, decision,
                silence_trigger=decision.get("silence_trigger", False),
            ):
                if ev == "start":
                    buf[aid] = ""
                    slot[aid] = len(history)
                    last_yield_len[aid] = 0
                    history.append({"role": "assistant",
                                    "content": _bubble(aid, AI_NAME_BY_ID[aid], CURSOR)})
                    yield _chat_only(history, clear_msg=False)
                elif ev == "update":
                    buf[aid] = buf.get(aid, "") + payload
                    history[slot[aid]] = {
                        "role": "assistant",
                        "content": _bubble(aid, AI_NAME_BY_ID[aid], buf[aid] + CURSOR),
                    }
                    cur_len = len(buf[aid])
                    if cur_len - last_yield_len.get(aid, 0) >= THROTTLE_CHARS or "\n" in payload:
                        last_yield_len[aid] = cur_len
                        yield _chat_only(history, clear_msg=False)
                elif ev == "done":
                    done_payloads[aid] = payload
                    history[slot[aid]] = {
                        "role": "assistant",
                        "content": _bubble(aid, AI_NAME_BY_ID[aid], payload),
                    }
                    yield _chat_only(history, clear_msg=False)

            # 1 발화 = 1 버블 - 스트리밍 중 슬롯 1개에 전문이 이미 채워졌다.

            if decision.get("stage_complete"):
                history.append({"role": "assistant",
                                "content": _system_bubble(f"Stage {session.current_stage} 완료")})
                if session.advance_stage():
                    s = session.current_stage_info()
                    history.append({"role": "assistant",
                                    "content": _stage_card(session.current_stage, s)})
                    # 새 Stage에 intro_message가 있으면 system 버블로 먼저 렌더링
                    intro_msg = s.get("intro_message")
                    if intro_msg:
                        history.append({"role": "assistant",
                                        "content": _system_bubble(f"📘 **수업 안내**\n\n{intro_msg}")})
                    yield _chat_only(history, clear_msg=False)
                    intro = session.stage_intro_utterance("ai_2")
                    history.extend(_bubble_messages("ai_2", n2, intro))
                else:
                    history.append({"role": "assistant",
                                    "content": _system_bubble("모든 Stage 완료. 오른쪽 탭이 자동으로 최신 상태입니다.")})

            yield _refresh_bundle(history, clear_msg=False)
        finally:
            _streaming_flag[0] = False

    def on_silence_tick(history):
        if _streaming_flag[0]:
            return history
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

    # 버전 표시 (lib/__init__.py의 __version__ 단일 소스)
    try:
        from . import __version__ as _app_version
    except Exception:
        _app_version = "v?.??"

    with gr.Blocks(title=f"협력학습 세션 {_app_version}", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            f"# {session.task['task_title']}   "
            f"<span style='font-size:0.55em; color:#888; font-weight:400; "
            f"background:#f0f0f0; padding:3px 8px; border-radius:10px; "
            f"margin-left:8px; vertical-align:middle;'>"
            f"⚙️ {_app_version}</span>"
        )
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
                    next_btn = gr.Button("다음 Stage")
                    refresh_all_btn = gr.Button("대시보드 전체 갱신")

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("① 영역 요약"):
                        gr.Markdown("_인지·정의 두 카테고리의 최상위 점수를 바 차트로 조감합니다. 하위 차원은 학습자 모델 탭에서 확인하세요._")
                        radar = gr.Plot(label="사용자 영역 요약 (인지/정의)")
                        gr.Button("갱신").click(_radar, outputs=radar)
                    with gr.Tab("② 학습자 모델"):
                        gr.Markdown("_인지/정의 카테고리별 하위요소 점수와 루브릭 해석입니다._")
                        model_md = gr.Markdown()
                        gr.Button("갱신").click(_model_md, outputs=model_md)
                    with gr.Tab("③ 변화 추이"):
                        gr.Markdown("_업데이트 회차에 따른 각 하위요소의 변화 추이입니다._")
                        hist_plot = gr.Plot(label="사용자 단계별 변화")
                        gr.Button("갱신").click(_history, outputs=hist_plot)
                    with gr.Tab("④ 교수자 디시젼"):
                        gr.Markdown("_가장 최근 턴의 교수자 모델이 내린 의사결정(JSON)입니다._")
                        dec_code = gr.Code(language="json", label="last_tutor_decision")
                        gr.Button("갱신").click(_decision_json, outputs=dec_code)
                    with gr.Tab("⑤ 오개념 타임라인"):
                        gr.Markdown(
                            "_관찰된 오개념이 언제 등장해서 언제 해소됐는지 간트 차트로 표시합니다. "
                            "초록=해소, 빨강=현재 지속._"
                        )
                        misc_plot = gr.Plot(label="오개념 타임라인")
                        gr.Button("갱신").click(_misconception_timeline, outputs=misc_plot)
                    with gr.Tab("⑥ CPS 히트맵"):
                        gr.Markdown(
                            "_Stage × CPS 하위구인(공동이해·행동·조직화·수정) 매트릭스._"
                        )
                        cps_plot = gr.Plot(label="CPS 히트맵")
                        gr.Button("갱신").click(_cps_heatmap, outputs=cps_plot)

        auto_outputs = [chatbot, msg, radar, hist_plot, model_md, dec_code, misc_plot, cps_plot]

        send.click(
            echo_user, [msg, chatbot], [msg, chatbot], queue=False
        ).then(stream_ai, chatbot, auto_outputs)
        msg.submit(
            echo_user, [msg, chatbot], [msg, chatbot], queue=False
        ).then(stream_ai, chatbot, auto_outputs)
        next_btn.click(on_next_stage, chatbot, chatbot)
        refresh_all_btn.click(
            lambda: (
                _radar(), _history(), _model_md(), _decision_json(),
                _misconception_timeline(), _cps_heatmap(),
            ),
            outputs=[radar, hist_plot, model_md, dec_code, misc_plot, cps_plot],
        )

        # 침묵 타이머: 120s. 프롬프트 다이어트 이후에도 토큰 예산 아끼려 폴링 주기는 길게.
        # 침묵 임계치(SILENCE_THRESHOLD_SECONDS=60)는 session.py에서 별도 판정.
        silence_timer = gr.Timer(value=120.0, active=True)
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
