"""Gradio 채팅 인터페이스 (V5, 1 발화 = 1 버블).

왼쪽 패널: Chatbot(발화자별 컬러 버블) + 입력 / 오른쪽 탭: 레이더 · 학습자모델 · 변화 추이 · 교수자 Decision · 오개념 · CPS
"""

import json

from .session import CollaborativeSession
from .visualize import (
    radar_figure,
    history_figure,
    user_model_markdown,
    checkpoint_markdown,
    misconception_timeline_figure,
    cps_heatmap_figure,
)


def launch_ui(*, config, prompts, learner_models, api, share=True, reset=True):
    """v1.53: reset=True (기본값) 시 launch마다 learner_models·conversation 초기화.
    이전 세션 누적으로 user 컬럼에 잘못된 hit이 남는 문제 방지.
    이전 진척을 이어받으려면 reset=False로 호출.
    """
    import gradio as gr

    if reset:
        from .learner_model import init_learners as _init_learners
        # ctx의 learner_models dict를 새 결과로 in-place 갱신 (외부 ctx도 동기화)
        fresh = _init_learners(config)
        learner_models.clear()
        learner_models.update(fresh)
        print(f"       · [launch_ui] reset=True → learner_models 초기화 완료", flush=True)

    session = CollaborativeSession(config, prompts, learner_models, api)
    # v1.56: 외부 진단용으로 session을 config dict에 노출
    # (Colab thread print 안 보이는 환경에서 사용자가 직접 dump 가능)
    config["_session"] = session

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
        # prompt_html 필드가 있으면 그것을 그대로 사용 (테이블 등 HTML 그대로 표시).
        # 없으면 plain prompt 의 \n 을 <br> 로 변환.
        prompt_html = stage_info.get("prompt_html") or prompt.replace("\n", "<br>")
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

    def _checkpoint_md():
        return checkpoint_markdown(config, learner_models)

    def _decision_json():
        d = getattr(session, "last_tutor_decision", None) or {}
        return json.dumps(d, ensure_ascii=False, indent=2)

    def _misconception_timeline():
        return misconception_timeline_figure(config, learner_models)

    def _cps_heatmap():
        return cps_heatmap_figure(config, learner_models)

    def _task_panel_html():
        """오른쪽 패널 상단에 항상 노출되는 '현재 과제' 카드.
        session.current_stage 기준으로 _stage_card 와 같은 모양을 그려서
        스크롤되며 사라지는 채팅 stage_card 의 영구 미러 역할을 한다.
        session.all_stages_complete=True 가 셋되면 마무리 카드로 교체.
        """
        try:
            if getattr(session, "all_stages_complete", False):
                return (
                    '<div style="border:2px solid #16a34a; border-radius:10px;'
                    ' overflow:hidden; margin:0 0 8px 0;">'
                    '<div style="background:#16a34a; color:white; padding:10px 16px;'
                    ' font-weight:700;">🎉 모든 Stage 완료</div>'
                    '<div style="background:#f0fdf4; padding:14px 16px; color:#14532d;">'
                    '협력학습이 모두 끝났습니다. 채팅창의 마무리 발화와 오른쪽 탭의 분석 결과를 확인해 보세요.'
                    '</div></div>'
                )
            s = session.current_stage_info()
            return _stage_card(session.current_stage, s)
        except Exception as e:
            return (
                f'<div style="border-left:4px solid #dc2626; background:#fef2f2;'
                f' padding:8px 12px; border-radius:6px; font-size:0.85em;">'
                f'과제 패널 렌더 오류: {type(e).__name__}: {e}</div>'
            )

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
            _checkpoint_md(),
            _task_panel_html(),
        )

    def _chat_only(history, clear_msg=False):
        return (
            history,
            "" if clear_msg else gr.update(),
            gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(),
            gr.update(),
            gr.update(),
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

            # --- Stage 종료 판단 우선 ---
            # 완료 턴에는 이번 턴의 AI 발화를 생략하고 즉시 종료 처리 + 다음 Stage intro.
            # (사용자가 "23, 29 맞아" 확정 → 불필요한 호응 발화 없이 바로 Stage 2 종료)
            if decision.get("stage_complete"):
                history.append({"role": "assistant",
                                "content": _system_bubble(f"Stage {session.current_stage} 완료")})
                yield _chat_only(history, clear_msg=False)
                if session.advance_stage():
                    s = session.current_stage_info()
                    # v1.43: 정의 안내(intro_message)를 stage_card보다 먼저 표시
                    intro_msg = session.get_stage_intro_message()
                    if intro_msg:
                        history.append({"role": "assistant",
                                        "content": _system_bubble(intro_msg)})
                        yield _chat_only(history, clear_msg=False)
                    history.append({"role": "assistant",
                                    "content": _stage_card(session.current_stage, s)})
                    yield _chat_only(history, clear_msg=False)
                    intro = session.stage_intro_utterance("ai_2")
                    history.extend(_bubble_messages("ai_2", n2, intro))
                else:
                    # v1.71: 모든 Stage 완료 시 AI 학생들이 한 마디씩 마무리 발화.
                    history.append({"role": "assistant",
                                    "content": _system_bubble("🎉 모든 Stage 완료!")})
                    yield _chat_only(history, clear_msg=False)
                    for msg in session.generate_completion_messages_llm():
                        aid = msg.get("agent_id")
                        text = msg.get("text", "")
                        if not aid or not text:
                            continue
                        history.extend(_bubble_messages(aid, AI_NAME_BY_ID[aid], text))
                        yield _chat_only(history, clear_msg=False)
                yield _refresh_bundle(history, clear_msg=False)
                return  # 이번 턴 AI 발화 생성 완전 스킵

            # user_mode == "teacher" 감지는 directive에 반영되지만 UI 안내문은 표시 안 함
            # (사용자가 교수자 모드 전환을 의식하지 않고 자연스럽게 진행하도록)

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
                elif ev == "buffering":
                    # 스트림 종료 + 재시도 진행 중. 커서 빼고 "💭 다시 생각 중..." 부착.
                    history[slot[aid]] = {
                        "role": "assistant",
                        "content": _bubble(
                            aid, AI_NAME_BY_ID[aid],
                            (buf.get(aid, "") or payload).rstrip()
                            + "\n\n_💭 다시 생각 중..._",
                        ),
                    }
                    yield _chat_only(history, clear_msg=False)
                elif ev == "done":
                    done_payloads[aid] = payload
                    history[slot[aid]] = {
                        "role": "assistant",
                        "content": _bubble(aid, AI_NAME_BY_ID[aid], payload),
                    }
                    yield _chat_only(history, clear_msg=False)

            # 1 발화 = 1 버블 - 스트리밍 중 슬롯 1개에 전문이 이미 채워졌다.

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
            # v1.71: 수동 종료 시에도 마무리 발화 노출.
            history = history + [{"role": "assistant",
                                  "content": _system_bubble("🎉 모든 Stage 완료!")}]
            for msg in session.generate_completion_messages_llm():
                aid = msg.get("agent_id")
                text = msg.get("text", "")
                if not aid or not text:
                    continue
                history = history + _bubble_messages(aid, AI_NAME_BY_ID[aid], text)
        # v1.73: task_panel HTML도 함께 반환해서 우상단 패널 갱신.
        return history, _task_panel_html()

    # 버전 표시 (lib/__init__.py의 __version__ 단일 소스)
    try:
        from . import __version__ as _app_version
    except Exception:
        _app_version = "v?.??"

    # Gradio 버전 감지 — 5.x와 6.0+ 사이 API 변경 호환 처리
    # - 5.x: gr.Blocks(theme=...), gr.Chatbot(type="messages")
    # - 6.0: theme은 launch()로 이동, Chatbot은 messages가 기본이라 type 인자 불필요
    _gr_major = 5
    try:
        _v = getattr(gr, "__version__", "5.0.0")
        _gr_major = int(str(_v).split(".")[0])
    except Exception:
        pass

    _blocks_kwargs = {"title": f"협력학습 세션 {_app_version}"}
    if _gr_major < 6:
        _blocks_kwargs["theme"] = gr.themes.Soft()

    with gr.Blocks(**_blocks_kwargs) as demo:
        gr.Markdown(
            f"# {session.task['task_title']}   "
            f"<span style='font-size:0.55em; color:#888; font-weight:400; "
            f"background:#f0f0f0; padding:3px 8px; border-radius:10px; "
            f"margin-left:8px; vertical-align:middle;'>"
            f"⚙️ {_app_version}</span>"
        )
        with gr.Row():
            with gr.Column(scale=3):
                _chatbot_kwargs = {
                    "value": [],
                    "height": 560,
                    "label": "세션 대화",
                    "show_copy_button": True,
                }
                # 5.x만 type="messages" 명시 (6.x는 기본이 messages라 불필요/오류)
                if _gr_major < 6:
                    _chatbot_kwargs["type"] = "messages"
                chatbot = gr.Chatbot(**_chatbot_kwargs)
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
                # v1.73: 채팅에서 stage_card 가 스크롤로 사라지는 문제 해결.
                # 탭 위에 항상 보이는 "현재 과제" 패널을 둔다.
                task_panel = gr.HTML(value=_task_panel_html(), label="현재 과제")
                with gr.Tabs():
                    with gr.Tab("① 영역 요약"):
                        gr.Markdown("_인지·정의 두 카테고리의 최상위 점수를 바 차트로 조감합니다. 하위 차원은 학습자 모델 탭에서 확인하세요._")
                        radar = gr.Plot(label="사용자 영역 요약 (인지/정의)")
                        gr.Button("갱신").click(_radar, outputs=radar)
                    with gr.Tab("② 지식 체크포인트"):
                        checkpoint_md = gr.Markdown()
                        gr.Button("갱신").click(_checkpoint_md, outputs=checkpoint_md)
                    with gr.Tab("③ 학습자 모델"):
                        gr.Markdown("_인지/정의 카테고리별 하위요소 점수와 루브릭 해석입니다._")
                        model_md = gr.Markdown()
                        gr.Button("갱신").click(_model_md, outputs=model_md)
                    with gr.Tab("④ 변화 추이"):
                        gr.Markdown("_업데이트 회차에 따른 각 하위요소의 변화 추이입니다._")
                        hist_plot = gr.Plot(label="사용자 단계별 변화")
                        gr.Button("갱신").click(_history, outputs=hist_plot)
                    with gr.Tab("⑤ 교수자 디시젼"):
                        gr.Markdown("_가장 최근 턴의 교수자 모델이 내린 의사결정(JSON)입니다._")
                        dec_code = gr.Code(language="json", label="last_tutor_decision")
                        gr.Button("갱신").click(_decision_json, outputs=dec_code)
                    with gr.Tab("⑥ 오개념 타임라인"):
                        gr.Markdown(
                            "_관찰된 오개념이 언제 등장해서 언제 해소됐는지 간트 차트로 표시합니다. "
                            "초록=해소, 빨강=현재 지속._"
                        )
                        misc_plot = gr.Plot(label="오개념 타임라인")
                        gr.Button("갱신").click(_misconception_timeline, outputs=misc_plot)
                    with gr.Tab("⑦ CPS 히트맵"):
                        gr.Markdown(
                            "_Stage × CPS 하위구인(공동이해·행동·조직화·수정) 매트릭스._"
                        )
                        cps_plot = gr.Plot(label="CPS 히트맵")
                        gr.Button("갱신").click(_cps_heatmap, outputs=cps_plot)

        auto_outputs = [chatbot, msg, radar, hist_plot, model_md, dec_code, misc_plot, cps_plot, checkpoint_md, task_panel]

        send.click(
            echo_user, [msg, chatbot], [msg, chatbot], queue=False
        ).then(stream_ai, chatbot, auto_outputs)
        msg.submit(
            echo_user, [msg, chatbot], [msg, chatbot], queue=False
        ).then(stream_ai, chatbot, auto_outputs)
        # v1.73: 다음 Stage 버튼은 채팅 + task_panel 두 곳을 갱신.
        next_btn.click(on_next_stage, chatbot, [chatbot, task_panel])
        refresh_all_btn.click(
            lambda: (
                _radar(), _history(), _model_md(), _decision_json(),
                _misconception_timeline(), _cps_heatmap(), _checkpoint_md(),
                _task_panel_html(),
            ),
            outputs=[radar, hist_plot, model_md, dec_code, misc_plot, cps_plot, checkpoint_md, task_panel],
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
        demo.load(_checkpoint_md, outputs=checkpoint_md)

    # Gradio 6.0+에서는 theme가 launch() 인자로 이동
    _launch_kwargs = {"share": share, "debug": False, "quiet": True}
    if _gr_major >= 6:
        try:
            _launch_kwargs["theme"] = gr.themes.Soft()
        except Exception:
            pass
    demo.launch(**_launch_kwargs)
    return demo
