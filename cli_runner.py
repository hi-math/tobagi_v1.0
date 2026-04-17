"""레거시 CLI 런너.

`input()` 루프로 돌아가는 텍스트 모드 세션. Gradio 가 동작하지 않는 환경이나
디버깅·단독 점검용으로 남겨둔다.
"""

import json

from .session import CollaborativeSession
from .visualize import (
    print_user_model,
    plot_radar_all,
    plot_user_history,
)


def run_session(*, config, prompts, learner_models, api):
    """CLI 세션 실행.

    명령어:
        /radar     — 3명 비교 레이더
        /history   — 사용자 변화 추이
        /model     — 사용자 학습자 모델 상세
        /next      — 다음 stage 수동 이동
        /decision  — 직전 tutor_decision JSON
        /quit      — 종료 (최종 시각화 출력 후 세션 반환)
    """
    session = CollaborativeSession(config, prompts, learner_models, api)
    print(f"\n{'═'*60}\n🎓 {session.task['task_title']}\n{'═'*60}")

    n1 = config["personas"]["ai_students"]["ai_1"]["name"]
    n2 = config["personas"]["ai_students"]["ai_2"]["name"]

    # Stage 1 intro
    stage = session.current_stage_info()
    print(f"\n📝 {stage['title']}")
    print(f"   {stage['prompt']}\n")
    print("   (AI 학생이 먼저 말문을 엽니다...)\n")
    intro = session.stage_intro_utterance("ai_1")
    print(f"🧑‍🎓 {n1}: {intro}")

    while True:
        user_input = input("\n🙋 나: ").strip()
        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd == "/quit":
                print("\n📊 최종 학습자 모델:")
                print_user_model(config, learner_models)
                plot_radar_all(config, learner_models)
                plot_user_history(config, learner_models)
                break
            elif cmd == "/radar":
                plot_radar_all(config, learner_models); continue
            elif cmd == "/history":
                plot_user_history(config, learner_models); continue
            elif cmd == "/model":
                print_user_model(config, learner_models); continue
            elif cmd == "/next":
                if session.advance_stage():
                    print(f"\n▶ Stage {session.current_stage}로 이동")
                    stage = session.current_stage_info()
                    print(f"📝 {stage['title']}\n   {stage['prompt']}\n")
                    intro = session.stage_intro_utterance("ai_2")
                    print(f"🧑‍🎓 {n2}: {intro}")
                else:
                    print("  모든 stage 완료")
                continue
            elif cmd == "/decision":
                print(json.dumps(session.last_tutor_decision, ensure_ascii=False, indent=2))
                continue
            else:
                print("  알 수 없는 명령어"); continue

        result = session.user_turn(user_input)
        print(f"\n🧑‍🎓 {n1}: {result['ai_1']}")
        print(f"🧑‍🎓 {n2}: {result['ai_2']}")

        # stage 자동 전환
        if result["decision"].get("stage_complete"):
            print(f"\n✅ Stage {session.current_stage} 완료 신호")
            if session.advance_stage():
                stage = session.current_stage_info()
                print(f"\n▶ Stage {session.current_stage} 시작")
                print(f"📝 {stage['title']}\n   {stage['prompt']}\n")
                intro = session.stage_intro_utterance("ai_2")
                print(f"🧑‍🎓 {n2}: {intro}")
            else:
                print("\n🎉 모든 Stage 완료! /quit 으로 종료하세요.")

    return session
