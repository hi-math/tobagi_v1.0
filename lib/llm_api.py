"""Claude API 호출 래퍼와 프롬프트 유틸.

외부에서 anthropic.Anthropic 클라이언트를 주입 받아 ClaudeAPI 인스턴스를 만든다.

기능:
- call(): 동기 1회성 호출 (기존)
- call(stream=True): 토큰 제너레이터 반환 (UX 개선용)
- model= per-call 오버라이드: analyze는 Haiku, 발화는 Sonnet 등 분리 라우팅
"""

import json
import re


# 기본 모델 상수 — 역할별로 속도/품질 트레이드오프
DEFAULT_SONNET = "claude-sonnet-4-20250514"
DEFAULT_HAIKU = "claude-haiku-4-5-20251001"


class ClaudeAPI:
    """anthropic 클라이언트를 래핑하는 얇은 API 헬퍼.

    `call(stream=False)`로 호출하면 문자열을 반환하고,
    `call(stream=True)`로 호출하면 토큰 청크를 yield하는 제너레이터를 반환한다.
    `model=` 인자로 호출마다 모델을 오버라이드할 수 있다 (analyze는 Haiku, 발화는 Sonnet).
    """

    def __init__(self, client, model=DEFAULT_SONNET):
        self.client = client
        self.model = model

    def call(self, prompt, max_tokens=1000, temperature=0.7, model=None, stream=False):
        """Claude 호출.

        stream=False  → 완료된 응답 텍스트(str) 반환 (기본)
        stream=True   → 토큰 청크(str)를 yield하는 제너레이터 반환
                        (full text는 마지막에 ''.join하면 됨)

        model=None이면 인스턴스 기본 모델 사용.
        """
        use_model = model or self.model
        if stream:
            return self._call_stream(prompt, max_tokens, temperature, use_model)
        resp = self.client.messages.create(
            model=use_model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    def _call_stream(self, prompt, max_tokens, temperature, model):
        """토큰 청크를 yield하는 내부 제너레이터.

        anthropic SDK의 messages.stream()을 사용한다. text_stream 이터레이터는
        순수 텍스트 델타만 내보내며, thread-safe하게 여러 스트림을 병렬로 돌릴 수 있다.
        """
        with self.client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        ) as s:
            for chunk in s.text_stream:
                if chunk:
                    yield chunk


def extract_json(text):
    """LLM 응답에서 JSON 블록을 추출해 dict로 반환.

    순서:
        1) ```json ... ``` 코드 블록
        2) 첫 '{' 부터 마지막 '}' 까지
    """
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        return json.loads(text[start:end])
    raise ValueError("JSON을 추출할 수 없습니다:\n" + text)


def render_prompt(template, vars_dict):
    """{{var}} 플레이스홀더를 vars_dict 값으로 치환.

    dict/list 값은 `json.dumps(..., ensure_ascii=False, indent=2)` 로 직렬화.
    """
    out = template
    for k, v in vars_dict.items():
        placeholder = "{{" + k + "}}"
        if isinstance(v, (dict, list)):
            v = json.dumps(v, ensure_ascii=False, indent=2)
        out = out.replace(placeholder, str(v))
    return out
