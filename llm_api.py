"""Claude API 호출 래퍼와 프롬프트 유틸.

외부에서 anthropic.Anthropic 클라이언트를 주입 받아 ClaudeAPI 인스턴스를 만든다.
"""

import json
import re


class ClaudeAPI:
    """anthropic 클라이언트를 래핑하는 얇은 API 헬퍼."""

    def __init__(self, client, model="claude-sonnet-4-20250514"):
        self.client = client
        self.model = model

    def call(self, prompt, max_tokens=1000, temperature=0.7):
        """단일 사용자 메시지로 Claude 호출하고 응답 텍스트를 반환."""
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text


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
