"""LLM API 호출 래퍼와 프롬프트 유틸.

Gemini와 Claude를 모두 지원하는 얇은 래퍼. 기본은 **Gemini 2.0 Flash (무료 티어)**.
인스턴스가 제공하는 `.call()` 인터페이스는 공급자와 무관하게 동일:
    api.call(prompt, max_tokens=..., temperature=..., model=None, stream=False) -> str
    api.call(..., stream=True) -> Iterator[str]  (토큰 청크)

session.py 쪽에서는 `self.api.call(...)` 만 호출하므로, 래퍼를 바꿔 끼우면
공급자 교체가 투명하게 된다.

사용:
    # Gemini (무료/저렴, 기본 권장)
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_KEY)
    api = GeminiAPI(model="gemini-2.0-flash")

    # Anthropic (기존)
    client = anthropic.Anthropic(api_key=CLAUDE_KEY)
    api = ClaudeAPI(client, model=DEFAULT_HAIKU)

bootstrap()은 provider= 인자로 둘 중 하나를 선택한다 (기본 "gemini").
"""

import json
import re


# ---- Anthropic 모델 상수 (기존 Claude 경로 유지용) ----
DEFAULT_SONNET = "claude-sonnet-4-20250514"
DEFAULT_HAIKU = "claude-haiku-4-5-20251001"

# ---- Gemini 모델 상수 ----
# 2.0 Flash: 무료 티어 지원. 2.5 Flash는 유료지만 더 똑똑.
DEFAULT_GEMINI_FLASH = "gemini-2.0-flash"
DEFAULT_GEMINI_FLASH_LITE = "gemini-2.0-flash-lite"  # 더 빠르고 저렴
DEFAULT_GEMINI_25 = "gemini-2.5-flash"               # 품질 필요 시


class ClaudeAPI:
    """anthropic 클라이언트를 래핑하는 얇은 API 헬퍼.

    `call(stream=False)`로 호출하면 문자열을 반환하고,
    `call(stream=True)`로 호출하면 토큰 청크를 yield하는 제너레이터를 반환한다.
    `model=` 인자로 호출마다 모델을 오버라이드할 수 있다.
    """

    provider = "anthropic"

    def __init__(self, client, model=DEFAULT_HAIKU):
        self.client = client
        self.model = model

    def call(self, prompt, max_tokens=1000, temperature=0.7, model=None, stream=False,
             json_mode=False):
        # ClaudeAPI는 json_mode 인자를 무시 (Anthropic은 JSON 모드가 별도 필요 없음).
        _ = json_mode
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
        with self.client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        ) as s:
            for chunk in s.text_stream:
                if chunk:
                    yield chunk


class GeminiAPI:
    """google-generativeai 클라이언트를 래핑한 얇은 API 헬퍼.

    인터페이스는 ClaudeAPI와 동일:
        call(prompt, max_tokens=..., temperature=..., model=None, stream=False) -> str
        call(..., stream=True) -> Iterator[str]

    사전 조건: `genai.configure(api_key=...)` 가 이미 호출되어 있어야 한다.
    bootstrap()이 이 초기화를 대신 해준다.

    노트:
    - Gemini SDK는 `max_output_tokens`, `temperature`를 generation_config로 받는다.
    - 스트리밍: `model.generate_content(prompt, stream=True)` → chunk.text 이터레이터.
    - 모델 인스턴스는 model 이름마다 캐시해 재사용 (호출마다 만들면 오버헤드).
    """

    provider = "gemini"

    def __init__(self, model=DEFAULT_GEMINI_FLASH):
        import google.generativeai as genai  # 지연 임포트
        self._genai = genai
        self.model = model
        self._model_cache = {}  # model_name -> GenerativeModel

    def _get_model(self, name):
        if name not in self._model_cache:
            self._model_cache[name] = self._genai.GenerativeModel(name)
        return self._model_cache[name]

    def call(self, prompt, max_tokens=1000, temperature=0.7, model=None, stream=False,
             json_mode=False):
        """json_mode=True면 response_mime_type='application/json'을 강제해 Gemini가
        JSON 블록만 반환하도록 제한한다. extract_json 실패로 fallback 경로가 타는
        것을 막아 레이턴시가 크게 줄어든다."""
        use_model = model or self.model
        gen_config = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            gen_config["response_mime_type"] = "application/json"
        gm = self._get_model(use_model)
        if stream:
            return self._call_stream(gm, prompt, gen_config)
        resp = gm.generate_content(prompt, generation_config=gen_config)
        # 안전 필터 차단 등으로 text 속성이 없을 수 있음 → 폴백
        try:
            return resp.text
        except Exception:
            # 파트 단위로 긁어모으기
            parts = []
            for c in getattr(resp, "candidates", []) or []:
                content = getattr(c, "content", None)
                if content and getattr(content, "parts", None):
                    for p in content.parts:
                        t = getattr(p, "text", None)
                        if t:
                            parts.append(t)
            if parts:
                return "".join(parts)
            # 마지막 수단: 빈 문자열. JSON extract_json이 ValueError를 내줄 것
            return ""

    def _call_stream(self, gm, prompt, gen_config):
        resp_iter = gm.generate_content(prompt, generation_config=gen_config, stream=True)
        for chunk in resp_iter:
            t = getattr(chunk, "text", None)
            if t:
                yield t


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
