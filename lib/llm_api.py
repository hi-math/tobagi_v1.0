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
    api = GeminiAPI(model="gemini-2.5-flash")

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
# 2026-04 기준: Google이 2.0 Flash 계열을 신규 유저에게 막음(404). 2.5 계열로 통일.
# 2.5 Flash-Lite: 무료 티어 최고 속도·저비용. 분석용으로 추천.
# 2.5 Flash      : 품질 중시. AI 발화용 기본.
DEFAULT_GEMINI_FLASH = "gemini-2.5-flash"
DEFAULT_GEMINI_FLASH_LITE = "gemini-2.5-flash-lite"
DEFAULT_GEMINI_25 = "gemini-2.5-flash"

# ---- OpenAI 모델 상수 ----
# RECITATION 필터가 없어 교과 표준 문구 안정. 비용은 Gemini 보다 높지만 품질·안정성 우수.
# gpt-4o-mini     : 저렴·빠름, 발화 품질 충분. 기본 선택.
# gpt-4o          : 플래그십. 섬세한 상호작용 필요 시.
# gpt-4.1-mini    : 최신, 4o-mini 후속
DEFAULT_OPENAI_MINI = "gpt-4o-mini"
DEFAULT_OPENAI_FULL = "gpt-4o"


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
    """google-genai (신규 공식 SDK) 클라이언트를 래핑한 얇은 API 헬퍼.

    중요: Colab의 구 `google.generativeai` 전용 import hook이 로컬 프록시로
    라우팅하는 문제(localhost:44179 Read timeout) 를 피하려고 **신규 SDK만**
    사용한다. 설치: `pip install google-genai` (기존 `google-generativeai` 아님).

    인터페이스는 ClaudeAPI와 동일:
        call(prompt, max_tokens=..., temperature=..., model=None, stream=False,
             json_mode=False) -> str
        call(..., stream=True) -> Iterator[str]

    노트:
    - 신규 SDK는 `google.genai.Client(api_key=...)`로 초기화하고 인스턴스를 보관.
    - 동기 호출:   client.models.generate_content(model=..., contents=..., config=...)
    - 스트리밍:   client.models.generate_content_stream(model=..., contents=..., config=...)
    - config는 `google.genai.types.GenerateContentConfig(...)`.
    """

    provider = "gemini"

    def __init__(self, model=DEFAULT_GEMINI_FLASH, api_key=None):
        try:
            from google import genai as _genai
            from google.genai import types as _types
        except ImportError as e:
            raise ImportError(
                "google-genai 패키지가 필요합니다. Colab에서 "
                "`!pip install google-genai -q` 를 먼저 실행하세요. "
                f"(원본: {e})"
            )
        self._genai = _genai
        self._types = _types
        self.model = model
        # api_key=None이면 google-genai가 GOOGLE_API_KEY / GEMINI_API_KEY 환경변수를 찾음.
        self._client = _genai.Client(api_key=api_key) if api_key else _genai.Client()

    def _build_config(self, max_tokens, temperature, json_mode):
        kwargs = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_mime_type"] = "application/json"
        return self._types.GenerateContentConfig(**kwargs)

    def call(self, prompt, max_tokens=1000, temperature=0.7, model=None, stream=False,
             json_mode=False):
        """json_mode=True면 response_mime_type='application/json'을 강제해 Gemini가
        JSON 블록만 반환하도록 제한한다. extract_json 실패로 fallback 경로가 타는
        것을 막아 레이턴시가 크게 줄어든다."""
        use_model = model or self.model
        config = self._build_config(max_tokens, temperature, json_mode)
        if stream:
            return self._call_stream(use_model, prompt, config)
        resp = self._client.models.generate_content(
            model=use_model, contents=prompt, config=config,
        )
        # finish_reason 진단: MAX_TOKENS(정상 종료) / STOP(정상) / SAFETY / RECITATION / OTHER
        # RECITATION: 교과서 문장과 유사해 차단 → 조기 종료로 짧은 응답 원인
        # SAFETY:     안전 필터 차단
        for c in getattr(resp, "candidates", []) or []:
            fr = getattr(c, "finish_reason", None)
            if fr and str(fr).upper() not in ("STOP", "FINISH_REASON_STOP", "1"):
                print(f"       · [gemini {use_model}] finish_reason={fr}")
                break
        # 안전 필터 차단 등으로 text 속성이 비어있을 수 있음 → 파트 긁어모으기 폴백
        try:
            text = resp.text
            if text:
                return text
        except Exception:
            pass
        parts = []
        for c in getattr(resp, "candidates", []) or []:
            content = getattr(c, "content", None)
            if content and getattr(content, "parts", None):
                for p in content.parts:
                    t = getattr(p, "text", None)
                    if t:
                        parts.append(t)
        return "".join(parts) if parts else ""

    def _call_stream(self, model, prompt, config):
        resp_iter = self._client.models.generate_content_stream(
            model=model, contents=prompt, config=config,
        )
        for chunk in resp_iter:
            t = getattr(chunk, "text", None)
            if t:
                yield t


class OpenAIAPI:
    """OpenAI `openai` SDK(v1.x+)를 래핑한 얇은 API 헬퍼.

    RECITATION 필터가 없어 교과 표준 문구가 안정적. Gemini 대비 비용은 높지만
    발화 품질과 응답 완결성이 더 우수.

    인터페이스는 ClaudeAPI/GeminiAPI와 동일:
        call(prompt, max_tokens=..., temperature=..., model=None,
             stream=False, json_mode=False) -> str
        call(..., stream=True) -> Iterator[str]

    설치: `pip install openai>=1.0`
    """

    provider = "openai"

    def __init__(self, model=DEFAULT_OPENAI_MINI, api_key=None):
        try:
            from openai import OpenAI as _OpenAI
        except ImportError as e:
            raise ImportError(
                "openai 패키지가 필요합니다. Colab에서 "
                "`!pip install -U openai` 를 먼저 실행하세요. "
                f"(원본: {e})"
            )
        self._OpenAI = _OpenAI
        self.model = model
        # api_key=None이면 SDK가 OPENAI_API_KEY 환경변수에서 자동 로드
        self._client = _OpenAI(api_key=api_key) if api_key else _OpenAI()

    _REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")

    def _is_reasoning_model(self, name):
        if not name:
            return False
        m = name.lower()
        for p in self._REASONING_PREFIXES:
            if m.startswith(p):
                return True
        return False

    def _build_kwargs(self, prompt, max_tokens, temperature, model, json_mode):
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self._is_reasoning_model(model):
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    def call(self, prompt, max_tokens=1000, temperature=0.7, model=None, stream=False,
             json_mode=False):
        use_model = model or self.model
        kwargs = self._build_kwargs(prompt, max_tokens, temperature, use_model, json_mode)
        if stream:
            return self._call_stream(kwargs)
        kwargs["stream"] = False
        resp = self._client.chat.completions.create(**kwargs)
        try:
            msg = resp.choices[0].message.content
            return msg or ""
        except Exception:
            return ""

    def _call_stream(self, kwargs):
        kwargs["stream"] = True
        stream = self._client.chat.completions.create(**kwargs)
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    yield content
            except (AttributeError, IndexError):
                continue


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
