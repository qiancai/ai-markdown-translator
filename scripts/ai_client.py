"""
Unified AI Client
Provides a unified interface for multiple AI providers:
  - deepseek: DeepSeek Chat  (env: DEEPSEEK_API_TOKEN)
  - gemini:   Google Gemini   (env: GEMINI_API_TOKEN)
  - openai:   OpenAI GPT      (env: OPENAI_API_TOKEN)
  - azure:    Azure OpenAI    (env: AZURE_OPENAI_KEY + OPENAI_BASE_URL)
"""

import os
import time
import threading
from log_sanitizer import sanitize_exception_message

# ---------------------------------------------------------------------------
# Gemini SDK – try the newer google-genai first, fall back to legacy
# ---------------------------------------------------------------------------
_GEMINI_NEW_SDK = False
GEMINI_AVAILABLE = False

try:
    from google import genai                    # google-genai (new)
    GEMINI_AVAILABLE = True
    _GEMINI_NEW_SDK = True
except ImportError:
    try:
        import google.generativeai as genai     # google-generativeai (legacy)
        GEMINI_AVAILABLE = True
    except ImportError:
        genai = None

# ---------------------------------------------------------------------------
# Thread-safe printing (also re-exported for callers)
# ---------------------------------------------------------------------------
print_lock = threading.Lock()

def thread_safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)

# ---------------------------------------------------------------------------
# API keys / endpoints  (read once at import time)
# ---------------------------------------------------------------------------
DEEPSEEK_API_KEY      = os.getenv("DEEPSEEK_API_TOKEN")
DEEPSEEK_BASE_URL     = "https://api.deepseek.com"
GEMINI_API_KEY        = os.getenv("GEMINI_API_TOKEN")
OPENAI_API_KEY        = os.getenv("OPENAI_API_TOKEN")
AZURE_OPENAI_KEY      = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# ---------------------------------------------------------------------------
# Model names  (override via env var if needed)
# ---------------------------------------------------------------------------
DEEPSEEK_MODEL_NAME       = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")
GEMINI_MODEL_NAME         = os.getenv("GEMINI_MODEL_NAME", "gemini-3-flash-preview")
OPENAI_MODEL_NAME         = os.getenv("OPENAI_MODEL_NAME", "gpt-4.1")
AZURE_OPENAI_MODEL_NAME   = os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-5.4")

# ---------------------------------------------------------------------------
# Provider-specific output-token limits
# ---------------------------------------------------------------------------
AI_MAX_TOKENS_DEEPSEEK = 8192   # deepseek-chat API maximum
AI_MAX_TOKENS_GEMINI   = 8192
AI_MAX_TOKENS_OPENAI   = 32768  # gpt-4.1 API maximum
AI_MAX_TOKENS_AZURE    = 65536  # Conservative ceiling for gpt-5.4 (128K maximum)

PROVIDER_MAX_TOKENS = {
    "deepseek": AI_MAX_TOKENS_DEEPSEEK,
    "gemini":   AI_MAX_TOKENS_GEMINI,
    "openai":   AI_MAX_TOKENS_OPENAI,
    "azure":    AI_MAX_TOKENS_AZURE,
}

PROVIDER_MAX_TOKENS_ENV = {
    "deepseek": "DEEPSEEK_MAX_OUTPUT_TOKENS",
    "gemini": "GEMINI_MAX_OUTPUT_TOKENS",
    "openai": "OPENAI_MAX_OUTPUT_TOKENS",
    "azure": "AZURE_MAX_OUTPUT_TOKENS",
}


class CompletionText(str):
    """String-compatible AI output carrying completion metadata."""

    def __new__(cls, value, status="complete", reason=""):
        obj = super().__new__(cls, value or "")
        obj.completion_status = status
        obj.completion_reason = reason
        return obj


def _configured_provider_max_tokens(provider, hard_limit):
    env_name = PROVIDER_MAX_TOKENS_ENV[provider]
    raw_value = os.getenv(env_name, "").strip()
    if not raw_value:
        return hard_limit
    try:
        configured = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{env_name} must be a positive integer") from exc
    if configured <= 0:
        raise ValueError(f"{env_name} must be a positive integer")
    return min(configured, hard_limit)


def get_provider_max_tokens(provider):
    """Return the configured output-token budget for a supported provider."""
    try:
        hard_limit = PROVIDER_MAX_TOKENS[provider]
    except KeyError as exc:
        raise ValueError(f"Unsupported AI provider: {provider}") from exc
    return _configured_provider_max_tokens(provider, hard_limit)

# ---------------------------------------------------------------------------
# Gemini rate-limiting state
# ---------------------------------------------------------------------------
_gemini_call_count = 0
_gemini_cooldown_until = 0.0
_gemini_rate_lock = threading.Lock()
GEMINI_RATE_LIMIT_CALLS = 10
GEMINI_RATE_LIMIT_SLEEP = 30


# ---------------------------------------------------------------------------
# Unified AI Client
# ---------------------------------------------------------------------------
class UnifiedAIClient:
    """Unified interface for different AI providers."""

    def __init__(self, provider="deepseek"):
        self.provider = provider

        if provider == "deepseek":
            from openai import OpenAI
            if not DEEPSEEK_API_KEY:
                raise ValueError("DEEPSEEK_API_TOKEN environment variable must be set")
            self.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
            self.model = DEEPSEEK_MODEL_NAME
            self.max_tokens = get_provider_max_tokens(provider)

        elif provider == "gemini":
            if not GEMINI_AVAILABLE:
                raise ImportError(
                    "Neither google-genai nor google-generativeai is installed. "
                    "Run: pip install google-genai  (or)  pip install google-generativeai"
                )
            if not GEMINI_API_KEY:
                raise ValueError("GEMINI_API_TOKEN environment variable must be set")
            if _GEMINI_NEW_SDK:
                self.client = genai.Client(api_key=GEMINI_API_KEY)
            else:
                genai.configure(api_key=GEMINI_API_KEY)
                self.client = None
            self.model = GEMINI_MODEL_NAME
            self.max_tokens = get_provider_max_tokens(provider)

        elif provider == "openai":
            from openai import OpenAI
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_TOKEN environment variable must be set")
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.model = OPENAI_MODEL_NAME
            self.max_tokens = get_provider_max_tokens(provider)

        elif provider == "azure":
            from openai import OpenAI
            if not AZURE_OPENAI_KEY:
                raise ValueError("AZURE_OPENAI_KEY environment variable must be set")
            if not AZURE_OPENAI_BASE_URL:
                raise ValueError("OPENAI_BASE_URL environment variable must be set")
            self.client = OpenAI(api_key=AZURE_OPENAI_KEY, base_url=AZURE_OPENAI_BASE_URL)
            self.model = AZURE_OPENAI_MODEL_NAME
            self.max_tokens = get_provider_max_tokens(provider)

        else:
            raise ValueError(f"Unsupported AI provider: {provider}")

    # -----------------------------------------------------------------------
    def chat_completion(self, messages, temperature=0.1, max_tokens=None):
        """Unified chat completion interface."""
        if max_tokens is None:
            max_tokens = self.max_tokens
        max_tokens = min(max_tokens, self.max_tokens)

        if self.provider in ("deepseek", "openai"):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            finish_reason = getattr(response.choices[0], "finish_reason", None)
            if finish_reason == "length":
                thread_safe_print(
                    f"   ⚠️  AI response truncated (finish_reason='length'). "
                    f"Output hit the {max_tokens}-token limit. "
                    f"Consider reducing batch size."
                )
            status = "incomplete" if finish_reason == "length" else "complete"
            reason = "max_output_tokens" if finish_reason == "length" else ""
            return CompletionText(
                response.choices[0].message.content.strip(),
                status=status,
                reason=reason,
            )

        elif self.provider == "azure":
            return self._azure_chat(messages, temperature, max_tokens)

        elif self.provider == "gemini":
            return self._gemini_chat(messages, temperature, max_tokens)

    # -----------------------------------------------------------------------
    def _azure_chat(self, messages, temperature, max_tokens):
        """Azure OpenAI via the Responses API (client.responses.create)."""
        instructions = None
        input_parts = []
        for msg in messages:
            if msg.get("role") == "system":
                instructions = msg.get("content", "")
            else:
                input_parts.append(msg)

        input_value = input_parts if input_parts else messages

        kwargs = dict(
            model=self.model,
            input=input_value,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if instructions:
            kwargs["instructions"] = instructions

        response = self.client.responses.create(**kwargs)
        status = getattr(response, "status", None)
        reason = ""
        if status == "incomplete":
            incomplete_details = getattr(response, "incomplete_details", None)
            reason = getattr(incomplete_details, "reason", "unknown") if incomplete_details else "unknown"
            thread_safe_print(
                f"   ⚠️  Azure AI response incomplete (reason='{reason}'). "
                f"Output may have hit the {max_tokens}-token limit. "
                f"Consider reducing batch size."
            )
        output_text = getattr(response, "output_text", None)
        if not output_text:
            raise RuntimeError(
                f"Azure AI response did not include output text (status='{status or 'unknown'}')"
            )
        return CompletionText(
            output_text.strip(),
            status="incomplete" if status == "incomplete" else "complete",
            reason=reason,
        )

    # -----------------------------------------------------------------------
    def _gemini_chat(self, messages, temperature, max_tokens):
        global _gemini_call_count, _gemini_cooldown_until
        try:
            while True:
                with _gemini_rate_lock:
                    now = time.monotonic()
                    if _gemini_cooldown_until > now:
                        wait_seconds = _gemini_cooldown_until - now
                        call_count = None
                    else:
                        _gemini_call_count += 1
                        if _gemini_call_count > GEMINI_RATE_LIMIT_CALLS:
                            _gemini_call_count = 1
                            _gemini_cooldown_until = now + GEMINI_RATE_LIMIT_SLEEP
                            wait_seconds = GEMINI_RATE_LIMIT_SLEEP
                        else:
                            wait_seconds = 0
                        call_count = _gemini_call_count

                if wait_seconds:
                    thread_safe_print(
                        f"   ⏳ Rate limit: {GEMINI_RATE_LIMIT_CALLS} calls reached, "
                        f"sleeping {wait_seconds:.1f}s..."
                    )
                    time.sleep(wait_seconds)
                    if call_count is None:
                        continue

                break

            prompt = self._convert_messages_to_prompt(messages)
            thread_safe_print(
                f"   🔄 Calling Gemini API ({call_count}/{GEMINI_RATE_LIMIT_CALLS})..."
            )

            if _GEMINI_NEW_SDK:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                    },
                )
            else:
                model = genai.GenerativeModel(self.model)
                response = model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                    },
                )

            if response and response.text:
                thread_safe_print("   ✅ Gemini response received")
                finish_reason = ""
                candidates = getattr(response, "candidates", None) or []
                if candidates:
                    finish_reason = str(getattr(candidates[0], "finish_reason", "") or "")
                normalized_reason = finish_reason.lower()
                incomplete = "max_token" in normalized_reason or normalized_reason.endswith("length")
                if incomplete:
                    thread_safe_print(
                        f"   ⚠️  Gemini AI response incomplete (reason='{finish_reason}')."
                    )
                return CompletionText(
                    response.text.strip(),
                    status="incomplete" if incomplete else "complete",
                    reason=finish_reason or ("max_output_tokens" if incomplete else ""),
                )

        except Exception as e:
            thread_safe_print(f"   ❌ Gemini API error: {sanitize_exception_message(e)}")
            thread_safe_print(
                "   💡 Consider switching to DeepSeek: AI_PROVIDER = 'deepseek'"
            )
            raise

        thread_safe_print("   ⚠️  Gemini response was empty or blocked")
        raise RuntimeError("Gemini returned an empty or blocked response")

    # -----------------------------------------------------------------------
    @staticmethod
    def _convert_messages_to_prompt(messages):
        """Convert OpenAI-style messages to a single prompt for Gemini."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            else:
                parts.append(content)
        return "\n\n".join(parts)
