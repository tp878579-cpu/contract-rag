# src/generation/llm.py
"""LLM client wrapper. Supports Anthropic, OpenAI, Ollama and Gemini behind one interface."""

import time
from dataclasses import dataclass
from functools import lru_cache

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Providers that speak the OpenAI API (value = base_url, None = OpenAI itself)
_OPENAI_COMPATIBLE = {
    "openai": None,
    "ollama": "http://localhost:11434/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
}
_NO_KEY_NEEDED = {"ollama"}


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    model: str


class LLMClient:
    def __init__(self):
        self.provider = settings.llm_provider.lower()
        self.model = settings.llm_model
        self._client = None

        valid = {"anthropic", *_OPENAI_COMPATIBLE}
        if self.provider not in valid:
            raise ValueError(f"LLM_PROVIDER must be one of {sorted(valid)}, got '{self.provider}'")
        if self.provider not in _NO_KEY_NEEDED and (
            not settings.llm_api_key or settings.llm_api_key.startswith("your-")
        ):
            raise ValueError("LLM_API_KEY is missing. Add your real key to the .env file.")
        if not self.model or self.model.startswith("your-"):
            raise ValueError("LLM_MODEL is missing. Add a model name to the .env file.")

    @property
    def client(self):
        """Create the provider client on first use."""
        if self._client is None:
            if self.provider == "anthropic":
                import anthropic
                self._client = anthropic.Anthropic(api_key=settings.llm_api_key)
            else:
                import openai
                self._client = openai.OpenAI(
                    api_key=settings.llm_api_key or "not-needed",
                    base_url=_OPENAI_COMPATIBLE[self.provider],
                )
        return self._client

    def generate(self, system: str, messages: list[dict]) -> LLMResponse:
        """Send the prompt and return the answer with usage stats."""
        start = time.perf_counter()

        if self.provider == "anthropic":
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
                system=system,
                messages=messages,
            )
            text = "".join(b.text for b in resp.content if b.type == "text")
            in_tok, out_tok = resp.usage.input_tokens, resp.usage.output_tokens
        else:
            # Real OpenAI uses max_completion_tokens; the others use max_tokens
            token_arg = (
                {"max_completion_tokens": settings.max_tokens}
                if self.provider == "openai"
                else {"max_tokens": settings.max_tokens}
            )
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=settings.temperature,
                messages=[{"role": "system", "content": system}, *messages],
                **token_arg,
            )
            text = resp.choices[0].message.content or ""
            usage = resp.usage
            in_tok = getattr(usage, "prompt_tokens", 0) or 0
            out_tok = getattr(usage, "completion_tokens", 0) or 0

        latency = time.perf_counter() - start
        logger.info("LLM call: %d in / %d out tokens, %.2fs", in_tok, out_tok, latency)
        return LLMResponse(
            text=text.strip(),
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_s=latency,
            model=self.model,
        )


@lru_cache(maxsize=1)
def get_llm() -> LLMClient:
    """Shared instance."""
    return LLMClient()