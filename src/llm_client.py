"""
Unified LLM client abstraction for AgenticSimLaw debate experiments.

Dual-path design:
- OllamaClient: Native ollama.chat() preserving all 6 nanosecond timing fields
- LiteLLMClient: litellm.acompletion() for commercial APIs (OpenAI, Google, xAI)

Both return a unified LLMResponse that produces metadata dicts backward-compatible
with ver25's MetaData.model_dump() output consumed by step2 aggregation.
"""

import asyncio
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; API keys must be set in environment

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Normalized response from any LLM provider."""
    content: str
    model: str
    provider: str  # "ollama" or "litellm"
    python_api_duration_sec: float

    # Token counts
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    # Ollama-specific nanosecond fields (None for commercial)
    total_duration_ns: Optional[float] = None
    load_duration_ns: Optional[float] = None
    prompt_eval_duration_ns: Optional[float] = None
    eval_duration_ns: Optional[float] = None
    created_at: Optional[str] = None
    done: Optional[bool] = None

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Produce a dict backward-compatible with ver25 MetaData.model_dump().

        Step2 aggregation reads: total_duration_ns, prompt_eval_count, eval_count,
        total_duration_sec, python_api_duration_sec per speaker.
        """
        def ns_to_s(val):
            return val / 1e9 if val is not None else None

        if self.provider == "ollama":
            return {
                "model": self.model,
                "created_at": self.created_at,
                "done": self.done,
                "total_duration_ns": self.total_duration_ns,
                "load_duration_ns": self.load_duration_ns,
                "prompt_eval_count": self.prompt_tokens,
                "prompt_eval_duration_ns": self.prompt_eval_duration_ns,
                "eval_count": self.completion_tokens,
                "eval_duration_ns": self.eval_duration_ns,
                "total_duration_sec": ns_to_s(self.total_duration_ns),
                "load_duration_sec": ns_to_s(self.load_duration_ns),
                "prompt_eval_duration_sec": ns_to_s(self.prompt_eval_duration_ns),
                "eval_duration_sec": ns_to_s(self.eval_duration_ns),
                "python_api_duration_sec": self.python_api_duration_sec,
            }
        else:
            # Commercial API — populate compatible keys with available data
            return {
                "model": self.model,
                "created_at": None,
                "done": True,
                "total_duration_ns": None,
                "load_duration_ns": None,
                "prompt_eval_count": self.prompt_tokens,
                "prompt_eval_duration_ns": None,
                "eval_count": self.completion_tokens,
                "eval_duration_ns": None,
                "total_duration_sec": self.python_api_duration_sec,
                "load_duration_sec": None,
                "prompt_eval_duration_sec": None,
                "eval_duration_sec": None,
                "python_api_duration_sec": self.python_api_duration_sec,
            }


class OllamaClient:
    """Native Ollama client preserving all timing metadata."""

    MAX_RETRIES = 3
    BACKOFF_BASE = 2.0
    BACKOFF_MAX = 30.0
    JITTER_MAX = 1.0

    def __init__(self, model_name: str, temperature: float = 0.7,
                 max_tokens: int = 1024, timeout: float = 900.0):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    async def chat(self, messages: List[Dict[str, str]],
                   model: Optional[str] = None) -> LLMResponse:
        """Call ollama.chat() with asyncio.to_thread + wait_for + retry."""
        from ollama import chat as ollama_chat

        effective_model = model or self.model_name
        last_err = None

        for attempt in range(self.MAX_RETRIES):
            try:
                start_time = time.time()

                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        ollama_chat,
                        messages=messages,
                        model=effective_model,
                        options={
                            "temperature": self.temperature,
                            "max_tokens": self.max_tokens,
                        },
                    ),
                    timeout=self.timeout,
                )

                duration = time.time() - start_time

                return LLMResponse(
                    content=response.message.content,
                    model=effective_model,
                    provider="ollama",
                    python_api_duration_sec=duration,
                    prompt_tokens=getattr(response, "prompt_eval_count", None),
                    completion_tokens=getattr(response, "eval_count", None),
                    total_duration_ns=getattr(response, "total_duration", None),
                    load_duration_ns=getattr(response, "load_duration", None),
                    prompt_eval_duration_ns=getattr(response, "prompt_eval_duration", None),
                    eval_duration_ns=getattr(response, "eval_duration", None),
                    created_at=getattr(response, "created_at", None),
                    done=getattr(response, "done", None),
                )

            except Exception as e:
                last_err = e
                if attempt < self.MAX_RETRIES - 1:
                    backoff = min(self.BACKOFF_BASE ** (attempt + 1), self.BACKOFF_MAX)
                    jitter = random.uniform(0, self.JITTER_MAX)
                    wait = backoff + jitter
                    logger.warning(
                        f"Retry {attempt + 1}/{self.MAX_RETRIES} for {effective_model} "
                        f"(wait {wait:.1f}s): {e}"
                    )
                    await asyncio.sleep(wait)

        raise RuntimeError(f"Failed after {self.MAX_RETRIES} retries for {effective_model}: {last_err}")


class LiteLLMClient:
    """LiteLLM client for commercial APIs with retry + exponential backoff."""

    MAX_RETRIES = 5
    BACKOFF_BASE = 2.0
    BACKOFF_MAX = 60.0
    JITTER_MAX = 2.0
    # Longer backoff for rate-limit errors (free-tier models)
    RATE_LIMIT_BACKOFF_BASE = 10.0
    RATE_LIMIT_BACKOFF_MAX = 120.0

    # OpenAI reasoning/thinking models that reject temperature and max_tokens
    THINKING_MODELS = frozenset({
        "gpt-5-mini", "gpt-5", "gpt-5.1", "gpt-5.2", "gpt-5-nano",
        "gpt-5-pro", "gpt-5.1-pro", "gpt-5.2-pro",
        "o3", "o3-pro", "o4-mini", "o1", "o1-pro",
    })
    # Thinking models need higher token budget: max_completion_tokens covers
    # BOTH reasoning tokens and output tokens. 1024 is far too low.
    THINKING_MODEL_TOKEN_MULTIPLIER = 16

    def __init__(self, model_name: str, temperature: float = 0.7,
                 max_tokens: int = 1024, timeout: float = 900.0,
                 reasoning_effort: Optional[str] = None):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort

    @classmethod
    def _is_thinking_model(cls, model_name: str) -> bool:
        """Check if model is a thinking/reasoning model that rejects sampling params."""
        return model_name in cls.THINKING_MODELS

    @staticmethod
    def _is_rate_limit(exc: Exception) -> bool:
        """Check if exception is a rate-limit error."""
        exc_name = type(exc).__name__
        if "RateLimit" in exc_name:
            return True
        err_str = str(exc).lower()
        return "rate_limit" in err_str or "rate limit" in err_str or "429" in err_str

    async def chat(self, messages: List[Dict[str, str]],
                   model: Optional[str] = None) -> LLMResponse:
        """Call litellm.acompletion() with retry + exponential backoff."""
        import litellm

        effective_model = model or self.model_name
        last_err = None

        # Build kwargs based on model type
        if self._is_thinking_model(effective_model):
            # max_completion_tokens covers BOTH reasoning + output tokens.
            # Use a multiplier so the model has room for visible output.
            thinking_budget = self.max_tokens * self.THINKING_MODEL_TOKEN_MULTIPLIER
            call_kwargs = {
                "model": effective_model,
                "messages": messages,
                "max_completion_tokens": thinking_budget,
            }
            if self.reasoning_effort:
                call_kwargs["reasoning_effort"] = self.reasoning_effort
        else:
            call_kwargs = {
                "model": effective_model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }

        for attempt in range(self.MAX_RETRIES):
            try:
                start_time = time.time()
                response = await asyncio.wait_for(
                    litellm.acompletion(**call_kwargs),
                    timeout=self.timeout,
                )
                duration = time.time() - start_time

                msg = response.choices[0].message
                content = msg.content or ""
                # Reasoning models may put output in reasoning_content
                if not content:
                    reasoning = getattr(msg, "reasoning_content", None)
                    if reasoning:
                        content = reasoning
                        logger.debug(
                            f"Using reasoning_content as content for {effective_model} "
                            f"({len(reasoning)} chars)"
                        )
                usage = getattr(response, "usage", None)

                return LLMResponse(
                    content=content,
                    model=effective_model,
                    provider="litellm",
                    python_api_duration_sec=duration,
                    prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
                    completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
                )

            except Exception as e:
                last_err = e
                if attempt < self.MAX_RETRIES - 1:
                    if self._is_rate_limit(e):
                        backoff = min(
                            self.RATE_LIMIT_BACKOFF_BASE * (attempt + 1),
                            self.RATE_LIMIT_BACKOFF_MAX,
                        )
                    else:
                        backoff = min(self.BACKOFF_BASE ** (attempt + 1), self.BACKOFF_MAX)
                    jitter = random.uniform(0, self.JITTER_MAX)
                    wait = backoff + jitter
                    logger.warning(
                        f"Retry {attempt + 1}/{self.MAX_RETRIES} for {effective_model} "
                        f"(wait {wait:.1f}s, rate_limit={self._is_rate_limit(e)}): {e}"
                    )
                    await asyncio.sleep(wait)

        raise RuntimeError(f"Failed after {self.MAX_RETRIES} retries for {effective_model}: {last_err}")


def _is_ollama_model(model_name: str) -> bool:
    """Detect whether a model name is an Ollama local model.

    Ollama models look like 'gemma2:9b-instruct-q4_K_M' — they contain ':'
    but never contain '/'.  Cloud-routed models (OpenRouter, Fireworks, etc.)
    always contain '/' in their provider prefix.
    """
    if "/" in model_name:
        return False
    if ":" in model_name:
        return True
    return False


def create_client(model_name: str, temperature: float = 0.7,
                  max_tokens: int = 1024, timeout: float = 900.0,
                  reasoning_effort: Optional[str] = None):
    """Factory: returns OllamaClient for local Ollama models, else LiteLLMClient."""
    if _is_ollama_model(model_name):
        return OllamaClient(model_name, temperature, max_tokens, timeout)
    else:
        return LiteLLMClient(model_name, temperature, max_tokens, timeout,
                             reasoning_effort=reasoning_effort)
