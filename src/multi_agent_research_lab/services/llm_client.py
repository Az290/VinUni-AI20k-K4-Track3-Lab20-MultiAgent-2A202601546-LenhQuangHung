"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

import logging
from dataclasses import dataclass

from openai import APIError, APITimeoutError, OpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError

logger = logging.getLogger(__name__)

# Rough per-1M-token pricing for gpt-4o-mini class models (USD). Used only for benchmark
# cost estimation, not for billing. Update if the configured model changes materially.
_PRICE_PER_1M_INPUT_TOKENS_USD = 0.15
_PRICE_PER_1M_OUTPUT_TOKENS_USD = 0.60


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


def _estimate_cost(input_tokens: int | None, output_tokens: int | None) -> float | None:
    if input_tokens is None or output_tokens is None:
        return None
    return (
        input_tokens / 1_000_000 * _PRICE_PER_1M_INPUT_TOKENS_USD
        + output_tokens / 1_000_000 * _PRICE_PER_1M_OUTPUT_TOKENS_USD
    )


class LLMClient:
    """Provider-agnostic LLM client. Backed by OpenAI's chat completions API."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        if not self._settings.openai_api_key:
            raise AgentExecutionError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and fill in your key."
            )
        self._client = OpenAI(
            api_key=self._settings.openai_api_key,
            timeout=self._settings.timeout_seconds,
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=1, max=10),
        retry=retry_if_exception_type((APITimeoutError, RateLimitError, APIError)),
    )
    def _call(self, system_prompt: str, user_prompt: str, temperature: float) -> LLMResponse:
        completion = self._client.chat.completions.create(
            model=self._settings.openai_model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        choice = completion.choices[0]
        content = choice.message.content or ""
        usage = completion.usage
        input_tokens = usage.prompt_tokens if usage else None
        output_tokens = usage.completion_tokens if usage else None
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=_estimate_cost(input_tokens, output_tokens),
        )

    def complete(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.2
    ) -> LLMResponse:
        """Return a model completion.

        Retries transient failures (timeout, rate limit, provider error) with exponential
        backoff. Raises `AgentExecutionError` once retries are exhausted so callers can apply
        their own fallback policy instead of crashing the whole workflow.
        """

        try:
            return self._call(system_prompt, user_prompt, temperature)
        except (APITimeoutError, RateLimitError, APIError) as exc:
            logger.error("LLM call failed after retries: %s", exc)
            raise AgentExecutionError(f"LLM call failed after retries: {exc}") from exc
