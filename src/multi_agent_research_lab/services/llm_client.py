"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

import logging
from dataclasses import dataclass

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client using OpenAI."""

    # Pricing per 1M tokens (approximate, update as needed)
    _PRICING: dict[str, tuple[float, float]] = {
        "gpt-4o": (5.0, 15.0),       # input, output per 1M
        "gpt-4o-mini": (0.15, 0.6),
        "gpt-4-turbo": (10.0, 30.0),
        "gpt-3.5-turbo": (0.5, 1.5),
    }

    def __init__(self) -> None:
        settings = get_settings()
        api_key = settings.openai_api_key
        if not api_key or api_key.startswith("sk-your"):
            raise ValueError(
                "OPENAI_API_KEY not set. Please set a valid API key in .env file."
            )
        self._client = OpenAI(api_key=api_key)
        self._model = settings.openai_model
        logger.info(f"LLMClient initialized with model: {self._model}")

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD based on model pricing."""
        model_key = self._model.lower()
        for key, (in_price, out_price) in self._PRICING.items():
            if key in model_key:
                return (input_tokens / 1_000_000) * in_price + \
                       (output_tokens / 1_000_000) * out_price
        # Default fallback: assume gpt-4o-mini pricing
        in_price, out_price = self._PRICING["gpt-4o-mini"]
        return (input_tokens / 1_000_000) * in_price + \
               (output_tokens / 1_000_000) * out_price

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion with retry logic and cost tracking.

        Args:
            system_prompt: System instructions for the model.
            user_prompt: User query or task description.

        Returns:
            LLMResponse with content and optional token/cost metadata.
        """
        logger.debug(f"Calling LLM with model={self._model}")
        logger.debug(f"System prompt length: {len(system_prompt)} chars")
        logger.debug(f"User prompt length: {len(user_prompt)} chars")

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=4096,
        )

        # Extract response content
        content = response.choices[0].message.content or ""

        # Extract token usage
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else None
        output_tokens = usage.completion_tokens if usage else None

        # Calculate estimated cost
        cost_usd = None
        if input_tokens is not None and output_tokens is not None:
            cost_usd = self._estimate_cost(input_tokens, output_tokens)
            logger.info(
                f"LLM response: {output_tokens} output tokens, "
                f"est. cost: ${cost_usd:.6f}"
            )

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )
