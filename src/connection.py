import os
import random
import time
from typing import Any
import openai
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


_FATAL_ERRORS = (
    openai.AuthenticationError,
    openai.BadRequestError,
)


def call_with_retry(fn, *, max_attempts: int = 100, base_delay: float = 1.0, max_delay: float = 120.0):
    """Call fn() up to max_attempts times with exponential backoff + jitter.

    Fatal errors (AuthenticationError, BadRequestError) are re-raised immediately.
    All other exceptions are retried.
    """
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except _FATAL_ERRORS:
            raise
        except Exception as exc:
            last_exc = exc
            if attempt + 1 >= max_attempts:
                break
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = delay * random.uniform(-0.1, 0.1)
            wait = max(0.0, delay + jitter)
            print(
                f"[retry] attempt {attempt + 1}/{max_attempts} failed "
                f"({type(exc).__name__}: {exc}). Retrying in {wait:.1f}s ...",
                flush=True,
            )
            time.sleep(wait)
    raise last_exc


def get_client():
    api_key = os.getenv("POE_API_KEY")
    if not api_key:
        raise ValueError("Missing POE API key (.env: POE_API_KEY).")
    timeout = float(os.getenv("POE_TIMEOUT", "180"))
    return OpenAI(api_key=api_key, base_url="https://api.poe.com/v1", timeout=timeout)


def get_deployment_name() -> str:
    model = os.getenv("POE_MODEL")
    if not model:
        raise ValueError("Missing POE model name (.env: POE_MODEL).")
    return model


def get_chat_completion_text(client, user_content: str | list[dict[str, Any]], system_content: str = "You are a helpful assistant.") -> str:
    model = get_deployment_name()
    _max = int(os.getenv("LLM_MAX_RETRIES", "100"))
    _base = float(os.getenv("LLM_BASE_DELAY", "1.0"))
    _max_d = float(os.getenv("LLM_MAX_DELAY", "120.0"))
    response = call_with_retry(
        lambda: client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            max_completion_tokens=16384,
            model=model,
        ),
        max_attempts=_max,
        base_delay=_base,
        max_delay=_max_d,
    )
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            text = getattr(item, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts)
    return ""


if __name__ == "__main__":
    client = get_client()
    print("Connection successful.")
