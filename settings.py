import json
import os
from pathlib import Path


def load_local_env() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export "):].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key or key in os.environ:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        value = os.path.expandvars(value)
        os.environ[key] = value


load_local_env()


def env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in {None, ""} else default


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value in {None, ""}:
        return default
    return int(value)


API_KEY = env_str("LLM_API_KEY", env_str("OPENAI_API_KEY", env_str("OPENROUTER_API_KEY", "sk-no-key")))
MODEL = env_str("LLM_MODEL", "ggpl")
PARALLEL = env_int("LLM_PARALLEL", 8)

BASE_URL = env_str("LLM_BASE_URL", "")
BASE_HOST = env_str("LLM_BASE_HOST", "127.0.0.1")
BASE_PORT = env_int("LLM_BASE_PORT", 8080)
MAX_PORTS_TO_SCAN = env_int("LLM_MAX_PORTS_TO_SCAN", 8)
TOP_P = float(env_str("LLM_TOP_P", "0"))
TEMPERATURE = float(env_str("LLM_TEMPERATURE", "0"))
EXTRA_BODY_JSON = env_str("LLM_EXTRA_BODY_JSON", "")
TOKEN_LIMIT_PARAM = env_str("LLM_TOKEN_LIMIT_PARAM", "")
REQUEST_TIMEOUT = float(env_str("LLM_REQUEST_TIMEOUT", "10"))
MAX_RETRIES = env_int("LLM_MAX_RETRIES", 2)
RETRY_BACKOFF_SECONDS = float(env_str("LLM_RETRY_BACKOFF_SECONDS", "2"))
REASONING_EFFORT = env_str("LLM_REASONING_EFFORT", "")
LABEL_MAX_TOKENS = env_int("LLM_LABEL_MAX_TOKENS", 32)
FOCAL_MAX_TOKENS = env_int("LLM_FOCAL_MAX_TOKENS", 8)
PROMPT_CACHE_KEY_PREFIX = env_str("LLM_PROMPT_CACHE_KEY_PREFIX", "")
PROMPT_CACHE_RETENTION = env_str("LLM_PROMPT_CACHE_RETENTION", "")


def openai_base_url(port: int | None = None) -> str:
    if BASE_URL:
        return BASE_URL
    chosen_port = BASE_PORT if port is None else port
    return f"http://{BASE_HOST}:{chosen_port}/v1"


def chat_completion_kwargs() -> dict:
    kwargs = {
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
    }
    if REASONING_EFFORT and "api.openai.com" in BASE_URL:
        kwargs["reasoning_effort"] = REASONING_EFFORT
    if EXTRA_BODY_JSON:
        kwargs["extra_body"] = json.loads(EXTRA_BODY_JSON)
    return kwargs


def token_limit_param_name() -> str:
    if TOKEN_LIMIT_PARAM:
        return TOKEN_LIMIT_PARAM
    if "api.openai.com" in BASE_URL:
        return "max_completion_tokens"
    return "max_tokens"


def prompt_cache_kwargs(cache_key_suffix: str | None = None) -> dict:
    if "api.openai.com" not in BASE_URL:
        return {}

    kwargs = {}
    if PROMPT_CACHE_KEY_PREFIX:
        parts = [PROMPT_CACHE_KEY_PREFIX, MODEL]
        if cache_key_suffix:
            parts.append(cache_key_suffix)
        kwargs["prompt_cache_key"] = ":".join(parts)
    if PROMPT_CACHE_RETENTION:
        kwargs["prompt_cache_retention"] = PROMPT_CACHE_RETENTION
    return kwargs
