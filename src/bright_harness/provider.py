"""Model providers in one place, so the task agent is configured identically
everywhere it is used (preflight, round driver, fitness probe, cross-domain eval).

Two roles, deliberately separated (HarnessX's AEGIS design):

* **task agent** — Qwen3.5-9B served by vLLM on GAAS, frozen for the whole
  experiment. Reached through an SSH tunnel at ``BRIGHT_TASK_BASE_URL``.
* **proposer** — DeepSeek-V4-Flash via API, drives Digester/Planner/Evolver/Critic.

Thinking mode: Qwen3.5 emits a long `` thinking`` trace by default — measured at
~2100 tokens and >11s even for "reply with OK", which would dominate wall-clock
across ~100 queries x several calls x many rounds. It is therefore **off** by
default and set per request via ``extra_body.chat_template_kwargs`` (the OpenAI
SDK's supported channel for non-standard parameters; the vLLM server reads it and
applies it to the chat template).

Turning thinking back on is a legitimate ``config``-bucket evolution edit, so it
stays a one-env-var switch rather than a serving-side decision.
"""
from __future__ import annotations

import os

from harnessx.core.model_config import ModelConfig
from harnessx.providers.openai_provider import OpenAIProvider

__all__ = [
    "truthy",
    "make_task_provider",
    "make_task_model_config",
    "make_proposer_provider",
    "make_proposer_model_config",
    "task_provider_summary",
]


def truthy(name: str, default: str = "0") -> bool:
    return (os.environ.get(name) or default).strip().lower() in ("1", "true", "yes", "on")


def _thinking_enabled() -> bool:
    return truthy("BRIGHT_ENABLE_THINKING", "0")


def make_task_provider() -> OpenAIProvider:
    extra: dict = {}
    if not _thinking_enabled():
        extra["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    return OpenAIProvider(
        model=os.environ.get("BRIGHT_TASK_MODEL") or "Qwen3.5-9B",
        base_url=os.environ.get("BRIGHT_TASK_BASE_URL") or "http://127.0.0.1:8000/v1",
        api_key=os.environ.get("BRIGHT_TASK_API_KEY") or "EMPTY",
        max_tokens=int(os.environ.get("BRIGHT_TASK_MAX_TOKENS") or 4096),
        temperature=float(os.environ.get("BRIGHT_TASK_TEMPERATURE") or 0),
        **extra,
    )


def make_task_model_config() -> ModelConfig:
    return ModelConfig(main=make_task_provider())


def make_proposer_provider() -> OpenAIProvider:
    return OpenAIProvider(
        model=os.environ.get("BRIGHT_PROPOSER_MODEL") or "deepseek-v4-flash",
        base_url=os.environ.get("BRIGHT_PROPOSER_BASE_URL") or "https://api.deepseek.com/v1",
        api_key=os.environ.get("BRIGHT_PROPOSER_API_KEY") or "",
        max_tokens=int(os.environ.get("BRIGHT_PROPOSER_MAX_TOKENS") or 8192),
        # A low default keeps the Flash-class proposer focused on shipping ONE
        # candidate instead of rambling. At 0.7 it rabbit-holed: 200 steps, $41,
        # pyarrow downloaded, zero manifests. This is a model-sampling knob, not
        # part of AEGIS's logic; override with BRIGHT_PROPOSER_TEMPERATURE.
        temperature=float(os.environ.get("BRIGHT_PROPOSER_TEMPERATURE") or 0.2),
    )


def make_proposer_model_config() -> ModelConfig:
    return ModelConfig(main=make_proposer_provider())


def task_provider_summary() -> dict:
    """Non-secret description of the task-agent configuration, for run metadata."""
    return {
        "base_url": os.environ.get("BRIGHT_TASK_BASE_URL"),
        "model": os.environ.get("BRIGHT_TASK_MODEL"),
        "max_tokens": int(os.environ.get("BRIGHT_TASK_MAX_TOKENS") or 4096),
        "temperature": float(os.environ.get("BRIGHT_TASK_TEMPERATURE") or 0),
        "thinking_enabled": _thinking_enabled(),
    }
