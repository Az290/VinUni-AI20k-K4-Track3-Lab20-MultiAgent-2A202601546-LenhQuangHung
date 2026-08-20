"""Tracing hooks.

This file intentionally avoids binding to one provider. When `LANGSMITH_API_KEY` is set,
spans are additionally reported through the LangSmith client. When `LANGFUSE_PUBLIC_KEY`/
`LANGFUSE_SECRET_KEY` are set, spans are also sent to Langfuse. Either, both, or neither can
be configured. Regardless, every span is appended to a local JSON trace file under
`reports/traces/` so a run's trace is always inspectable, even fully offline.

Nesting: `trace_span` calls made while inside another `trace_span` become *children* of the
outer one, both in the exported providers (via OpenTelemetry/Langfuse context propagation and
LangSmith parent-run tracking) and in the local JSON export. Wrap a full workflow run in one
outer `trace_span("multi_agent_run", ...)` so the whole Supervisor -> Researcher -> Analyst ->
Writer -> Critic sequence shows up as a single tree instead of unrelated flat spans.
"""

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

from multi_agent_research_lab.core.config import get_settings

if TYPE_CHECKING:
    from langfuse import Langfuse

logger = logging.getLogger(__name__)

_TRACE_DIR = Path("reports") / "traces"

# Tracks the currently active LangSmith run id, so nested `trace_span` calls can be recorded
# as children of it instead of unrelated top-level runs.
_current_langsmith_run_id: ContextVar[str | None] = ContextVar(
    "current_langsmith_run_id", default=None
)


def _langsmith_enabled() -> bool:
    return bool(get_settings().langsmith_api_key)


def _langfuse_enabled() -> bool:
    settings = get_settings()
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key)


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Minimal span context used by the skeleton.

    Always records locally; additionally forwards to LangSmith/Langfuse when configured.
    Nests under the currently active span (if any) in all three sinks.
    """

    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}

    langfuse_cm = _open_langfuse_span(name, span) if _langfuse_enabled() else None
    langfuse_obs = langfuse_cm.__enter__() if langfuse_cm is not None else None

    langsmith_run_id = _start_langsmith_run(name, span) if _langsmith_enabled() else None
    langsmith_token = _current_langsmith_run_id.set(
        langsmith_run_id or _current_langsmith_run_id.get()
    )

    try:
        yield span
    finally:
        span["duration_seconds"] = perf_counter() - started
        _record_local(span)

        if langfuse_obs is not None:
            langfuse_obs.update(output={"duration_seconds": span["duration_seconds"]})
        if langfuse_cm is not None:
            langfuse_cm.__exit__(None, None, None)
            _flush_langfuse()

        if langsmith_run_id is not None:
            _end_langsmith_run(langsmith_run_id, span)
        _current_langsmith_run_id.reset(langsmith_token)


def _record_local(span: dict[str, Any]) -> None:
    try:
        _TRACE_DIR.mkdir(parents=True, exist_ok=True)
        entry = {**span, "timestamp": datetime.now(UTC).isoformat()}
        log_path = _TRACE_DIR / "local_trace.jsonl"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except OSError as exc:  # pragma: no cover - best-effort tracing, never fatal
        logger.warning("Failed to write local trace: %s", exc)


def _start_langsmith_run(name: str, span: dict[str, Any]) -> str | None:
    try:
        import uuid

        from langsmith import Client

        settings = get_settings()
        run_id = str(uuid.uuid4())
        client = Client(api_key=settings.langsmith_api_key)
        client.create_run(
            id=run_id,
            name=name,
            run_type="chain",
            inputs=span.get("attributes", {}),
            parent_run_id=_current_langsmith_run_id.get(),
            project_name=settings.langsmith_project,
        )
        return run_id
    except Exception as exc:  # pragma: no cover - tracing must never break the workflow
        logger.warning("LangSmith run start failed: %s", exc)
        return None


def _end_langsmith_run(run_id: str, span: dict[str, Any]) -> None:
    try:
        from langsmith import Client

        settings = get_settings()
        client = Client(api_key=settings.langsmith_api_key)
        client.update_run(run_id, outputs={"duration_seconds": span.get("duration_seconds")})
    except Exception as exc:  # pragma: no cover - tracing must never break the workflow
        logger.warning("LangSmith run end failed: %s", exc)


@lru_cache(maxsize=1)
def _langfuse_client() -> "Langfuse":
    """One process-wide Langfuse client, built from `Settings` (not raw env vars), so nested
    `trace_span` calls share the same OTel tracer and nest correctly instead of each creating
    an unrelated client/trace.
    """

    from langfuse import Langfuse

    settings = get_settings()
    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )


def _open_langfuse_span(name: str, span: dict[str, Any]) -> Any:
    try:
        client = _langfuse_client()
        return client.start_as_current_observation(
            name=name,
            as_type="span",
            input=span.get("attributes", {}),
        )
    except Exception as exc:  # pragma: no cover - tracing must never break the workflow
        logger.warning("Langfuse span start failed: %s", exc)
        return None


def _flush_langfuse() -> None:
    try:
        _langfuse_client().flush()
    except Exception as exc:  # pragma: no cover - tracing must never break the workflow
        logger.warning("Langfuse flush failed: %s", exc)


def export_state_trace(run_name: str, trace: list[dict[str, Any]]) -> Path:
    """Write a full run's trace (from `ResearchState.trace`) to a single JSON file."""

    _TRACE_DIR.mkdir(parents=True, exist_ok=True)
    path = _TRACE_DIR / f"{run_name}.json"
    path.write_text(json.dumps(trace, indent=2, default=str), encoding="utf-8")
    return path
