"""Optional middleware hooks for local-only execution."""

from typing import Callable

from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call


@wrap_model_call
def orchestrator_middleware(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Pass requests through without swapping in hosted models."""
    return handler(request)
