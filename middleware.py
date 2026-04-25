from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from typing import Callable
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 🔹 Models
large_model = init_chat_model(
    model="gpt-5-mini",
    model_provider="openai",
    api_key=OPENAI_API_KEY
)

small_model = init_chat_model(
    model="gpt-5-nano",
    model_provider="openai",
    api_key=OPENAI_API_KEY
)


# ==============================
# 🔹 Middleware Layer
# ==============================

@wrap_model_call
def orchestrator_middleware(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """
    Central decision engine for:
    - Model selection
    - Future tool routing
    - RAG triggering
    """

    messages = request.messages
    last_user_msg = messages[-1].content.lower() if messages else ""

    # --------------------------
    # 🔹 Heuristic Routing Logic
    # --------------------------

    # 1. Numerical / calculation heavy → bigger model
    if any(word in last_user_msg for word in ["calculate", "percentage", "interest", "days", "months"]):
        model = large_model

    # 2. Long conversation → bigger model
    elif len(messages) > 8:
        model = large_model

    # 3. Policy / compliance queries → bigger model (for reasoning)
    elif any(word in last_user_msg for word in ["policy", "compliance", "rule", "clause"]):
        model = large_model

    # 4. Default → small model
    else:
        model = small_model

    # Override model
    request = request.override(model=model)

    # Continue execution
    response = handler(request)

    return response
