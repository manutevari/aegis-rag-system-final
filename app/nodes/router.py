from app.core.models import get_llm
from app.core.model_guard import get_llm_model

def route_intent(query: str) -> dict:
# use nano for cheap routing
llm = get_llm(model_override="gpt-5-nano")
prompt = f"Classify: factual | lookup | compute | unclear\nQuery: {query}\nLabel:"
out = llm.invoke(prompt).content.lower()
label = ("compute" if "compute" in out else
"lookup" if "lookup" in out else
"factual" if "factual" in out else
"unclear")
return {"intent": label}
