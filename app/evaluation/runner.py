import time
from app.evaluation.metrics import exact_match, recall_at_k

def run_pipeline(graph, question: str):
    start = time.time()
    result = graph.invoke({"question": question})
    latency = time.time() - start

    return {
        "answer": result.get("answer", ""),
        "contexts": result.get("retrieved_docs", []),
        "latency": latency,
        "tokens": result.get("tokens_used", 0)
    }

def evaluate(graph, dataset, cost_tracker):
    outputs = []

    for item in dataset:
        res = run_pipeline(graph, item["question"])

        cost_tracker.log(
            tokens=res["tokens"],
            cost=res["tokens"] * 0.000002,
            latency=res["latency"]
        )

        outputs.append({
            "question": item["question"],
            "answer": res["answer"],
            "contexts": res["contexts"],
            "ground_truth": item["ground_truth"],
            "source": item["source"],
            "em": exact_match(res["answer"], item["ground_truth"]),
            "recall": recall_at_k(res["contexts"], item["source"])
        })

    return outputs
