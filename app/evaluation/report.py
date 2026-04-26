def summarize(results, cost_summary, ragas_scores):
    total = len(results)

    em = sum(r["em"] for r in results) / total
    recall = sum(r["recall"] for r in results) / total

    return {
        "exact_match": round(em, 3),
        "recall@5": round(recall, 3),
        "faithfulness": ragas_scores["faithfulness"],
        "answer_relevancy": ragas_scores["answer_relevancy"],
        "context_precision": ragas_scores["context_precision"],
        "avg_cost": cost_summary["avg_cost"],
        "avg_latency": cost_summary["avg_latency"]
    }
