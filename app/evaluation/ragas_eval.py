from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

def run_ragas(samples):
    dataset = {
        "question": [s["question"] for s in samples],
        "answer": [s["answer"] for s in samples],
        "contexts": [s["contexts"] for s in samples],
        "ground_truth": [s["ground_truth"] for s in samples],
    }

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision]
    )
    return result
