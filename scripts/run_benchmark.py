from app.evaluation.dataset import load_dataset
from app.evaluation.runner import evaluate
from app.evaluation.cost import CostTracker
from app.evaluation.ragas_eval import run_ragas
from app.evaluation.report import summarize

# You must implement build_graph in your project
from app.graph import build_graph

def main():
    dataset = load_dataset("data/eval_dataset.json")
    graph = build_graph()

    cost_tracker = CostTracker()

    results = evaluate(graph, dataset, cost_tracker)

    ragas_scores = run_ragas(results)
    summary = summarize(results, cost_tracker.summary(), ragas_scores)

    print("\n=== BENCHMARK REPORT ===\n")
    for k, v in summary.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
