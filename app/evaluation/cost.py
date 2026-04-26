class CostTracker:
    def __init__(self):
        self.total_tokens = 0
        self.total_cost = 0.0
        self.total_latency = 0.0
        self.calls = 0

    def log(self, tokens: int, cost: float, latency: float):
        self.total_tokens += tokens
        self.total_cost += cost
        self.total_latency += latency
        self.calls += 1

    def summary(self):
        if self.calls == 0:
            return {}
        return {
            "avg_tokens": self.total_tokens / self.calls,
            "avg_cost": self.total_cost / self.calls,
            "avg_latency": self.total_latency / self.calls,
        }
