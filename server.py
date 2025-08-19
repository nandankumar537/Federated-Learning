# import flwr as fl
# from flwr.server.strategy import FedAvg

# def main():
#     # Configure the FedAvg strategy (you can extend with evaluate_fn, min_fit_clients, etc.)
#     strategy = FedAvg(
#         fraction_fit=1.0,          # Sample all clients each round (adjust for larger simulations)
#         fraction_evaluate=1.0,     # Evaluate on all available clients
#         min_fit_clients=2,         # Minimum number of clients to be sampled for fit
#         min_evaluate_clients=2,    # Minimum number of clients to be sampled for evaluation
#         min_available_clients=2,   # Minimum number of clients that need to be connected
#     )

#     # Start Flower server
#     fl.server.start_server(
#         server_address="0.0.0.0:8080",
#         strategy=strategy,
#         config=fl.server.ServerConfig(num_rounds=5),
#     )

# if __name__ == "__main__":
#     main()

import flwr as fl
from flwr.server.strategy import FedAvg
from typing import Dict, List, Tuple, Optional

# Type aliases
Metrics = Dict[str, float]
EvaluateRes = Tuple[float, Dict[str, float]]  # (loss, metrics)

def aggregate_metrics_weighted(metrics_and_sizes: List[Tuple[Metrics, int]]) -> Metrics:
    # Weighted average by number of examples
    if not metrics_and_sizes:
        return {}
    totals = {}
    total_w = 0
    for m, n in metrics_and_sizes:
        total_w += n
        for k, v in m.items():
            totals[k] = totals.get(k, 0.0) + v * n
    return {k: (totals[k] / total_w if total_w > 0 else 0.0) for k in totals}

def get_evaluate_fn():
    # Server-side evaluate_fn aggregates client metrics sent via client.evaluate
    def evaluate(server_round: int, parameters, config) -> Optional[EvaluateRes]:
        # No centralized dataset here; return None to skip global server-side evaluation
        return None
    return evaluate

class FedAvgWithLogging(FedAvg):
    def __init__(self, server_logfile: str = "server_round_logs.txt", **kwargs):
        super().__init__(**kwargs)
        self.server_logfile = server_logfile

    def configure_fit(self, server_round, parameters, client_manager):
        # Send the round number to clients for logging
        config = {
            "server_round": server_round,
        }
        fit_ins = super().configure_fit(server_round, parameters, client_manager)
        # Inject config into each client's FitIns
        fit_cfg = []
        for cid, ins in fit_ins:
            ins.config.update(config)
            fit_cfg.append((cid, ins))
        return fit_cfg

    def configure_evaluate(self, server_round, parameters, client_manager):
        # Send the round number to clients for logging on evaluate as well
        eval_ins = super().configure_evaluate(server_round, parameters, client_manager)
        eval_cfg = []
        for cid, ins in eval_ins:
            ins.config.update({"server_round": server_round})
            eval_cfg.append((cid, ins))
        return eval_cfg

    def aggregate_evaluate(self, server_round, results, failures):
        # results: List of (ClientProxy, EvaluateResProto) -> has loss/num_examples/metrics
        agg = super().aggregate_evaluate(server_round, results, failures)
        # Collect metrics for manual weighted aggregation
        metrics_and_sizes = []
        for _, eval_res in results:
            n = eval_res.num_examples
            m = eval_res.metrics or {}
            # We’re interested in Accuracy, Precision, Recall, F1, MeanCVAccuracy
            selected = {k: float(m[k]) for k in ["accuracy", "precision", "recall", "f1", "mean_cv_accuracy"] if k in m}
            if selected:
                metrics_and_sizes.append((selected, n))

        if metrics_and_sizes:
            weighted = aggregate_metrics_weighted(metrics_and_sizes)
            # Log aggregated metrics to console
            print(f"[Server][Round {server_round}] Aggregated metrics: {weighted}")
            # Append to server log file
            with open(self.server_logfile, "a", encoding="utf-8") as f:
                if server_round == 1:
                    f.write("Round\tAccuracy\tPrecision\tRecall\tF1\tMeanCVAccuracy\n")
                f.write(
                    f"{server_round}\t"
                    f"{weighted.get('accuracy', float('nan')):.4f}\t"
                    f"{weighted.get('precision', float('nan')):.4f}\t"
                    f"{weighted.get('recall', float('nan')):.4f}\t"
                    f"{weighted.get('f1', float('nan')):.4f}\t"
                    f"{weighted.get('mean_cv_accuracy', float('nan')):.4f}\n"
                )
        return agg

def main():
    strategy = FedAvgWithLogging(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(),  # no centralized dataset, but keeps API consistent
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=5),
    )

if __name__ == "__main__":
    main()

