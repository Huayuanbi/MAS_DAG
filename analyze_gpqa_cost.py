from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def load_runs(pattern: str, first: int, last: int) -> list[list[dict]]:
    runs = []
    for version in range(first, last + 1):
        path = Path(pattern.format(version=version))
        runs.append(json.loads(path.read_text(encoding="utf-8")))
    shape = [[len(record["graphs"]) for record in run] for run in runs]
    if not runs or any(len(run) != len(runs[0]) for run in runs) or any(
        row != shape[0] for row in shape[1:]
    ):
        raise ValueError("runs do not contain the same questions and graph counts")
    return runs


def build_frames(runs: list[list[dict]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    trial_rows = []
    graph_rows = []
    for question, record in enumerate(runs[0]):
        for graph_index, graph in enumerate(record["graphs"]):
            trials = [run[question]["graphs"][graph_index] for run in runs]
            graph_id = str(graph.get("id", f"q{question}_g{graph_index}"))
            active_nodes = sum(value == 0 for value in graph["mask"])
            edge_count = sum(
                value != 0 for row in graph["edge_weight"] for value in row
            )
            for run_index, item in enumerate(trials):
                trial_rows.append(
                    {
                        "question": question,
                        "graph_index": graph_index,
                        "graph_id": graph_id,
                        "family": graph.get("generator", "unknown"),
                        "run": run_index,
                        "accuracy": float(item["accuracy"]),
                        "input_tokens": float(item["total_input_tokens"]),
                        "output_tokens": float(item["total_output_tokens"]),
                        "total_tokens": float(
                            item["total_input_tokens"] + item["total_output_tokens"]
                        ),
                        "seconds": float(item["wall_time_seconds"]),
                        "truncated": float(
                            "length" in item.get("node_finish_reason", [])
                        ),
                        "missing_answer": float(item.get("prediction") is None),
                        "active_nodes": active_nodes,
                        "edge_count": edge_count,
                    }
                )
            graph_rows.append(
                {
                    "question": question,
                    "graph_index": graph_index,
                    "graph_id": graph_id,
                    "family": graph.get("generator", "unknown"),
                    "score": np.mean([float(item["accuracy"]) for item in trials]),
                    "input_tokens": np.mean(
                        [float(item["total_input_tokens"]) for item in trials]
                    ),
                    "output_tokens": np.mean(
                        [float(item["total_output_tokens"]) for item in trials]
                    ),
                    "total_tokens": np.mean(
                        [
                            float(item["total_input_tokens"])
                            + float(item["total_output_tokens"])
                            for item in trials
                        ]
                    ),
                    "seconds": np.mean(
                        [float(item["wall_time_seconds"]) for item in trials]
                    ),
                    "truncation_rate": np.mean(
                        ["length" in item.get("node_finish_reason", []) for item in trials]
                    ),
                    "missing_rate": np.mean(
                        [item.get("prediction") is None for item in trials]
                    ),
                    "active_nodes": active_nodes,
                    "edge_count": edge_count,
                }
            )
    return pd.DataFrame(trial_rows), pd.DataFrame(graph_rows)


def correlation_table(graphs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "seconds",
        "active_nodes",
        "edge_count",
    ):
        pearson = stats.pearsonr(graphs[column], graphs["score"])
        spearman = stats.spearmanr(graphs[column], graphs["score"])
        centered_x = graphs[column] - graphs.groupby("question")[column].transform("mean")
        centered_y = graphs["score"] - graphs.groupby("question")["score"].transform("mean")
        within = stats.pearsonr(centered_x, centered_y)
        rows.append(
            {
                "cost": column,
                "pearson": pearson.statistic,
                "pearson_p": pearson.pvalue,
                "spearman": spearman.statistic,
                "within_question": within.statistic,
                "within_p": within.pvalue,
            }
        )
    return pd.DataFrame(rows)


def fixed_effect_models(graphs: pd.DataFrame) -> pd.DataFrame:
    frame = graphs.copy()
    for column in ("input_tokens", "output_tokens", "total_tokens", "seconds"):
        logged = np.log1p(frame[column])
        frame[f"zlog_{column}"] = (logged - logged.mean()) / logged.std()
    formulas = {
        "total_cost+question_FE": "score ~ zlog_total_tokens + C(question)",
        "io_cost+question_FE": (
            "score ~ zlog_input_tokens + zlog_output_tokens + C(question)"
        ),
        "io_cost+question+family_FE": (
            "score ~ zlog_input_tokens + zlog_output_tokens "
            "+ C(question) + C(family)"
        ),
        "structure+question_FE": (
            "score ~ active_nodes + edge_count + C(question)"
        ),
    }
    rows = []
    for name, formula in formulas.items():
        fitted = smf.ols(formula, data=frame).fit(
            cov_type="cluster", cov_kwds={"groups": frame["question"]}
        )
        for term in (
            "zlog_total_tokens",
            "zlog_input_tokens",
            "zlog_output_tokens",
            "active_nodes",
            "edge_count",
        ):
            if term in fitted.params:
                rows.append(
                    {
                        "model": name,
                        "term": term,
                        "coefficient": fitted.params[term],
                        "std_error": fitted.bse[term],
                        "p_value": fitted.pvalues[term],
                        "r_squared": fitted.rsquared,
                    }
                )
    return pd.DataFrame(rows)


def pairwise_cost_order(graphs: pd.DataFrame) -> dict[str, float | int]:
    counts = {"higher_score": 0, "equal_score": 0, "lower_score": 0, "pairs": 0}
    material = dict(counts)
    for _, group in graphs.groupby("question"):
        for (_, left), (_, right) in itertools.combinations(group.iterrows(), 2):
            expensive, cheap = (left, right) if left.total_tokens > right.total_tokens else (right, left)
            if expensive.total_tokens == cheap.total_tokens:
                continue
            counts["pairs"] += 1
            relation = (
                "higher_score" if expensive.score > cheap.score
                else "lower_score" if expensive.score < cheap.score
                else "equal_score"
            )
            counts[relation] += 1
            if expensive.total_tokens >= 1.25 * cheap.total_tokens:
                material["pairs"] += 1
                material[relation] += 1
    return {f"all_{key}": value for key, value in counts.items()} | {
        f"cost_gap_ge_25pct_{key}": value for key, value in material.items()
    }


def pareto_summary(graphs: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    frontier_counts = []
    family_frontier: dict[str, int] = {}
    cheapest_scores = []
    best_scores = []
    best_cost_ratios = []
    for _, group in graphs.groupby("question"):
        frontier = []
        for index, row in group.iterrows():
            dominated = any(
                other.score >= row.score
                and other.total_tokens <= row.total_tokens
                and (other.score > row.score or other.total_tokens < row.total_tokens)
                for other_index, other in group.iterrows()
                if other_index != index
            )
            if not dominated:
                frontier.append(row)
                family_frontier[row.family] = family_frontier.get(row.family, 0) + 1
        frontier_counts.append(len(frontier))
        cheapest = group.loc[group.total_tokens.idxmin()]
        best_score = group.score.max()
        best_low_cost = group[group.score == best_score].sort_values("total_tokens").iloc[0]
        cheapest_scores.append(cheapest.score)
        best_scores.append(best_score)
        best_cost_ratios.append(best_low_cost.total_tokens / cheapest.total_tokens)
    summary = {
        "mean_frontier_size": float(np.mean(frontier_counts)),
        "mean_cheapest_score": float(np.mean(cheapest_scores)),
        "mean_best_score": float(np.mean(best_scores)),
        "mean_best_to_cheapest_token_ratio": float(np.mean(best_cost_ratios)),
    }
    families = pd.DataFrame(
        sorted(family_frontier.items(), key=lambda item: -item[1]),
        columns=["family", "pareto_appearances"],
    )
    return summary, families


def utility_tradeoff(graphs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for penalty in (0.0, 0.02, 0.05, 0.10, 0.20, 0.50):
        chosen = []
        for _, group in graphs.groupby("question"):
            utility = group.score - penalty * group.total_tokens / 5000.0
            best = group.loc[utility.idxmax()]
            chosen.append(best)
        frame = pd.DataFrame(chosen)
        rows.append(
            {
                "token_penalty_per_5000": penalty,
                "oracle_score": frame.score.mean(),
                "total_tokens": frame.total_tokens.mean(),
                "seconds": frame.seconds.mean(),
                "active_nodes": frame.active_nodes.mean(),
                "single_agent_selected": (frame.family == "finalizer_only").mean(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze GPQA accuracy/cost trends.")
    parser.add_argument(
        "--pattern",
        default="data/gpqa/train_candidate_graphs_scored_with_outputs_v{version}.json",
    )
    parser.add_argument("--first-version", type=int, default=2)
    parser.add_argument("--last-version", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=Path("data/gpqa/cost_analysis"))
    args = parser.parse_args()

    runs = load_runs(args.pattern, args.first_version, args.last_version)
    trials, graphs = build_frames(runs)
    correlations = correlation_table(graphs)
    models = fixed_effect_models(graphs)
    family = graphs.groupby("family", as_index=False).agg(
        graphs=("score", "size"),
        score=("score", "mean"),
        input_tokens=("input_tokens", "mean"),
        output_tokens=("output_tokens", "mean"),
        total_tokens=("total_tokens", "mean"),
        seconds=("seconds", "mean"),
        truncation_rate=("truncation_rate", "mean"),
        missing_rate=("missing_rate", "mean"),
        active_nodes=("active_nodes", "mean"),
        edge_count=("edge_count", "mean"),
    ).sort_values("score", ascending=False)
    graphs["token_quintile"] = pd.qcut(graphs.total_tokens, 5, labels=False) + 1
    quintiles = graphs.groupby("token_quintile", as_index=False).agg(
        graphs=("score", "size"),
        score=("score", "mean"),
        total_tokens=("total_tokens", "mean"),
        input_tokens=("input_tokens", "mean"),
        output_tokens=("output_tokens", "mean"),
        seconds=("seconds", "mean"),
    )
    pairs = pairwise_cost_order(graphs)
    pareto, pareto_families = pareto_summary(graphs)
    tradeoff = utility_tradeoff(graphs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trials.to_csv(args.output_dir / "trial_level.csv", index=False)
    graphs.to_csv(args.output_dir / "graph_mean5.csv", index=False)
    family.to_csv(args.output_dir / "family_summary.csv", index=False)
    quintiles.to_csv(args.output_dir / "token_quintiles.csv", index=False)
    correlations.to_csv(args.output_dir / "correlations.csv", index=False)
    models.to_csv(args.output_dir / "fixed_effect_models.csv", index=False)
    tradeoff.to_csv(args.output_dir / "utility_tradeoff.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps({"pairwise": pairs, "pareto": pareto}, indent=2) + "\n",
        encoding="utf-8",
    )

    print("TOKEN QUINTILES")
    print(quintiles.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nCORRELATIONS")
    print(correlations.to_string(index=False, float_format=lambda value: f"{value:.4g}"))
    print("\nFIXED EFFECT MODELS")
    print(models.to_string(index=False, float_format=lambda value: f"{value:.4g}"))
    print("\nFAMILIES")
    print(family.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nPAIRWISE", json.dumps(pairs, sort_keys=True))
    print("PARETO", json.dumps(pareto, sort_keys=True))
    print(pareto_families.to_string(index=False))
    print("\nUTILITY TRADEOFF")
    print(tradeoff.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\noutputs={args.output_dir}")


if __name__ == "__main__":
    main()
