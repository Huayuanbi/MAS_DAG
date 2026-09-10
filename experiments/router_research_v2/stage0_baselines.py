from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler


HERE = Path(__file__).resolve().parent


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def load_cells(path: Path) -> tuple[list[dict], list[dict]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    cells = []
    for question_index, record in enumerate(records):
        for candidate_index, graph in enumerate(record["graphs"]):
            trials = [float(value) for value in graph.get("reward_trials", [])]
            if not trials:
                count = int(graph.get("num_trials", 1))
                successes = float(graph["reward"]) * count
                trials = [1.0] * round(successes) + [0.0] * (count - round(successes))
            cells.append(
                {
                    "question": question_index,
                    "candidate": candidate_index,
                    "family": str(graph.get("generator", "unknown")),
                    "active_nodes": sum(value == 0 for value in graph["mask"]),
                    "edge_count": sum(
                        value != 0 for row in graph["edge_weight"] for value in row
                    ),
                    "planned_tokens": sum(
                        int(value or 0) for value in graph.get("node_max_new_tokens", [])
                    ),
                    "tokens": float(graph.get("total_input_tokens", 0))
                    + float(graph.get("total_output_tokens", 0)),
                    "successes": float(sum(trials)),
                    "trials": len(trials),
                    "reward": float(sum(trials) / len(trials)),
                }
            )
    return records, cells


class FeatureBuilder:
    def __init__(self, records: list[dict], cells: list[dict], train_questions: np.ndarray):
        self.num_candidates = max(cell["candidate"] for cell in cells) + 1
        self.vectorizer = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), min_df=2, max_features=768,
            sublinear_tf=True,
        )
        train_text = [records[index]["task"] for index in train_questions]
        self.vectorizer.fit(train_text)
        families = np.asarray([[cell["family"]] for cell in cells], dtype=object)
        self.family_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        train_set = set(map(int, train_questions))
        self.family_encoder.fit(
            np.asarray([[cell["family"]] for cell in cells if cell["question"] in train_set], dtype=object)
        )
        structural = np.asarray(
            [[cell["active_nodes"], cell["edge_count"], np.log1p(cell["planned_tokens"])] for cell in cells],
            dtype=float,
        )
        self.scaler = StandardScaler().fit(
            structural[[cell["question"] in train_set for cell in cells]]
        )

    def transform(
        self,
        records: list[dict],
        cells: list[dict],
        question_permutation: dict[int, int] | None = None,
    ) -> sparse.csr_matrix:
        question_ids = [cell["question"] for cell in cells]
        text_ids = [
            question_permutation.get(index, index) if question_permutation else index
            for index in question_ids
        ]
        text = self.vectorizer.transform([records[index]["task"] for index in text_ids]).tocsr()
        candidate = sparse.csr_matrix(
            (
                np.ones(len(cells)),
                (np.arange(len(cells)), [cell["candidate"] for cell in cells]),
            ),
            shape=(len(cells), self.num_candidates),
        )
        family = self.family_encoder.transform(
            np.asarray([[cell["family"]] for cell in cells], dtype=object)
        )
        structural = sparse.csr_matrix(
            self.scaler.transform(
                np.asarray(
                    [
                        [cell["active_nodes"], cell["edge_count"], np.log1p(cell["planned_tokens"])]
                        for cell in cells
                    ],
                    dtype=float,
                )
            )
        )
        rows, columns = text.nonzero()
        interaction_columns = np.asarray([cells[row]["candidate"] for row in rows]) * text.shape[1] + columns
        interaction = sparse.csr_matrix(
            (text.data, (rows, interaction_columns)),
            shape=(len(cells), self.num_candidates * text.shape[1]),
        )
        return sparse.hstack((candidate, family, structural, text, interaction), format="csr")


def fit_binomial(
    features: sparse.csr_matrix,
    cells: list[dict],
    indices: list[int],
    c_value: float,
    alpha: float,
    beta: float,
) -> LogisticRegression:
    selected = features[indices]
    x = sparse.vstack((selected, selected), format="csr")
    y = np.concatenate((np.ones(len(indices)), np.zeros(len(indices))))
    successes = np.asarray([cells[index]["successes"] for index in indices])
    trials = np.asarray([cells[index]["trials"] for index in indices])
    weights = np.concatenate((successes + alpha, trials - successes + beta))
    model = LogisticRegression(C=c_value, max_iter=2000, solver="liblinear")
    model.fit(x, y, sample_weight=weights)
    return model


def select_by_scores(cells: list[dict], indices: list[int], scores: np.ndarray) -> list[int]:
    groups: dict[int, list[tuple[int, float]]] = {}
    for index, score in zip(indices, scores):
        groups.setdefault(cells[index]["question"], []).append((index, float(score)))
    return [max(rows, key=lambda item: (item[1], -item[0]))[0] for rows in groups.values()]


def select_fixed_family(cells: list[dict], indices: list[int], family: str) -> list[int]:
    groups: dict[int, list[int]] = {}
    for index in indices:
        if cells[index]["family"] == family:
            groups.setdefault(cells[index]["question"], []).append(index)
    expected_questions = {cells[index]["question"] for index in indices}
    if set(groups) != expected_questions:
        missing = sorted(expected_questions - set(groups))
        raise ValueError(f"fixed family {family!r} missing for questions {missing}")
    return [min(rows) for rows in groups.values()]


def selected_summary(cells: list[dict], selected: list[int], alpha: float, beta: float) -> dict:
    raw = np.asarray([cells[index]["reward"] for index in selected])
    posterior = np.asarray(
        [
            (cells[index]["successes"] + alpha)
            / (cells[index]["trials"] + alpha + beta)
            for index in selected
        ]
    )
    return {
        "questions": len(selected),
        "observed_reward": float(raw.mean()),
        "posterior_reward": float(posterior.mean()),
        "mean_tokens_diagnostic": float(np.mean([cells[index]["tokens"] for index in selected])),
        "selected_indices": selected,
    }


def prior_scores(
    cells: list[dict], train_indices: list[int], test_indices: list[int], key: str, alpha: float, beta: float
) -> np.ndarray:
    totals: dict[object, list[float]] = {}
    for index in train_indices:
        value = cells[index][key]
        row = totals.setdefault(value, [0.0, 0.0])
        row[0] += cells[index]["successes"]
        row[1] += cells[index]["trials"]
    global_success = sum(cells[index]["successes"] for index in train_indices)
    global_trials = sum(cells[index]["trials"] for index in train_indices)
    fallback = (global_success + alpha) / (global_trials + alpha + beta)
    return np.asarray(
        [
            (totals[cells[index][key]][0] + alpha)
            / (totals[cells[index][key]][1] + alpha + beta)
            if cells[index][key] in totals
            else fallback
            for index in test_indices
        ]
    )


def cell_metrics(cells: list[dict], indices: list[int], probabilities: np.ndarray) -> dict:
    targets = np.asarray([cells[index]["reward"] for index in indices])
    expanded_y = []
    expanded_p = []
    for index, probability in zip(indices, probabilities):
        successes = round(cells[index]["successes"])
        trials = cells[index]["trials"]
        expanded_y.extend([1] * successes + [0] * (trials - successes))
        expanded_p.extend([float(probability)] * trials)
    return {
        "cell_brier": float(np.mean((targets - probabilities) ** 2)),
        "trial_log_loss": float(log_loss(expanded_y, expanded_p, labels=[0, 1])),
    }


def tune_c(
    records: list[dict], cells: list[dict], outer_train_questions: np.ndarray,
    c_grid: list[float], alpha: float, beta: float, seed: int,
) -> float:
    inner = KFold(n_splits=3, shuffle=True, random_state=seed)
    values = {value: [] for value in c_grid}
    outer_train_questions = np.asarray(outer_train_questions)
    for inner_train_pos, inner_val_pos in inner.split(outer_train_questions):
        train_questions = outer_train_questions[inner_train_pos]
        val_questions = set(map(int, outer_train_questions[inner_val_pos]))
        train_set = set(map(int, train_questions))
        train_indices = [i for i, cell in enumerate(cells) if cell["question"] in train_set]
        val_indices = [i for i, cell in enumerate(cells) if cell["question"] in val_questions]
        builder = FeatureBuilder(records, cells, train_questions)
        features = builder.transform(records, cells)
        for c_value in c_grid:
            model = fit_binomial(features, cells, train_indices, c_value, alpha, beta)
            probabilities = model.predict_proba(features[val_indices])[:, 1]
            selected = select_by_scores(cells, val_indices, probabilities)
            values[c_value].append(np.mean([cells[index]["reward"] for index in selected]))
    return max(c_grid, key=lambda value: (np.mean(values[value]), -value))


def bootstrap_difference(values: np.ndarray, seed: int, draws: int = 10000) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    for index in range(draws):
        means[index] = rng.choice(values, size=len(values), replace=True).mean()
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser(description="Grouped-CV Stage-0 routing baselines.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "artifacts" / "stage0_baselines.json")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--fixed-family", default="parallel_solvers_verify")
    parser.add_argument("--c-grid", type=float, nargs="+", default=[0.03, 0.1, 0.3, 1.0])
    args = parser.parse_args()
    if args.folds < 2 or args.alpha <= 0 or args.beta <= 0:
        raise ValueError("invalid folds or beta prior")

    records, cells = load_cells(args.input)
    questions = np.arange(len(records))
    outer = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_results = []
    oof_difference = np.empty(len(records))
    oof_fixed_difference = np.empty(len(records))
    for fold, (train_pos, test_pos) in enumerate(outer.split(questions), start=1):
        train_questions = questions[train_pos]
        test_questions = questions[test_pos]
        train_set = set(map(int, train_questions))
        test_set = set(map(int, test_questions))
        train_indices = [i for i, cell in enumerate(cells) if cell["question"] in train_set]
        test_indices = [i for i, cell in enumerate(cells) if cell["question"] in test_set]
        c_value = tune_c(
            records, cells, train_questions, args.c_grid, args.alpha, args.beta,
            args.seed + fold,
        )
        builder = FeatureBuilder(records, cells, train_questions)
        features = builder.transform(records, cells)
        direct = fit_binomial(features, cells, train_indices, c_value, args.alpha, args.beta)
        direct_prob = direct.predict_proba(features[test_indices])[:, 1]

        rng = np.random.default_rng(args.seed + 1000 + fold)
        shuffled_questions = train_questions.copy()
        rng.shuffle(shuffled_questions)
        permutation = dict(zip(map(int, train_questions), map(int, shuffled_questions)))
        shuffled_features = builder.transform(records, cells, permutation)
        shuffled = fit_binomial(
            shuffled_features, cells, train_indices, c_value, args.alpha, args.beta
        )
        shuffled_prob = shuffled.predict_proba(features[test_indices])[:, 1]

        candidate_prob = prior_scores(
            cells, train_indices, test_indices, "candidate", args.alpha, args.beta
        )
        family_prob = prior_scores(
            cells, train_indices, test_indices, "family", args.alpha, args.beta
        )
        methods = {}
        selections = {}
        for name, probabilities in (
            ("candidate_prior", candidate_prob),
            ("family_prior", family_prob),
            ("direct_qg", direct_prob),
            ("question_shuffled", shuffled_prob),
        ):
            selected = select_by_scores(cells, test_indices, probabilities)
            selections[name] = selected
            methods[name] = selected_summary(cells, selected, args.alpha, args.beta) | cell_metrics(
                cells, test_indices, probabilities
            )
        fixed_selected = select_fixed_family(cells, test_indices, args.fixed_family)
        selections["fixed_family"] = fixed_selected
        methods["fixed_family"] = selected_summary(
            cells, fixed_selected, args.alpha, args.beta
        ) | {"family": args.fixed_family}
        for question, direct_index, prior_index, fixed_index in zip(
            test_questions,
            selections["direct_qg"],
            selections["candidate_prior"],
            selections["fixed_family"],
        ):
            oof_difference[int(question)] = cells[direct_index]["reward"] - cells[prior_index]["reward"]
            oof_fixed_difference[int(question)] = (
                cells[direct_index]["reward"] - cells[fixed_index]["reward"]
            )
        fold_results.append(
            {
                "fold": fold,
                "train_questions": list(map(int, train_questions)),
                "test_questions": list(map(int, test_questions)),
                "selected_c": c_value,
                "methods": methods,
                "direct_minus_candidate_prior": methods["direct_qg"]["observed_reward"]
                - methods["candidate_prior"]["observed_reward"],
                "direct_minus_fixed_family": methods["direct_qg"]["observed_reward"]
                - methods["fixed_family"]["observed_reward"],
            }
        )
        print(
            f"fold={fold} C={c_value:g} direct={methods['direct_qg']['observed_reward']:.3f} "
            f"candidate_prior={methods['candidate_prior']['observed_reward']:.3f} "
            f"shuffled={methods['question_shuffled']['observed_reward']:.3f}",
            flush=True,
        )

    prior_wins = sum(
        result["direct_minus_candidate_prior"] > 0 for result in fold_results
    )
    fixed_wins = sum(
        result["direct_minus_fixed_family"] > 0 for result in fold_results
    )
    result = {
        "schema_version": 1,
        "input": str(args.input.resolve()),
        "questions": len(records),
        "candidates_per_question": len(cells) // len(records),
        "nominal_trials_per_cell": sorted({cell["trials"] for cell in cells}),
        "split": {"type": "question_grouped_kfold", "folds": args.folds, "seed": args.seed},
        "label_model": {"likelihood": "binomial_logistic", "beta_prior": [args.alpha, args.beta]},
        "fixed_family_reference": args.fixed_family,
        "folds": fold_results,
        "gate": {
            "rule": (
                "direct_qg selected observed reward exceeds both the fold-trained "
                "candidate prior and the predeclared fixed family in >=4/5 folds"
            ),
            "candidate_prior_wins": prior_wins,
            "fixed_family_wins": fixed_wins,
            "passed": prior_wins >= 4 and fixed_wins >= 4,
            "direct_minus_candidate_prior": {
                "mean": float(oof_difference.mean()),
                "question_bootstrap_95_interval": bootstrap_difference(
                    oof_difference, args.seed
                ),
            },
            "direct_minus_fixed_family": {
                "mean": float(oof_fixed_difference.mean()),
                "question_bootstrap_95_interval": bootstrap_difference(
                    oof_fixed_difference, args.seed + 1
                ),
            },
            "interpretation": "exploratory progression gate only",
        },
    }
    atomic_json(args.output, result)
    print(json.dumps(result["gate"], indent=2))
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
