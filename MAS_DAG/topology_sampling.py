from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, Iterable, Mapping, Sequence


DEFAULT_GSM8K_ROLES = (
    "problem_parser",
    "cot_solver",
    "equation_solver",
    "python_calculator",
    "critic",
    "finalizer",
)

# Used by the random component for a six-node pool. Anchor graphs already cover
# the all-node case heavily, so random sampling deliberately favors small graphs.
DEFAULT_ACTIVE_COUNT_WEIGHTS = (0.15, 0.25, 0.25, 0.20, 0.10, 0.05)


@dataclass(frozen=True)
class SampledTopology:
    generator: str
    mask: tuple[int, ...]
    adjacency: tuple[tuple[int, ...], ...]
    topological_order: tuple[int, ...]

    @property
    def num_nodes(self) -> int:
        return len(self.mask)

    @property
    def active_nodes(self) -> tuple[int, ...]:
        return tuple(index for index, value in enumerate(self.mask) if value == 0)

    @property
    def num_edges(self) -> int:
        return sum(sum(row) for row in self.adjacency)

    @property
    def signature(self) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
        return self.mask, self.adjacency

    def to_graph_record(self, reward: float | None = None) -> dict:
        """Convert to the grouped JSON candidate-graph representation."""
        zeros = [[0.0] * self.num_nodes for _ in range(self.num_nodes)]
        return {
            "generator": self.generator,
            "topological_order": list(self.topological_order),
            "reward": reward,
            "mask": list(self.mask),
            "edge_weight": [list(map(float, row)) for row in self.adjacency],
            "edge_token_cost": [row.copy() for row in zeros],
            "edge_time_cost": [row.copy() for row in zeros],
        }


def _get_rng(rng: random.Random | None) -> random.Random:
    return rng if rng is not None else random.Random()


def _active_nodes(
    num_nodes: int,
    finalizer: int,
    active_nodes: Iterable[int] | None,
) -> tuple[int, ...]:
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    if not 0 <= finalizer < num_nodes:
        raise ValueError("finalizer must be a valid node index")
    active = tuple(
        range(num_nodes) if active_nodes is None else dict.fromkeys(active_nodes)
    )
    if not active:
        raise ValueError("at least one node must be active")
    if any(node < 0 or node >= num_nodes for node in active):
        raise ValueError("active_nodes contains an invalid node index")
    if finalizer not in active:
        raise ValueError("the finalizer must be active")
    return active


def _node_order(
    active_nodes: Sequence[int],
    finalizer: int,
    rng: random.Random,
    fixed_order: Sequence[int] | None = None,
) -> tuple[int, ...]:
    if fixed_order is not None:
        if len(fixed_order) != len(set(fixed_order)):
            raise ValueError("fixed_order must not contain duplicate nodes")
        if finalizer not in fixed_order or fixed_order[-1] != finalizer:
            raise ValueError("fixed_order must contain the finalizer last")
        active = set(active_nodes)
        if not active.issubset(fixed_order):
            raise ValueError("fixed_order must contain every active node")
        return tuple(node for node in fixed_order if node in active)
    prefix = [node for node in active_nodes if node != finalizer]
    rng.shuffle(prefix)
    return tuple(prefix + [finalizer])


def _build_topology(
    generator: str,
    num_nodes: int,
    active_nodes: Sequence[int],
    order: Sequence[int],
    edges: Iterable[tuple[int, int]],
    finalizer: int,
) -> SampledTopology:
    active = set(active_nodes)
    adjacency = [[0] * num_nodes for _ in range(num_nodes)]
    for source, target in edges:
        if source not in active or target not in active:
            raise ValueError("an edge references a masked node")
        adjacency[source][target] = 1
    topology = SampledTopology(
        generator=generator,
        mask=tuple(0 if node in active else 1 for node in range(num_nodes)),
        adjacency=tuple(tuple(row) for row in adjacency),
        topological_order=tuple(order),
    )
    validate_topology(topology, finalizer)
    return topology


def _has_path(
    adjacency: Sequence[Sequence[int]], source: int, target: int
) -> bool:
    stack = [source]
    visited: set[int] = set()
    while stack:
        node = stack.pop()
        if node == target:
            return True
        if node in visited:
            continue
        visited.add(node)
        stack.extend(index for index, edge in enumerate(adjacency[node]) if edge)
    return False


def validate_topology(topology: SampledTopology, finalizer: int) -> None:
    """Validate DAG, masking, unique-sink, and finalizer reachability rules."""
    num_nodes = topology.num_nodes
    if len(topology.adjacency) != num_nodes or any(
        len(row) != num_nodes for row in topology.adjacency
    ):
        raise ValueError("adjacency must have shape [num_nodes, num_nodes]")
    if any(value not in (0, 1) for value in topology.mask):
        raise ValueError("mask values must be 0 or 1")
    if topology.mask[finalizer] != 0:
        raise ValueError("the finalizer must be active")

    active = set(topology.active_nodes)
    if set(topology.topological_order) != active or len(topology.topological_order) != len(active):
        raise ValueError("topological_order must contain every active node exactly once")
    if topology.topological_order[-1] != finalizer:
        raise ValueError("the finalizer must be last in topological_order")

    rank = {node: index for index, node in enumerate(topology.topological_order)}
    for source, row in enumerate(topology.adjacency):
        for target, edge in enumerate(row):
            if edge not in (0, 1):
                raise ValueError("adjacency values must be 0 or 1")
            if not edge:
                continue
            if source == target:
                raise ValueError("self-loops are not allowed")
            if source not in active or target not in active:
                raise ValueError("masked nodes cannot have incident edges")
            if rank[source] >= rank[target]:
                raise ValueError("an edge violates topological_order")

    if any(topology.adjacency[finalizer]):
        raise ValueError("the finalizer must not have outgoing edges")
    for node in active:
        if not _has_path(topology.adjacency, node, finalizer):
            raise ValueError(f"active node {node} cannot reach the finalizer")


def generate_chain(
    num_nodes: int,
    finalizer: int,
    *,
    active_nodes: Iterable[int] | None = None,
    rng: random.Random | None = None,
    fixed_order: Sequence[int] | None = None,
) -> SampledTopology:
    rng = _get_rng(rng)
    active = _active_nodes(num_nodes, finalizer, active_nodes)
    order = _node_order(active, finalizer, rng, fixed_order)
    return _build_topology(
        "chain", num_nodes, active, order, zip(order, order[1:]), finalizer
    )


def generate_star(
    num_nodes: int,
    finalizer: int,
    *,
    active_nodes: Iterable[int] | None = None,
    rng: random.Random | None = None,
    fixed_order: Sequence[int] | None = None,
) -> SampledTopology:
    rng = _get_rng(rng)
    active = _active_nodes(num_nodes, finalizer, active_nodes)
    order = _node_order(active, finalizer, rng, fixed_order)
    edges = ((node, finalizer) for node in active if node != finalizer)
    return _build_topology("star", num_nodes, active, order, edges, finalizer)


def generate_tree(
    num_nodes: int,
    finalizer: int,
    *,
    active_nodes: Iterable[int] | None = None,
    rng: random.Random | None = None,
    fixed_order: Sequence[int] | None = None,
) -> SampledTopology:
    rng = _get_rng(rng)
    active = _active_nodes(num_nodes, finalizer, active_nodes)
    order = _node_order(active, finalizer, rng, fixed_order)
    edges = [
        (node, rng.choice(order[index + 1 :]))
        for index, node in enumerate(order[:-1])
    ]
    return _build_topology("tree", num_nodes, active, order, edges, finalizer)


def generate_complete_dag(
    num_nodes: int,
    finalizer: int,
    *,
    active_nodes: Iterable[int] | None = None,
    rng: random.Random | None = None,
    fixed_order: Sequence[int] | None = None,
) -> SampledTopology:
    rng = _get_rng(rng)
    active = _active_nodes(num_nodes, finalizer, active_nodes)
    order = _node_order(active, finalizer, rng, fixed_order)
    edges = (
        (source, target)
        for index, source in enumerate(order)
        for target in order[index + 1 :]
    )
    return _build_topology(
        "complete_dag", num_nodes, active, order, edges, finalizer
    )


def _backbone_edges(order: Sequence[int], rng: random.Random) -> set[tuple[int, int]]:
    return {
        (node, rng.choice(order[index + 1 :]))
        for index, node in enumerate(order[:-1])
    }


def _add_random_forward_edges(
    order: Sequence[int],
    edges: set[tuple[int, int]],
    probability: float,
    rng: random.Random,
    *,
    force_one: bool = False,
) -> None:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("edge probability must be between 0 and 1")
    candidates = [
        (source, target)
        for index, source in enumerate(order)
        for target in order[index + 1 :]
        if (source, target) not in edges
    ]
    selected = [edge for edge in candidates if rng.random() < probability]
    if force_one and probability > 0 and candidates and not selected:
        selected.append(rng.choice(candidates))
    edges.update(selected)


def generate_sparse_random(
    num_nodes: int,
    finalizer: int,
    *,
    active_nodes: Iterable[int] | None = None,
    extra_edge_probability: float = 0.15,
    rng: random.Random | None = None,
    fixed_order: Sequence[int] | None = None,
) -> SampledTopology:
    rng = _get_rng(rng)
    active = _active_nodes(num_nodes, finalizer, active_nodes)
    order = _node_order(active, finalizer, rng, fixed_order)
    edges = _backbone_edges(order, rng)
    _add_random_forward_edges(
        order, edges, extra_edge_probability, rng, force_one=True
    )
    return _build_topology(
        "sparse_random", num_nodes, active, order, edges, finalizer
    )


def generate_finalizer_only(num_nodes: int, finalizer: int) -> SampledTopology:
    return _build_topology(
        "finalizer_only",
        num_nodes,
        (finalizer,),
        (finalizer,),
        (),
        finalizer,
    )


def generate_two_node(
    num_nodes: int,
    finalizer: int,
    *,
    specialist: int | None = None,
    rng: random.Random | None = None,
) -> SampledTopology:
    rng = _get_rng(rng)
    candidates = [node for node in range(num_nodes) if node != finalizer]
    if not candidates:
        raise ValueError("a two-node graph requires a non-finalizer node")
    specialist = rng.choice(candidates) if specialist is None else specialist
    if specialist not in candidates:
        raise ValueError("specialist must be a non-finalizer node")
    return _build_topology(
        "two_node",
        num_nodes,
        (specialist, finalizer),
        (specialist, finalizer),
        ((specialist, finalizer),),
        finalizer,
    )


def generate_random_dag(
    num_nodes: int,
    finalizer: int,
    *,
    active_count: int | None = None,
    active_count_weights: Sequence[float] | None = None,
    extra_edge_probability: float | None = None,
    rng: random.Random | None = None,
    fixed_order: Sequence[int] | None = None,
) -> SampledTopology:
    rng = _get_rng(rng)
    if active_count is None:
        weights = (
            tuple(active_count_weights)
            if active_count_weights is not None
            else (DEFAULT_ACTIVE_COUNT_WEIGHTS[:num_nodes] if num_nodes <= 6 else (1.0,) * num_nodes)
        )
        if len(weights) != num_nodes or any(weight < 0 for weight in weights):
            raise ValueError("active_count_weights must contain one non-negative value per node")
        if not any(weights):
            raise ValueError("active_count_weights must include a positive value")
        active_count = rng.choices(range(1, num_nodes + 1), weights=weights, k=1)[0]
    if not 1 <= active_count <= num_nodes:
        raise ValueError("active_count must be between 1 and num_nodes")

    non_finalizers = [node for node in range(num_nodes) if node != finalizer]
    selected = rng.sample(non_finalizers, active_count - 1)
    active = tuple(selected + [finalizer])
    order = _node_order(active, finalizer, rng, fixed_order)
    edges = _backbone_edges(order, rng)
    probability = (
        rng.choice((0.1, 0.3, 0.5))
        if extra_edge_probability is None
        else extra_edge_probability
    )
    _add_random_forward_edges(order, edges, probability, rng)
    return _build_topology("random_dag", num_nodes, active, order, edges, finalizer)


MATH_ROLE_NAMES = (
    "problem_analyst",
    "strategy_planner",
    "primary_solver",
    "alternative_solver",
    "symbolic_proof_verifier",
    "finalizer",
)


def generate_math_role_anchors(
    num_nodes: int,
    finalizer: int,
    role_indices: Mapping[str, int],
    *,
    fixed_order: Sequence[int],
) -> list[SampledTopology]:
    """Build meaningful MATH agent DAGs instead of role-agnostic shapes."""
    missing = [role for role in MATH_ROLE_NAMES if role not in role_indices]
    if missing:
        raise ValueError(f"missing MATH roles: {', '.join(missing)}")
    a = role_indices["problem_analyst"]
    p = role_indices["strategy_planner"]
    s = role_indices["primary_solver"]
    alt = role_indices["alternative_solver"]
    v = role_indices["symbolic_proof_verifier"]
    f = role_indices["finalizer"]
    if f != finalizer:
        raise ValueError("the finalizer role does not match finalizer")

    def build(
        generator: str,
        active: Sequence[int],
        edges: Sequence[tuple[int, int]],
    ) -> SampledTopology:
        active_set = set(active)
        order = tuple(node for node in fixed_order if node in active_set)
        return _build_topology(generator, num_nodes, active, order, edges, finalizer)

    return [
        # Strongest anchor: two independent solutions, a joint verifier, and
        # direct solver-to-finalizer edges so verification is not a bottleneck.
        build(
            "expert_anchor",
            (a, p, s, alt, v, f),
            (
                (a, p),
                (p, s),
                (s, v),
                (alt, v),
                (s, f),
                (alt, f),
                (v, f),
            ),
        ),
        # Lower-cost primary route with Alternative Solver removed.
        build(
            "primary_pipeline",
            (a, p, s, v, f),
            ((a, p), (p, s), (s, v), (s, f), (v, f)),
        ),
        # Compare two independent solvers without spending tokens on planning.
        build(
            "dual_solver_review",
            (s, alt, v, f),
            ((s, v), (alt, v), (s, f), (alt, f), (v, f)),
        ),
        # Test whether direct aggregation is better than an explicit verifier.
        build(
            "dual_solver_direct",
            (s, alt, f),
            ((s, f), (alt, f)),
        ),
        # A conventional analysis-plan-solve route with direct finalization.
        build(
            "planned_primary",
            (a, p, s, f),
            ((a, p), (p, s), (s, f)),
        ),
    ]


def generate_math_random_dag(
    num_nodes: int,
    finalizer: int,
    role_indices: Mapping[str, int],
    *,
    rng: random.Random,
    fixed_order: Sequence[int],
) -> SampledTopology:
    """Sample a diverse DAG while preserving the semantics of MATH roles."""
    missing = [role for role in MATH_ROLE_NAMES if role not in role_indices]
    if missing:
        raise ValueError(f"missing MATH roles: {', '.join(missing)}")
    a = role_indices["problem_analyst"]
    p = role_indices["strategy_planner"]
    s = role_indices["primary_solver"]
    alt = role_indices["alternative_solver"]
    v = role_indices["symbolic_proof_verifier"]
    f = role_indices["finalizer"]
    if f != finalizer:
        raise ValueError("the finalizer role does not match finalizer")

    # finalizer_only is already an explicit baseline. Random candidates should
    # contain at least one agent capable of producing a solution.
    active_count = rng.choices(
        range(2, num_nodes + 1),
        weights=DEFAULT_ACTIVE_COUNT_WEIGHTS[1:num_nodes],
        k=1,
    )[0]
    required_solver = rng.choice((s, alt))
    optional = [node for node in (a, p, s, alt, v) if node != required_solver]
    active = set(rng.sample(optional, active_count - 2))
    active.update((required_solver, f))
    order = tuple(node for node in fixed_order if node in active)

    allowed_targets = {
        a: (p, s, alt),
        p: (s, alt),
        s: (v, f),
        alt: (v, f),
        v: (f,),
        f: (),
    }
    edges: set[tuple[int, int]] = set()
    reaches_finalizer = {f}
    for source in reversed(order[:-1]):
        candidates = [
            target
            for target in allowed_targets[source]
            if target in active and target in reaches_finalizer
        ]
        if not candidates:
            # This can only happen for a planner/analyst when the randomly
            # required solver is absent from its allowed descendants. Retry at
            # suite level with another sampled active set.
            raise ValueError("sampled roles do not form a semantic path")
        edges.add((source, rng.choice(candidates)))
        reaches_finalizer.add(source)

    extra_candidates = [
        (source, target)
        for source in order[:-1]
        for target in allowed_targets[source]
        if target in active and (source, target) not in edges
    ]
    edges.update(edge for edge in extra_candidates if rng.random() < 0.3)
    return _build_topology(
        "random_dag", num_nodes, tuple(active), order, edges, finalizer
    )


def generate_candidate_suite(
    num_nodes: int = 6,
    finalizer: int = 5,
    *,
    random_count: int = 5,
    seed: int | None = None,
    fixed_order: Sequence[int] | None = None,
    role_indices: Mapping[str, int] | None = None,
) -> list[SampledTopology]:
    """Generate 5 anchors, 2 low-cost baselines, and random connected DAGs."""
    if random_count < 0:
        raise ValueError("random_count must be non-negative")
    rng = random.Random(seed)
    # With semantic role ordering, anchors are dataset-wide controls and must
    # not change from one task seed to another. Random DAGs still use `seed`.
    anchor_rng = random.Random(0) if fixed_order is not None else rng
    topologies: list[SampledTopology] = []
    signatures: set[tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]] = set()

    def add_unique(factory: Callable[[], SampledTopology]) -> None:
        for _ in range(200):
            topology = factory()
            if topology.signature not in signatures:
                signatures.add(topology.signature)
                topologies.append(topology)
                return
        raise RuntimeError("could not generate a unique candidate topology")

    if fixed_order is not None and set(fixed_order) != set(range(num_nodes)):
        raise ValueError("fixed_order must contain every node exactly once")
    if fixed_order is not None and fixed_order[-1] != finalizer:
        raise ValueError("fixed_order must place the finalizer last")

    if role_indices is not None:
        if fixed_order is None:
            raise ValueError("role-aware anchors require fixed_order")
        for anchor in generate_math_role_anchors(
            num_nodes,
            finalizer,
            role_indices,
            fixed_order=fixed_order,
        ):
            add_unique(lambda anchor=anchor: anchor)
    else:
        add_unique(
            lambda: generate_chain(
                num_nodes, finalizer, rng=anchor_rng, fixed_order=fixed_order
            )
        )
        add_unique(
            lambda: generate_star(
                num_nodes, finalizer, rng=anchor_rng, fixed_order=fixed_order
            )
        )
        add_unique(
            lambda: generate_tree(
                num_nodes, finalizer, rng=anchor_rng, fixed_order=fixed_order
            )
        )
        add_unique(
            lambda: generate_complete_dag(
                num_nodes, finalizer, rng=anchor_rng, fixed_order=fixed_order
            )
        )
        add_unique(
            lambda: generate_sparse_random(
                num_nodes, finalizer, rng=anchor_rng, fixed_order=fixed_order
            )
        )
    add_unique(lambda: generate_finalizer_only(num_nodes, finalizer))
    specialist = (
        role_indices.get("primary_solver") if role_indices is not None else None
    )
    add_unique(
        lambda: generate_two_node(
            num_nodes, finalizer, specialist=specialist, rng=anchor_rng
        )
    )
    for _ in range(random_count):
        if role_indices is not None:
            def semantic_random() -> SampledTopology:
                for _ in range(200):
                    try:
                        return generate_math_random_dag(
                            num_nodes,
                            finalizer,
                            role_indices,
                            rng=rng,
                            fixed_order=fixed_order,
                        )
                    except ValueError as exc:
                        if str(exc) != "sampled roles do not form a semantic path":
                            raise
                raise RuntimeError("could not sample compatible MATH roles")

            add_unique(semantic_random)
        else:
            add_unique(
                lambda: generate_random_dag(
                    num_nodes,
                    finalizer,
                    active_count_weights=DEFAULT_ACTIVE_COUNT_WEIGHTS[:num_nodes],
                    rng=rng,
                    fixed_order=fixed_order,
                )
            )
    return topologies
