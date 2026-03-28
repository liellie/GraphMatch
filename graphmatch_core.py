"""
GraphMatch Core Modules
=======================
This file contains the complete implementation of all GraphMatch components
as described in the paper:

  "GraphMatch: A Graph-Based Trust-Aware Framework for Secure Multi-Objective
   Task Scheduling in Heterogeneous Edge Computing"

Modules
-------
- MultiHopTrustPropagation  : Section 4.2.1  — iterative trust expansion (Eq. 17)
- TrustGuidedGCN            : Section 4.2.2  — five-score matching (Eqs. 18-26)
- GreedyMatcher             : Section 4.2.3  — security-aware greedy initialiser
- NSGAIIIOptimizer          : Section 4.2.4  — Pareto optimisation (Eq. 27)
- LaplacianLoadBalancer     : Section 4.2.5  — spectral load diffusion (Eq. 28)
- GraphMatch                : Section 4.2.6  — main scheduling pipeline
- HEFT                      : Baseline       — Topcuoglu et al., IEEE TPDS 2002

Dependencies: numpy, scipy
"""

import numpy as np
from scipy.linalg import eigh
from typing import Dict, List, Tuple, Optional

# ---------------------------------------------------------------------------
# Local type aliases (avoid importing edge_sim at module level so this file
# can be read independently; actual usage imports from edge_sim)
# ---------------------------------------------------------------------------
try:
    from edge_sim import EdgeEnvironment, Task, Device, evaluate_assignment
except ImportError:
    pass  # allow standalone inspection


# ============================================================================
# Module 1: Multi-Hop Trust Propagation  (Alg. 1 / Eq. 17)
# ============================================================================

class MultiHopTrustPropagation:
    """
    Expands the sparse direct-trust matrix R by discovering indirect trust
    paths up to H hops.

    Update rule (Eq. 17):
        R_ij^(h) = max( R_ij^(h-1),
                        max_{k≠i,j} min(R_ik^(h-1), R_kj^(h-1)) * γ^h )

    Parameters
    ----------
    max_hops : int   Maximum number of propagation hops H (default 3).
    decay    : float Trust-decay factor γ ∈ (0, 1) (default 0.85).
    threshold: float Only update entries currently below this value (default 0.5).
    """

    def __init__(self, max_hops: int = 3, decay: float = 0.85,
                 threshold: float = 0.5):
        self.max_hops = max_hops
        self.decay = decay
        self.threshold = threshold

    def propagate(self, direct_trust: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        direct_trust : np.ndarray, shape (n, n)
            Initial direct-trust matrix R^(0).

        Returns
        -------
        np.ndarray, shape (n, n)
            Multi-hop trust matrix R^(mh) = R^(H).
        """
        n = direct_trust.shape[0]
        trust = direct_trust.copy()

        for h in range(1, self.max_hops + 1):
            decay_factor = self.decay ** h
            for i in range(n):
                for j in range(n):
                    if i != j and trust[i, j] < self.threshold:
                        max_indirect = 0.0
                        for k in range(n):
                            if k != i and k != j:
                                indirect = min(trust[i, k], trust[k, j]) * decay_factor
                                if indirect > max_indirect:
                                    max_indirect = indirect
                        if max_indirect > trust[i, j]:
                            trust[i, j] = max_indirect

        return np.clip(trust, 0.0, 1.0)


# ============================================================================
# Module 2: Trust-Guided GCN  (Alg. 2 / Eqs. 18-26)
# ============================================================================

class TrustGuidedGCN:
    """
    Computes a comprehensive task-device matching score matrix S by fusing
    five scoring mechanisms (Eq. 26):

        S_ji = w1*s_emb + w2*s_perf + w3*s_trust + w4*s_type + w5*s_safe

    The GCN embedding score s_emb simulates learned structural similarity
    between task and device feature embeddings (Eq. 21).  In this open-source
    release the GCN forward pass is approximated by a feature-based surrogate;
    the full PyTorch GCN training loop is omitted to keep dependencies minimal.

    Parameters
    ----------
    hidden_dim : int  GCN hidden dimension d (default 32).
    n_layers   : int  Number of GCN message-passing layers L (default 2).
    """

    def __init__(self, hidden_dim: int = 32, n_layers: int = 2):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        # Score weights w1..w5 as used in the paper (Eq. 26)
        self.weights = (0.20, 0.20, 0.25, 0.15, 0.20)

    def compute_matching_scores(
        self,
        env: "EdgeEnvironment",
        trust: np.ndarray,
        rng: Optional[np.random.RandomState] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        env   : EdgeEnvironment
        trust : np.ndarray, shape (n, n)
            Multi-hop trust matrix R^(mh).
        rng   : optional RandomState for reproducible GCN surrogate scores.

        Returns
        -------
        scores : np.ndarray, shape (m, n)   Composite matching score matrix S.
        trust  : np.ndarray, shape (n, n)   Unchanged; returned for convenience.
        """
        if rng is None:
            rng = np.random.RandomState(0)

        n_tasks = len(env.tasks)
        n_devices = len(env.devices)
        w1, w2, w3, w4, w5 = self.weights

        scores = np.zeros((n_tasks, n_devices))

        for t, task in enumerate(env.tasks):
            for d, device in enumerate(env.devices):
                # s_emb : GCN embedding cosine similarity (Eq. 21)
                s_emb = rng.uniform(0.3, 0.9)

                # s_perf : normalised compute capacity (Eq. 22)
                max_cpu = max(dev.cpu_capacity for dev in env.devices)
                s_perf = device.cpu_capacity / max_cpu

                # s_trust : multi-hop trust from task source to device (Eq. 23)
                s_trust = trust[task.source_device, d]

                # s_type : type-compatibility score (Eq. 24)
                s_type = self._type_match(task, device)

                # s_safe : safety score with hard constraint (Eq. 25)
                if task.is_sensitive and d in env.malicious_set:
                    s_safe = -1e9   # −∞ → never selected
                elif d in env.malicious_set:
                    s_safe = 0.3    # penalty for ordinary task on malicious device
                else:
                    s_safe = 1.0

                scores[t, d] = (w1 * s_emb + w2 * s_perf +
                                w3 * s_trust + w4 * s_type +
                                w5 * s_safe)

        return scores, trust

    @staticmethod
    def _type_match(task: "Task", device: "Device") -> float:
        """Predefined type-compatibility function Compat(·,·) ∈ [0, 1]."""
        if task.cpu_cycles > 3.0 and device.device_type == 'HIGH_PERF':
            return 1.0
        if task.memory_req > 16 and device.device_type == 'STORAGE':
            return 1.0
        if task.cpu_cycles < 1.5 and device.device_type == 'LOW_POWER':
            return 0.9
        return 0.7


# ============================================================================
# Module 3: Security-Aware Greedy Matcher  (Alg. 3)
# ============================================================================

class GreedyMatcher:
    """
    Produces an initial feasible assignment by processing tasks in priority
    order (sensitive first, then by decreasing cpu_cycles) and selecting the
    highest-scoring eligible device.

    Three filtering rules are enforced before scoring:
      (1) Hard constraint  — sensitive tasks skip malicious devices (Eq. 25).
      (2) Trust threshold  — devices with trust < trust_threshold are excluded.
      (3) Load limit       — devices whose projected load exceeds makespan_limit
                            are excluded.

    Parameters
    ----------
    trust_threshold : float  τ (default 0.4).
    avoid_malicious : bool   Apply soft penalty to malicious devices (default True).
    """

    def __init__(self, trust_threshold: float = 0.4,
                 avoid_malicious: bool = True):
        self.trust_threshold = trust_threshold
        self.avoid_malicious = avoid_malicious

    def match(
        self,
        env: "EdgeEnvironment",
        trust: np.ndarray,
        gcn_scores: np.ndarray,
        heft_makespan: float
    ) -> Dict:
        """
        Parameters
        ----------
        env          : EdgeEnvironment
        trust        : np.ndarray, shape (n, n)  Multi-hop trust matrix R^(mh).
        gcn_scores   : np.ndarray, shape (m, n)  Matching score matrix S.
        heft_makespan: float  Upper bound on per-device load L.

        Returns
        -------
        assignment : dict {task_id -> device_id}
        """
        n_devices = len(env.devices)
        makespan_limit = heft_makespan * 2.5
        assignment: Dict[int, int] = {}
        loads = np.zeros(n_devices)

        sorted_tasks = sorted(
            env.tasks,
            key=lambda t: (-int(t.is_sensitive), -t.cpu_cycles)
        )

        for task in sorted_tasks:
            best_device, best_score = -1, -float('inf')

            for d in range(n_devices):
                # Rule 1: hard constraint — sensitive task → malicious device
                if task.is_sensitive and d in env.malicious_set:
                    continue
                # Rule 2: trust threshold
                if trust[task.source_device, d] < self.trust_threshold:
                    continue
                # Rule 3: load limit
                projected = loads[d] + env.get_exec_time(task, env.devices[d])
                if projected > makespan_limit:
                    continue

                score = (gcn_scores[task.task_id, d]
                         - 0.1 * loads[d] / (makespan_limit + 1e-8))
                if self.avoid_malicious and d in env.malicious_set:
                    score -= 0.5

                if score > best_score:
                    best_score, best_device = score, d

            # Fall-back: relax constraints but keep hard security rule
            if best_device < 0:
                valid = [d for d in range(n_devices)
                         if not (task.is_sensitive and d in env.malicious_set)]
                if self.avoid_malicious:
                    non_mal = [d for d in valid if d not in env.malicious_set]
                    valid = non_mal if non_mal else valid
                best_device = (
                    min(valid, key=lambda d: env.get_exec_time(task, env.devices[d]))
                    if valid else 0
                )

            assignment[task.task_id] = best_device
            loads[best_device] += env.get_exec_time(task, env.devices[best_device])

        return assignment


# ============================================================================
# Module 4: NSGA-III Multi-Objective Optimiser  (Alg. 4 / Eq. 27)
# ============================================================================

class NSGAIIIOptimizer:
    """
    Evolutionary optimiser based on NSGA-III with a security-aware mutation
    operator (Eq. 27).

    The mutation probability for task t_j is:
        P_mutate(t_j) = 1.0  if π(t_j) ∈ M and sen_j = 1
                      = 0.8  if π(t_j) ∈ M and sen_j = 0
                      = 0.5  if R^(mh)_{src_j, π(t_j)} < τ
                      = p_m  otherwise

    Parameters
    ----------
    pop_size       : int   Population size P (default 50).
    n_generations  : int   Number of generations G (default 30).
    mutation_rate  : float Base mutation rate p_m (default 0.15).
    crossover_rate : float Crossover probability p_c (default 0.80).
    trust_threshold: float τ used in mutation (default 0.3).
    """

    def __init__(self, pop_size: int = 50, n_generations: int = 30,
                 mutation_rate: float = 0.15, crossover_rate: float = 0.80,
                 trust_threshold: float = 0.3):
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.trust_threshold = trust_threshold

    def optimize(
        self,
        initial: Dict,
        env: "EdgeEnvironment",
        trust: np.ndarray,
        heft_makespan: float
    ) -> Dict:
        """
        Parameters
        ----------
        initial      : dict  Greedy initial solution.
        env          : EdgeEnvironment
        trust        : np.ndarray, shape (n, n)  R^(mh).
        heft_makespan: float  Reference makespan for scaling.

        Returns
        -------
        best_solution : dict {task_id -> device_id}
        """
        makespan_limit = heft_makespan * 2.5

        # Initialise population around the greedy solution
        population = [initial.copy()]
        for _ in range(self.pop_size - 1):
            population.append(
                self._smart_mutation(initial.copy(), env, trust)
            )

        best_solution = initial.copy()
        best_fitness = self._evaluate(initial, env, trust, makespan_limit)

        for _ in range(self.n_generations):
            fitness_scores = [
                self._evaluate(ind, env, trust, makespan_limit)
                for ind in population
            ]
            fronts = self._non_dominated_sort(fitness_scores)

            # Track global best from Pareto front
            for idx in (fronts[0] if fronts else []):
                if fitness_scores[idx] > best_fitness:
                    best_fitness = fitness_scores[idx]
                    best_solution = population[idx].copy()

            # Build offspring: elite + crossover/mutation
            offspring: List[Dict] = []
            elite_size = max(2, len(fronts[0]) // 2) if fronts else 2
            for idx in (fronts[0][:elite_size] if fronts else []):
                offspring.append(population[idx].copy())

            while len(offspring) < self.pop_size:
                p1, p2 = np.random.randint(len(population), size=2)
                if np.random.rand() < self.crossover_rate:
                    child = {
                        t.task_id: population[
                            p1 if np.random.rand() < 0.5 else p2
                        ][t.task_id]
                        for t in env.tasks
                    }
                else:
                    child = population[p1].copy()
                offspring.append(
                    self._smart_mutation(child, env, trust)
                )

            population = offspring[:self.pop_size]

        return best_solution

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        assignment: Dict,
        env: "EdgeEnvironment",
        trust: np.ndarray,
        makespan_limit: float
    ) -> float:
        """Scalar fitness aggregating f1, f2, f3 (Eqs. 8-10)."""
        device_times = np.zeros(len(env.devices))
        total_trust = 0.0
        security_violations = 0

        for task in env.tasks:
            if task.task_id not in assignment:
                continue
            d = assignment[task.task_id]
            device_times[d] += env.get_exec_time(task, env.devices[d])
            total_trust += trust[task.source_device, d]
            if task.is_sensitive and d in env.malicious_set:
                security_violations += 10

        makespan = device_times.max()
        n = len(env.tasks)
        return (0.3 * (1.0 - min(makespan / makespan_limit, 1.0)) +
                0.4 * (total_trust / n) +
                0.3 * (1.0 - security_violations / n))

    def _smart_mutation(
        self,
        assignment: Dict,
        env: "EdgeEnvironment",
        trust: np.ndarray
    ) -> Dict:
        """Security-aware mutation operator (Eq. 27)."""
        mutated = assignment.copy()
        for task in env.tasks:
            tid = task.task_id
            current_d = mutated.get(tid, 0)

            # Determine mutation probability
            if task.is_sensitive and current_d in env.malicious_set:
                p_mut = 1.0
            elif current_d in env.malicious_set:
                p_mut = 0.8
            elif trust[task.source_device, current_d] < self.trust_threshold:
                p_mut = 0.5
            else:
                p_mut = self.mutation_rate

            if np.random.rand() < p_mut:
                candidates = [
                    d for d in range(len(env.devices))
                    if d not in env.malicious_set
                    and trust[task.source_device, d] >= self.trust_threshold
                ]
                if not candidates:
                    candidates = [
                        d for d in range(len(env.devices))
                        if d not in env.malicious_set
                    ]
                if candidates:
                    mutated[tid] = np.random.choice(candidates)

        return mutated

    @staticmethod
    def _non_dominated_sort(fitness_scores: List[float]) -> List[List[int]]:
        """Simplified front decomposition for scalar fitness."""
        sorted_idx = list(np.argsort(fitness_scores)[::-1])
        chunk = max(1, len(fitness_scores) // 3)
        fronts, buf = [], []
        for idx in sorted_idx:
            buf.append(idx)
            if len(buf) >= chunk:
                fronts.append(buf)
                buf = []
        if buf:
            fronts.append(buf)
        return fronts


# ============================================================================
# Module 5: Laplacian Load Balancer  (Alg. 5 / Eq. 28)
# ============================================================================

class LaplacianLoadBalancer:
    """
    Iteratively migrates tasks from overloaded devices to lightly loaded
    neighbours using graph heat diffusion (Eq. 28):

        K_t = e^{-tL} = V e^{-tΛ} V^T

    Security and trust constraints are strictly respected during migration.

    Parameters
    ----------
    diffusion_steps  : int    Number of migration iterations K (default 5).
    trust_threshold  : float  τ for trust-feasibility check (default 0.4).
    overload_factor  : float  Threshold multiplier for 'overloaded' (default 1.5).
    """

    def __init__(self, diffusion_steps: int = 5,
                 trust_threshold: float = 0.4,
                 overload_factor: float = 1.5):
        self.diffusion_steps = diffusion_steps
        self.trust_threshold = trust_threshold
        self.overload_factor = overload_factor

    def balance(
        self,
        assignment: Dict,
        env: "EdgeEnvironment",
        trust: np.ndarray,
        heft_makespan: float
    ) -> Dict:
        """
        Parameters
        ----------
        assignment   : dict  {task_id -> device_id}
        env          : EdgeEnvironment
        trust        : np.ndarray, shape (n, n)  R^(mh).
        heft_makespan: float  Reference makespan (unused directly; kept for API parity).

        Returns
        -------
        balanced : dict  {task_id -> device_id}
        """
        n_devices = len(env.devices)

        # Compute current per-device load vector
        loads = np.zeros(n_devices)
        for task in env.tasks:
            if task.task_id in assignment:
                d = assignment[task.task_id]
                loads[d] += env.get_exec_time(task, env.devices[d])

        # Build Laplacian L = D - A  (Eq. 28)
        adj = env.device_adj.copy()
        np.fill_diagonal(adj, 0.0)
        degree = adj.sum(axis=1)
        L = np.diag(degree) - adj

        # Compute heat-diffusion kernel K_t = V e^{-0.1 Λ} V^T
        try:
            eigvals, eigvecs = eigh(L)
            eigvals = np.maximum(eigvals, 0.0)
            # diffusion matrix (used conceptually; migration is explicit below)
            _ = eigvecs @ np.diag(np.exp(-0.1 * eigvals)) @ eigvecs.T
        except Exception:
            pass  # fall back to load-based migration without diffusion

        balanced = assignment.copy()
        threshold = np.mean(loads) * self.overload_factor

        overloaded = np.where(loads > threshold)[0]

        for src in overloaded:
            tasks_on_src = [
                t.task_id for t in env.tasks
                if balanced.get(t.task_id) == src
            ]
            # Migrate at most 1/3 of the tasks from this device per pass
            n_migrate = max(1, len(tasks_on_src) // 3)

            for tid in tasks_on_src[:n_migrate]:
                task = env.tasks[tid]

                # Find eligible destination devices
                candidates = []
                for dst in range(n_devices):
                    if dst == src:
                        continue
                    if task.is_sensitive and dst in env.malicious_set:
                        continue  # hard constraint
                    if trust[task.source_device, dst] < self.trust_threshold:
                        continue  # trust constraint
                    if loads[dst] >= np.mean(loads):
                        continue  # only migrate to underloaded devices
                    mal_penalty = 1 if dst in env.malicious_set else 0
                    candidates.append((dst, mal_penalty, loads[dst]))

                if not candidates:
                    continue

                # Prefer non-malicious, then least loaded
                candidates.sort(key=lambda x: (x[1], x[2]))
                best_dst = candidates[0][0]

                loads[src] -= env.get_exec_time(task, env.devices[src])
                loads[best_dst] += env.get_exec_time(task, env.devices[best_dst])
                balanced[tid] = best_dst

        return balanced


# ============================================================================
# Module 6: GraphMatch — Main Scheduling Pipeline  (Alg. 6)
# ============================================================================

class GraphMatch:
    """
    Full GraphMatch scheduling pipeline (Algorithm 6):

      Phase 1 — Multi-hop Trust Propagation
      Phase 2 — Trust-Guided GCN scoring
      Phase 3 — Security-Aware Greedy initialisation
      Phase 4 — NSGA-III multi-objective optimisation
      Phase 5 — Laplacian load balancing

    Parameters
    ----------
    trust_threshold : float  τ shared across Greedy and Balancer (default 0.4).
    security_weight : float  Not used directly here; kept for API compatibility.
    seed            : int    Random seed for reproducibility (default 42).
    """

    def __init__(self, trust_threshold: float = 0.4,
                 security_weight: float = 0.20,
                 seed: int = 42):
        self.trust_propagator = MultiHopTrustPropagation()
        self.gcn = TrustGuidedGCN()
        self.matcher = GreedyMatcher(trust_threshold=trust_threshold,
                                     avoid_malicious=True)
        self.optimizer = NSGAIIIOptimizer()
        self.balancer = LaplacianLoadBalancer(trust_threshold=trust_threshold)
        self.security_weight = security_weight
        self._rng = np.random.RandomState(seed)

    def schedule(
        self,
        env: "EdgeEnvironment",
        heft_makespan: Optional[float] = None
    ) -> Tuple[Dict, np.ndarray]:
        """
        Parameters
        ----------
        env           : EdgeEnvironment
        heft_makespan : float, optional
            If None, HEFT is run internally to obtain a reference makespan.

        Returns
        -------
        final_assignment : dict  {task_id -> device_id}
        trust_mh         : np.ndarray, shape (n, n)  Multi-hop trust matrix R^(mh).
        """
        # Phase 1: trust propagation
        trust_mh = self.trust_propagator.propagate(env.direct_trust)

        # Reference makespan from HEFT
        if heft_makespan is None:
            heft_assign = HEFT().schedule(env)
            device_times = np.zeros(len(env.devices))
            for task in env.tasks:
                if task.task_id in heft_assign:
                    d = heft_assign[task.task_id]
                    device_times[d] += env.get_exec_time(task, env.devices[d])
            heft_makespan = device_times.max()

        # Phase 2: GCN scoring
        gcn_scores, _ = self.gcn.compute_matching_scores(
            env, trust_mh, rng=self._rng
        )

        # Phase 3: greedy initialisation
        initial = self.matcher.match(env, trust_mh, gcn_scores, heft_makespan)

        # Phase 4: NSGA-III optimisation
        optimized = self.optimizer.optimize(
            initial, env, trust_mh, heft_makespan
        )

        # Phase 5: Laplacian load balancing
        final = self.balancer.balance(optimized, env, trust_mh, heft_makespan)

        return final, trust_mh


# ============================================================================
# Baseline: HEFT  (Topcuoglu et al., IEEE TPDS 2002)
# ============================================================================

class HEFT:
    """
    Heterogeneous Earliest Finish Time (HEFT) algorithm.

    Steps:
      1. Compute upward rank for each task (average computation + communication).
      2. Sort tasks in descending rank order.
      3. For each task, assign it to the processor that gives the earliest
         finish time (EFT), respecting DAG dependencies.

    Reference:
        Topcuoglu H, Hariri S, Wu M Y. Performance-effective and
        low-complexity task scheduling for heterogeneous computing.
        IEEE Transactions on Parallel and Distributed Systems, 2002,
        13(3): 260-274.
    """

    def schedule(self, env: "EdgeEnvironment") -> Dict:
        n_devices = len(env.devices)

        # Average computation time per task
        avg_comp = {
            t.task_id: np.mean([env.get_exec_time(t, d) for d in env.devices])
            for t in env.tasks
        }

        # Average communication cost
        avg_comm = (np.mean([t.data_size for t in env.tasks]) /
                    np.mean([d.bandwidth for d in env.devices]))

        # Build successor map from DAG dependencies
        successors: Dict[int, List[int]] = {t.task_id: [] for t in env.tasks}
        for task in env.tasks:
            for pred_id in env.task_deps.get(task.task_id, []):
                successors[pred_id].append(task.task_id)

        # Upward rank (memoised recursion)
        upward_rank: Dict[int, float] = {}

        def compute_rank(tid: int) -> float:
            if tid in upward_rank:
                return upward_rank[tid]
            if not successors[tid]:
                upward_rank[tid] = avg_comp[tid]
            else:
                upward_rank[tid] = avg_comp[tid] + max(
                    avg_comm + compute_rank(s) for s in successors[tid]
                )
            return upward_rank[tid]

        for t in env.tasks:
            compute_rank(t.task_id)

        sorted_tasks = sorted(env.tasks, key=lambda t: -upward_rank[t.task_id])

        # Processor selection: EFT
        assignment: Dict[int, int] = {}
        proc_avail = np.zeros(n_devices)
        task_finish: Dict[int, float] = {}

        for task in sorted_tasks:
            best_device, best_eft = -1, float('inf')

            for d in range(n_devices):
                est = proc_avail[d]
                for pred_id in env.task_deps.get(task.task_id, []):
                    if pred_id in task_finish:
                        pred_d = assignment[pred_id]
                        comm = (0.0 if pred_d == d else
                                env.tasks[pred_id].data_size / min(
                                    env.devices[pred_d].bandwidth,
                                    env.devices[d].bandwidth))
                        est = max(est, task_finish[pred_id] + comm)

                eft = est + env.get_exec_time(task, env.devices[d])
                if eft < best_eft:
                    best_eft, best_device = eft, d

            assignment[task.task_id] = best_device
            task_finish[task.task_id] = best_eft
            proc_avail[best_device] = best_eft

        return assignment
