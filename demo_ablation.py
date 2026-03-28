"""
demo_ablation.py — Ablation Study Demo
=======================================
Reproduces the core ablation results reported in Table 6 of:

  "GraphMatch: A Graph-Based Trust-Aware Framework for Secure Multi-Objective
   Task Scheduling in Heterogeneous Edge Computing"

Each configuration adds one GraphMatch module on top of the previous:

  HEFT (Baseline)
  → +Trust-Guided GCN
  → +Multi-hop Trust
  → +NSGA-III
  → Full GraphMatch  (all four modules)

Usage
-----
    python demo_ablation.py

Requirements: numpy, scipy
"""

import numpy as np
from typing import Dict

from edge_sim import EdgeEnvironment, evaluate_assignment
from graphmatch_core import (
    MultiHopTrustPropagation,
    TrustGuidedGCN,
    GreedyMatcher,
    NSGAIIIOptimizer,
    LaplacianLoadBalancer,
    GraphMatch,
    HEFT,
)


# ============================================================================
# Ablation Study
# ============================================================================

def run_ablation_study(env: EdgeEnvironment) -> Dict[str, Dict]:
    """
    Incrementally enable GraphMatch modules and record performance.

    Returns
    -------
    results : dict  {configuration_name -> metrics_dict}
    """
    results: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # Step 0: HEFT baseline
    # ------------------------------------------------------------------
    heft = HEFT()
    heft_assign = heft.schedule(env)
    results['HEFT (Baseline)'] = evaluate_assignment(
        heft_assign, env, env.direct_trust
    )
    heft_makespan = results['HEFT (Baseline)']['raw_makespan']

    # ------------------------------------------------------------------
    # Step 1: +Trust-Guided GCN  (direct trust, no multi-hop)
    # ------------------------------------------------------------------
    gcn = TrustGuidedGCN()
    rng = np.random.RandomState(42)
    gcn_scores_direct, _ = gcn.compute_matching_scores(
        env, env.direct_trust, rng=rng
    )
    loads = np.zeros(len(env.devices))
    gcn_assign: Dict[int, int] = {}
    for task in sorted(env.tasks, key=lambda t: (-int(t.is_sensitive), -t.cpu_cycles)):
        best = max(
            range(len(env.devices)),
            key=lambda d: gcn_scores_direct[task.task_id, d] - 0.1 * loads[d]
        )
        gcn_assign[task.task_id] = best
        loads[best] += env.get_exec_time(task, env.devices[best])

    results['+Trust-Guided GCN'] = evaluate_assignment(
        gcn_assign, env, env.direct_trust
    )

    # ------------------------------------------------------------------
    # Step 2: +Multi-hop Trust Propagation
    # ------------------------------------------------------------------
    trust_prop = MultiHopTrustPropagation()
    trust_mh = trust_prop.propagate(env.direct_trust)

    rng2 = np.random.RandomState(42)
    gcn_scores_mh, _ = gcn.compute_matching_scores(env, trust_mh, rng=rng2)

    loads = np.zeros(len(env.devices))
    mh_assign: Dict[int, int] = {}
    for task in sorted(env.tasks, key=lambda t: (-int(t.is_sensitive), -t.cpu_cycles)):
        best = max(
            range(len(env.devices)),
            key=lambda d: gcn_scores_mh[task.task_id, d] - 0.1 * loads[d]
        )
        mh_assign[task.task_id] = best
        loads[best] += env.get_exec_time(task, env.devices[best])

    results['+Multi-hop Trust'] = evaluate_assignment(
        mh_assign, env, trust_mh
    )

    # ------------------------------------------------------------------
    # Step 3: +NSGA-III  (greedy init + evolutionary optimisation)
    # ------------------------------------------------------------------
    matcher = GreedyMatcher()
    initial = matcher.match(env, trust_mh, gcn_scores_mh, heft_makespan)

    optimizer = NSGAIIIOptimizer()
    nsga_assign = optimizer.optimize(initial, env, trust_mh, heft_makespan)

    results['+NSGA-III'] = evaluate_assignment(nsga_assign, env, trust_mh)

    # ------------------------------------------------------------------
    # Step 4: Full GraphMatch  (+Laplacian load balancing)
    # ------------------------------------------------------------------
    balancer = LaplacianLoadBalancer()
    final_assign = balancer.balance(nsga_assign, env, trust_mh, heft_makespan)

    results['Full GraphMatch'] = evaluate_assignment(final_assign, env, trust_mh)

    return results


# ============================================================================
# Reporting
# ============================================================================

def print_ablation_table(results: Dict[str, Dict]) -> None:
    """Print an ASCII table matching the structure of Table 6 in the paper."""
    configs = [
        'HEFT (Baseline)',
        '+Trust-Guided GCN',
        '+Multi-hop Trust',
        '+NSGA-III',
        'Full GraphMatch',
    ]

    header = (f"{'Configuration':<25} | {'Trust':>8} | "
              f"{'Eff.Makespan':>12} | {'Security':>10} | "
              f"{'ToMalicious':>12} | {'JainIndex':>10}")
    sep = '-' * len(header)

    print()
    print("=" * len(header))
    print("  Ablation Study Results  (cf. Table 6 in the paper)")
    print("=" * len(header))
    print(header)
    print(sep)

    for cfg in configs:
        if cfg not in results:
            continue
        r = results[cfg]
        print(
            f"{cfg:<25} | "
            f"{r['avg_trust']:>8.4f} | "
            f"{r['effective_makespan']:>12.2f} | "
            f"{r['security_score']:>9.1f}% | "
            f"{r['to_malicious']:>12d} | "
            f"{r['jain_index']:>10.3f}"
        )

    print(sep)
    print()
    gm = results.get('Full GraphMatch', {})
    base = results.get('HEFT (Baseline)', {})
    if gm and base:
        trust_gain = (gm['avg_trust'] - base['avg_trust']) / base['avg_trust'] * 100
        ms_drop = (base['effective_makespan'] - gm['effective_makespan']) / \
                   base['effective_makespan'] * 100
        print(f"  GraphMatch vs HEFT: Trust +{trust_gain:.1f}%,"
              f" Effective Makespan -{ms_drop:.1f}%")
    print()


# ============================================================================
# Trust coverage verification
# ============================================================================

def print_trust_coverage(env: EdgeEnvironment) -> None:
    """Show the trust coverage expansion from Proposition 1."""
    rho0 = 0.5
    n = env.n_devices

    def coverage(R: np.ndarray) -> float:
        mask = (R > rho0)
        np.fill_diagonal(mask, False)
        return mask.sum() / (n * (n - 1))

    direct_cov = coverage(env.direct_trust)

    tp = MultiHopTrustPropagation()
    trust_mh = tp.propagate(env.direct_trust)
    mh_cov = coverage(trust_mh)

    H = tp.max_hops
    theoretical_lb = 1.0 - (1.0 - direct_cov) ** H

    print("Trust Coverage Analysis  (cf. Proposition 1)")
    print(f"  Initial coverage  C(R^(0)) : {direct_cov:.3f}")
    print(f"  After {H}-hop propagation  : {mh_cov:.3f}")
    print(f"  Theoretical lower bound   : {theoretical_lb:.3f}")
    print(f"  Observed improvement      : +{(mh_cov - direct_cov)*100:.1f}%")
    print()


# ============================================================================
# Entry point
# ============================================================================

if __name__ == '__main__':
    print("Initialising EdgeComputingEnv (n_devices=100, n_tasks=80, "
          "malicious=15%, sensitive=20%, seed=42) …")
    env = EdgeEnvironment(
        n_devices=100,
        n_tasks=80,
        malicious_ratio=0.15,
        sensitive_ratio=0.20,
        seed=42,
    )

    print_trust_coverage(env)

    print("Running ablation study (this may take ~30 s) …")
    results = run_ablation_study(env)
    print_ablation_table(results)
