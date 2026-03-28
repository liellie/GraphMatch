# GraphMatch

**GraphMatch: A Graph-Based Trust-Aware Framework for Secure Multi-Objective Task Scheduling in Heterogeneous Edge Computing**

> This repository provides the open-source simulator and core scheduling modules associated with the paper submitted to *Future Generation Computer Systems* (FGCS-D-26-00370).

---

## Overview

Edge computing environments face a fundamental tension: task scheduling must simultaneously optimise *performance* (low makespan), *trustworthiness* (avoid low-trust devices), and *security* (never assign sensitive tasks to malicious nodes). GraphMatch addresses all three objectives through a four-stage pipeline:

```
Direct Trust Matrix R
        │
        ▼
┌─────────────────────────┐
│ Stage 1                 │  Multi-Hop Trust Propagation
│ R  →  R^(mh)            │  Expands sparse trust coverage (~75% gain)
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Stage 2                 │  Trust-Guided GCN
│ Score matrix  S         │  Fuses 5 scoring dimensions per task-device pair
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Stage 3                 │  Security-Aware Greedy Matcher
│ Initial assignment      │  Hard constraint: sensitive tasks ≠ malicious devices
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Stage 4                 │  NSGA-III + Laplacian Load Balancing
│ Final assignment A*     │  Pareto optimisation + spectral workload diffusion
└─────────────────────────┘
```

---

## Repository Structure

```
GraphMatch/
├── edge_sim.py          # EdgeComputingEnv simulator (4-layer architecture)
├── graphmatch_core.py   # All GraphMatch scheduling modules
├── demo_ablation.py     # Ablation study demo reproducing Table 6
└── README.md
```

### `edge_sim.py` — EdgeComputingEnv Simulator

Implements the four-layer simulator described in Section 5.1.2 of the paper:

| Layer | Responsibility |
|---|---|
| **Configuration** | Accepts `n_devices`, `n_tasks`, `malicious_ratio`, `sensitive_ratio` |
| **Entity Generation** | Creates heterogeneous devices (4 types) and tasks (4 categories) |
| **Relationship Modeling** | Small-world topology, trust matrix, DAG dependencies |
| **Interface** | Standardised API: query devices/tasks, access adjacency/trust matrices |

Key classes and functions:

- `Device` — edge device with `cpu_capacity`, `memory`, `bandwidth`, `power_coefficient`, `device_type`
- `Task` — compute task with `cpu_cycles`, `memory_req`, `data_size`, `source_device`, `is_sensitive`
- `EdgeEnvironment` — main simulator class
- `evaluate_assignment(assignment, env, trust)` — computes trust score, makespan, security score, Jain index, and more

### `graphmatch_core.py` — Scheduling Modules

| Class | Paper Section | Description |
|---|---|---|
| `MultiHopTrustPropagation` | §4.2.1 / Eq. 17 | Iterative trust expansion up to H hops |
| `TrustGuidedGCN` | §4.2.2 / Eqs. 18–26 | Five-score matching matrix computation |
| `GreedyMatcher` | §4.2.3 | Security-aware greedy initialiser |
| `NSGAIIIOptimizer` | §4.2.4 / Eq. 27 | NSGA-III with security-aware mutation |
| `LaplacianLoadBalancer` | §4.2.5 / Eq. 28 | Spectral heat-diffusion load balancing |
| `GraphMatch` | §4.2.6 / Alg. 6 | Full five-stage scheduling pipeline |
| `HEFT` | Baseline | Topcuoglu et al., IEEE TPDS 2002 |

### `demo_ablation.py` — Ablation Study

Reproduces Table 6 from the paper by incrementally enabling modules:

```
HEFT (Baseline)  →  +Trust-Guided GCN  →  +Multi-hop Trust
                 →  +NSGA-III          →  Full GraphMatch
```

Also prints the trust coverage analysis corresponding to Proposition 1.

---

## Requirements

```
Python  >= 3.10
numpy   >= 1.24
scipy   >= 1.10
```

Install dependencies:

```bash
pip install numpy scipy
```

---

## Quick Start

### Run the ablation demo

```bash
python demo_ablation.py
```

Expected output (seed=42, 100 devices, 80 tasks, 15% malicious, 20% sensitive):

```
Trust Coverage Analysis  (cf. Proposition 1)
  Initial coverage  C(R^(0)) : 0.628
  After 3-hop propagation    : 0.903
  Theoretical lower bound    : 0.949
  Observed improvement       : +27.5%

Ablation Study Results  (cf. Table 6 in the paper)
Configuration             |    Trust | Eff.Makespan |   Security |  ToMalicious |  JainIndex
HEFT (Baseline)           |   0.5724 |         8.01 |      83.8% |           13 |      0.258
+Trust-Guided GCN         |   0.9109 |         1.72 |     100.0% |            0 |      0.406
+Multi-hop Trust          |   0.9015 |         1.88 |     100.0% |            0 |      0.406
+NSGA-III                 |   0.8675 |         2.34 |     100.0% |            0 |      0.264
Full GraphMatch           |   0.8341 |         5.09 |     100.0% |            0 |      0.311

  GraphMatch vs HEFT: Trust +45.7%, Effective Makespan -36.4%
```

### Use GraphMatch in your own code

```python
from edge_sim import EdgeEnvironment, evaluate_assignment
from graphmatch_core import GraphMatch

# Create a simulation environment
env = EdgeEnvironment(
    n_devices=100,
    n_tasks=80,
    malicious_ratio=0.15,
    sensitive_ratio=0.20,
    seed=42,
)

# Run GraphMatch
gm = GraphMatch(trust_threshold=0.4, seed=42)
assignment, trust_mh = gm.schedule(env)

# Evaluate results
metrics = evaluate_assignment(assignment, env, trust_mh)
print(f"Trust Score   : {metrics['avg_trust']:.4f}")
print(f"Security      : {metrics['security_score']:.1f}%")
print(f"Eff. Makespan : {metrics['effective_makespan']:.2f}")
print(f"Jain Index    : {metrics['jain_index']:.3f}")
```

### Use individual modules

```python
from edge_sim import EdgeEnvironment
from graphmatch_core import MultiHopTrustPropagation, TrustGuidedGCN

env = EdgeEnvironment(seed=42)

# Multi-hop trust propagation only
tp = MultiHopTrustPropagation(max_hops=3, decay=0.85)
trust_mh = tp.propagate(env.direct_trust)

# GCN scoring only
import numpy as np
gcn = TrustGuidedGCN(hidden_dim=32, n_layers=2)
scores, _ = gcn.compute_matching_scores(
    env, trust_mh, rng=np.random.RandomState(42)
)
print(f"Score matrix shape: {scores.shape}")  # (n_tasks, n_devices)
```

---

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `max_hops` H | 3 | Maximum propagation hops in trust expansion |
| `decay` γ | 0.85 | Trust-decay factor per hop (Eq. 17) |
| `trust_threshold` τ | 0.4 | Minimum trust required for device eligibility |
| `pop_size` P | 50 | NSGA-III population size |
| `n_generations` G | 30 | NSGA-III generations |
| `mutation_rate` p_m | 0.15 | Base mutation probability |
| `hidden_dim` d | 32 | GCN embedding dimension |
| `n_layers` L | 2 | Number of GCN message-passing layers |

---

## Notes on the GCN Embedding Score

The GCN embedding score `s_emb` in `TrustGuidedGCN` simulates the cosine similarity between learned task and device embeddings (Eq. 21). In this open-source release, the full PyTorch GCN training loop is replaced by a feature-based surrogate to keep external dependencies minimal (NumPy and SciPy only). The remaining four scoring dimensions — performance, trust, type-compatibility, and safety — are computed exactly as described in the paper (Eqs. 22–25).

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{li2026graphmatch,
  title   = {{GraphMatch}: A Graph-Based Trust-Aware Framework for Secure
             Multi-Objective Task Scheduling in Heterogeneous Edge Computing},
  author  = {Li, Wenjuan and Zhang, Qifei and Yang, Dingyu and
             Pan, Chengjie and Wang, Ben and Deng, Shuiguang},
  journal = {Future Generation Computer Systems},
  year    = {2026},
  note    = {Under review, Manuscript No.\ FGCS-D-26-00370}
}
```

---

## License

This project is released for academic and research use. Please contact the corresponding author (liwenjuan@hznu.edu.cn) for commercial use enquiries.
