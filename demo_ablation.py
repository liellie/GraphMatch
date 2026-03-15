from edge_sim import EdgeEnvironment, evaluate_assignment
from graphmatch_core import MultiHopTrustPropagation, TrustGuidedGCNLogic, GraphMatchOptimizer, LaplacianBalancer
import numpy as np

def run_ablation_demo():
    env = EdgeEnvironment(seed=42)
    tp, gcn, opt, lb = MultiHopTrustPropagation(), TrustGuidedGCNLogic(), GraphMatchOptimizer(), LaplacianBalancer()
    
    # 1. HEFT (Baseline 模拟)
    assign_base = {t.task_id: np.random.randint(env.n_devices) for t in env.tasks}
    res_base = evaluate_assignment(assign_base, env, env.direct_trust)
    
    # 2. Full GraphMatch
    trust_mh = tp.propagate(env.direct_trust)
    scores = gcn.compute_scores(env, trust_mh)
    assign_gm = {t.task_id: np.argmax(scores[t.task_id]) for t in env.tasks}
    assign_gm = opt.smart_mutation(assign_gm, env)
    assign_gm = lb.balance(assign_gm, env)
    res_gm = evaluate_assignment(assign_gm, env, trust_mh)

    print(f"{'Method':<20} | {'Trust':<8} | {'Security'}")
    print("-" * 45)
    print(f"{'HEFT':<20} | {res_base['avg_trust']:.4f} | {res_base['security_score']:.1f}%")
    print(f"{'GraphMatch':<20} | {res_gm['avg_trust']:.4f} | {res_gm['security_score']:.1f}%")

if __name__ == "__main__":
    run_ablation_demo()
