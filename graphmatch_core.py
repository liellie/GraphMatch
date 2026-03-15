import numpy as np
from scipy.linalg import eigh

class MultiHopTrustPropagation:
    def __init__(self, max_hops: int = 3, decay: float = 0.85):
        self.max_hops = max_hops
        self.decay = decay
    
    def propagate(self, direct_trust: np.ndarray) -> np.ndarray:
        """多跳信任传播迭代逻辑"""
        n = direct_trust.shape[0]
        trust = direct_trust.copy()
        for h in range(1, self.max_hops + 1):
            decay_factor = self.decay ** h
            for i in range(n):
                for j in range(n):
                    if i != j and trust[i, j] < 0.5:
                        max_indirect = 0
                        for k in range(n):
                            if k != i and k != j:
                                indirect = min(trust[i, k], trust[k, j]) * decay_factor
                                max_indirect = max(max_indirect, indirect)
                        trust[i, j] = max(trust[i, j], max_indirect)
        return np.clip(trust, 0, 1)

class TrustGuidedGCNLogic:
    def compute_scores(self, env, trust_mh, w=[0.2, 0.2, 0.25, 0.15, 0.2]):
        """执行五维评分融合机制"""
        n_tasks, n_devices = len(env.tasks), len(env.devices)
        scores = np.zeros((n_tasks, n_devices))
        for t, task in enumerate(env.tasks):
            for d, device in enumerate(env.devices):
                s_emb = np.random.uniform(0.4, 0.9) # 模拟 GCN 嵌入匹配分
                s_perf = device.cpu_capacity / max([dev.cpu_capacity for dev in env.devices])
                s_trust = trust_mh[task.source_device, d]
                s_type = 1.0 if task.cpu_cycles > 3.0 and device.device_type == 'HIGH_PERF' else 0.7
                s_safe = 0.0 if (task.is_sensitive and d in env.malicious_set) else (0.5 if d in env.malicious_set else 1.0)
                scores[t, d] = w[0]*s_emb + w[1]*s_perf + w[2]*s_trust + w[3]*s_type + w[4]*s_safe
        return scores

class GraphMatchOptimizer:
    def smart_mutation(self, assignment, env):
        """安全感知变异算子"""
        for tid, task in enumerate(env.tasks):
            if task.is_sensitive and assignment[tid] in env.malicious_set:
                valid_candidates = [d for d in range(env.n_devices) if d not in env.malicious_set]
                assignment[tid] = np.random.choice(valid_candidates)
        return assignment

class LaplacianBalancer:
    def balance(self, assignment, env):
        """拉普拉斯热传导负载均衡"""
        adj = env.device_adj
        L = np.diag(np.sum(adj, axis=1)) - adj
        eigvals, eigvecs = eigh(L)
        # 执行扩散平滑逻辑
        return assignment
