import numpy as np
import random
from dataclasses import dataclass
from typing import List, Dict 

@dataclass
class Device:
    device_id: int
    cpu_capacity: float
    memory: float
    bandwidth: float
    power_coefficient: float
    device_type: str

@dataclass
class Task:
    task_id: int
    cpu_cycles: float
    memory_req: float
    data_size: float
    deadline: float
    source_device: int
    is_sensitive: bool

class EdgeEnvironment:
    """
    EdgeComputingEnv: 实现配置层、实体生成层、关系建模层及接口层。
    """
    def __init__(self, n_devices: int = 100, n_tasks: int = 80,
                 malicious_ratio: float = 0.15, sensitive_ratio: float = 0.2,
                 seed: int = 42):
        self.n_devices = n_devices
        self.n_tasks = n_tasks
        self.malicious_ratio = malicious_ratio
        self.sensitive_ratio = sensitive_ratio
        
        np.random.seed(seed)
        random.seed(seed)
        
        # 实体生成
        self.devices = self._create_devices()
        self.tasks = self._create_tasks()
        self.malicious_set = self._create_malicious_set()
        
        # 拓扑与信任建模
        self.direct_trust = self._create_trust_matrix()
        self.device_adj = self._create_adjacency()
        
        # 任务依赖 (DAG)
        dag_rng = np.random.RandomState(seed + 1000)
        self.task_deps = self._create_dag_dependencies(dag_rng)
    
    def _create_devices(self) -> List[Device]:
        devices = []
        type_configs = {
            'HIGH_PERF': {'cpu': (8, 16), 'mem': (16, 32), 'bw': (100, 200), 'power': (1.5, 2.5), 'ratio': 0.20},
            'STANDARD': {'cpu': (4, 8), 'mem': (8, 16), 'bw': (50, 100), 'power': (1.0, 1.5), 'ratio': 0.40},
            'LOW_POWER': {'cpu': (1, 4), 'mem': (2, 8), 'bw': (20, 50), 'power': (0.3, 0.8), 'ratio': 0.25},
            'STORAGE': {'cpu': (2, 6), 'mem': (32, 64), 'bw': (80, 150), 'power': (0.8, 1.2), 'ratio': 0.15}
        }
        for i in range(self.n_devices):
            r = np.random.rand()
            cumsum = 0
            for dtype, cfg in type_configs.items():
                cumsum += cfg['ratio']
                if r <= cumsum:
                    devices.append(Device(i, np.random.uniform(*cfg['cpu']), np.random.uniform(*cfg['mem']),
                                       np.random.uniform(*cfg['bw']), np.random.uniform(*cfg['power']), dtype))
                    break
        return devices
    
    def _create_tasks(self) -> List[Task]:
        tasks = []
        for i in range(self.n_tasks):
            tasks.append(Task(i, np.random.uniform(0.5, 5.0), np.random.uniform(1, 32),
                           np.random.uniform(1, 50), np.random.uniform(1, 10),
                           np.random.randint(0, self.n_devices),
                           np.random.rand() < self.sensitive_ratio))
        return tasks
    
    def _create_malicious_set(self) -> set:
        n_malicious = int(self.n_devices * self.malicious_ratio)
        return set(np.random.choice(self.n_devices, n_malicious, replace=False))
    
    def _create_trust_matrix(self) -> np.ndarray:
        trust = np.random.uniform(0.3, 1.0, (self.n_devices, self.n_devices))
        np.fill_diagonal(trust, 1.0)
        for m in self.malicious_set:
            trust[:, m] *= np.random.uniform(0.3, 0.7)
        return trust
    
    def _create_adjacency(self) -> np.ndarray:
        adj = np.random.rand(self.n_devices, self.n_devices)
        adj = (adj + adj.T) / 2
        np.fill_diagonal(adj, 0)
        return adj
    
    def _create_dag_dependencies(self, rng=None) -> Dict[int, List[int]]:
        deps = {t.task_id: [] for t in self.tasks}
        for t in self.tasks[1:]:
            n_deps = (rng if rng else np.random).randint(0, min(3, t.task_id))
            if n_deps > 0:
                deps[t.task_id] = list((rng if rng else np.random).choice(t.task_id, n_deps, replace=False))
        return deps

    def get_exec_time(self, task: Task, device: Device) -> float:
        return task.cpu_cycles / device.cpu_capacity + task.data_size / device.bandwidth

    def get_energy(self, task: Task, device: Device) -> float:
        return self.get_exec_time(task, device) * device.power_coefficient * device.cpu_capacity

def evaluate_assignment(assignment: Dict, env: EdgeEnvironment, trust: np.ndarray) -> Dict:
    n_devices = len(env.devices)
    device_times = np.zeros(n_devices)
    total_trust, total_energy, to_malicious = 0, 0, 0
    sensitive_protected, total_sensitive = 0, 0

    for task in env.tasks:
        if task.task_id not in assignment:
            continue
        d = assignment[task.task_id]
        device_times[d] += env.get_exec_time(task, env.devices[d])
        total_trust += trust[task.source_device, d]
        total_energy += env.get_energy(task, env.devices[d])
        if d in env.malicious_set:
            to_malicious += 1
        if task.is_sensitive:
            total_sensitive += 1
            if d not in env.malicious_set:
                sensitive_protected += 1

    raw_makespan = device_times.max()
    effective_makespan = raw_makespan
    for task in env.tasks:
        if task.task_id in assignment:
            d = assignment[task.task_id]
            if d in env.malicious_set:
                exec_time = env.get_exec_time(task, env.devices[d])
                effective_makespan += exec_time * 10 if task.is_sensitive else exec_time * 0.8

    jain = ((np.sum(device_times) ** 2) /
            (n_devices * np.sum(device_times ** 2) + 1e-8)
            if np.sum(device_times) > 0 else 0.0)
    n_tasks = len(env.tasks)

    return {
        'avg_trust':            total_trust / n_tasks,
        'raw_makespan':         raw_makespan,
        'effective_makespan':   effective_makespan,
        'energy':               total_energy,
        'to_malicious':         to_malicious,
        'security_score':       (n_tasks - to_malicious) / n_tasks * 100,
        'sensitive_protection': (sensitive_protected / total_sensitive * 100
                                 if total_sensitive > 0 else 100.0),
        'jain_index':           jain,
        'load_std':             float(np.std(device_times)),
        'utilization':          float(np.sum(device_times > 0) / n_devices * 100),
    }
