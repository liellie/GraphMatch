# GraphMatch
GraphMatch is a graph-based trust-aware framework for secure edge computing task scheduling. It features Multi-hop Trust Propagation , Trust-Guided GCN , NSGA-III for multi-objective optimization , and Laplacian Load Balancing. Includes the EdgeComputingEnv simulator for reproducible research.

# GraphMatch: Trust-Aware Task Scheduling Simulator

## Overview
This repository provides the core modules for **GraphMatch**, a graph-based framework for multi-objective task scheduling in edge computing. It includes a hierarchical simulator and a security-aware scheduling engine.

## Project Structure
- `edge_sim.py`: The **EdgeComputingEnv** simulator (Environment modeling).
- `graphmatch_core.py`: Core components (Multi-hop trust, GCN scores, Mutation, and Balancing).
- `demo_ablation.py`: Demonstration script for reproducing research results.

## Requirements
- Python 3.10+
- NumPy
- SciPy

## Quick Start
Run the ablation study demo:
```bash
python demo_ablation.py
