# Implicit Differentiation for Optimal Control (IDOC)
This repository contains a reference implementation for the paper "Revisiting Implicit Differentiation for Learning Problems in Optimal Control" (NeurIPS 2023) for the settings with and without inequality constraints in the forward optimal control problem. 

Our method, named "Implicit Differentiation for Optimal Control" (IDOC) is used to __differentiate through optimal control problems__ (sensitivity analysis), and improves on previous methods such as [DiffMPC](https://github.com/locuslab/differentiable-mpc), [PDP](https://github.com/wanxinjin/Pontryagin-Differentiable-Programming) and [Safe-PDP](https://github.com/wanxinjin/Safe-PDP). IDOC can handle (smooth) optimal control problems which are

* Non-convex (non-linear dynamics and non-convex cost and constraint functions)
* Inequality constrained
* Contains non-linear equality constraints in addition to dynamics (such as terminal constraints)

For more details, see our [paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/bcfcf7232cb74e1ef82d751880ff835b-Paper-Conference.pdf). Our code integrates seamlessly with the Safe-PDP codebase.

## Trajectory Derivatives

We provide the implementation of our trajectory derivative computations within the `src/IDOC_eq.py` (no inequality constraints) and `src/IDOC_ineq.py` (inequality constraints) file. 

## Installation

IDOC makes use of [Safe-PDP](https://github.com/wanxinjin/Safe-PDP/tree/main), therefore it is necessary to clone this repository including all the submodules. 

```bash
git clone --recurse-submodules https://github.com/mingu6/Implicit-Diff-Optimal-Control.git
```

Then install the requirements:

```bash
pip install -r requirements.txt
```

The directory `examples` contains bi-level optimization problems solved using IDOC trajectory derivatives.
E.g.

```bash
python CIOC_Cartpole_IDOC.py -m full
```
