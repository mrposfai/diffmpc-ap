# Nonlinear Model Predictive Control with Differentiable Cost and Dynamics in Artificial Pancreas

> **⚠️ Warning**: The package `py-mgipsim` must be installed to run `lin_vs_nonlin_mpc_test.py`. 
> Install it from: [https://github.com/illinoistech-itm/py-mgipsim](https://github.com/illinoistech-itm/py-mgipsim)

## Files
#### Main simulation and analysis scripts
- `lin_vs_nonlin_mpc_test.py` - Main benchmarking script comparing three MPC controllers:
  - Nonlinear MPC with finite-difference gradients (SciPy L-BFGS-B)
  - Nonlinear MPC with automatic differentiation (JAX/jaxopt)
  - Linear MPC with quadratic programming (QP solver)
  - Tests controllers on virtual patient cohort
  - Saves results to `data/` directory as `.npy` files

- `plot_errors.py` - Visualization script:
  - Creates box plots comparing controller tracking errors
  - Generates publication-ready figures

- `states_eval.py` - Additional state trajectory analysis and visualization:
  - Plots and detailed state trajectory comparisons
  - Time-series analysis of glucose and insulin dynamics
  - Model analysis with statsmodels

#### Model implementations
- `nmpc.py` - Nonlinear Hovorka glucose-insulin dynamics model:
  - 13-state physiological model for Type 1 Diabetes
  - Numba-JIT compiled for performance
  - Includes nonlinear insulin sensitivity mapping (hypoglycemia/hyperglycemia)
  - Used by SciPy-based nonlinear MPC optimizer

- `nmpc_jax.py` - JAX implementation of Hovorka model:
  - Same 13-state dynamics as `nmpc.py` but with JAX primitives
  - Enables automatic differentiation for gradient-based optimization
  - JIT-compiled for GPU/TPU acceleration
  - Supports `jax.lax.scan` for efficient trajectory rollouts

- `mpc.py` - Linear MPC implementation:
  - Builds QP formulation from linearized Hovorka model
  - Uses `qpsolvers` library with Clarabel solver
  - Prediction horizon optimization with box constraints
  - Includes utility functions for Hovorka model linearization

#### Configuration
- `OPT_GLOBALS.py` - Global optimization parameters:
  - Time step (TS = 1 minute)
  - Prediction horizon (PRED_LEN = 180 minutes)
  - Glucose setpoint (SETPOINT = 110 mg/dL)
  - Optimization tolerances and iteration limits
  - Input rate constraints (MAX_RATE = 1000 mU/min)
  - Weight matrices (RU = 1E-4 for insulin penalty)

#### Data directories
- `SimulationResults/` - Virtual patient simulation data:
  - `gri_fit/model.pkl` - Pickled model data with patient parameters and initial states
  
- `data/` - Generated results from benchmark simulations:
  - `saved_states_nl_TS.npy` - simulation states from Finite diff nmpc with QP mpc insulins, TS is the sampling time.
  - `saved_states_l_TS.npy` - simulation states from QP mpc, TS is the sampling time.
  - `saved_states_nlnl_TS.npy` - simulation states from Finite diff nmpc, TS is the sampling time.
  - `saved_states_jax_TS.npy` - simulation states from Auto diff nmpc, TS is the sampling time.
  - `failed_sim_states.npy` - states from failed simulations
  - `failed_sim_insulins.npy` - insulins from failed simulations
  - `results_lin.npy` - optimized insulins by QP mpc
  - `results_nl.npy` - optimized insulins by Finite diff nmpc
  - `results_jax.npy` - optimized insulins by Auto diff nmpc
  - `costs.npy` - optimization costs for all controllers
  - `lin_mod_errors.npy` - linear model errors compared to the nonlinear model
  - `time_of_optimizations.npy` - time taken for optimizations for all controllers
  - `nit_of_optimizations.npy` - number of iterations for optimizations for nonlinear controllers
