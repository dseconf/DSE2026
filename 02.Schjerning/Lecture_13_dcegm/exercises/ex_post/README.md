# Lecture 13 exercises: EGM and discrete-continuous EGM

This is the **completed reference version** of the exercises. The student version, with selected calculations left incomplete, is available in [`../ex_ante`](../ex_ante/).

The exercises progress from a standard consumption-saving problem to a life-cycle buffer-stock model, structural estimation, and a discrete-continuous retirement model.

## Recommended order

| Notebook | Main purpose |
|---|---|
| [`01_time_iteration.ipynb`](01_time_iteration.ipynb) | Solve Deaton's consumption-saving model by time iteration, simulate it, and calculate Euler errors. |
| [`02_EGM.ipynb`](02_EGM.ipynb) | Introduce EGM, compare policies with time iteration, study the liquidity constraint, and run a transparent flat-path accuracy test. |
| [`03_buffer_stock_egm.ipynb`](03_buffer_stock_egm.ipynb) | Solve and simulate the normalized life-cycle buffer-stock model; compare loop and vectorized EGM; explore comparative statics. |
| [`04_estimate_buffer_stock.ipynb`](04_estimate_buffer_stock.ipynb) | Estimate preference parameters by maximum likelihood and the method of simulated moments. |
| [`05_dc_egm.ipynb`](05_dc_egm.ipynb) | Solve a discrete-continuous work/retirement model, study choice thresholds and taste-shock smoothing, and inspect an optional multidimensional extension. |

The notebooks are designed to be run in numerical order. In particular, notebooks 03 and 04 use the same buffer-stock model and notebook 04 repeatedly solves it inside an estimator.

## Running the exercises

Start Jupyter from this directory so that local imports such as `import Exercise_2` and `from model import model_bufferstock` resolve correctly:

```bash
cd exercises/ex_post
jupyter lab
```

The code uses NumPy, SciPy, Matplotlib, and Jupyter/IPython. Run notebook cells from top to bottom after restarting the kernel. The first cell enables automatic reloading of edited Python modules.

## File map

| Files | Role |
|---|---|
| `Exercise_1.py` | Time-iteration solution, simulation, and Euler-error calculations for notebooks 01 and 02. |
| `Exercise_2.py` | Loop and vectorized EGM implementations for notebook 02. |
| `model.py` | Setup, grids, solution, and simulation for the life-cycle buffer-stock model. |
| `egm.py` | Loop and vectorized EGM steps for the buffer-stock model. |
| `utility.py` | Utility, marginal utility, and inverse marginal utility for the buffer-stock model. |
| `estimate.py` | Likelihood, simulated moments, and optimization routines used in notebook 04. |
| `model_dc.py`, `egm_dc.py` | One-dimensional discrete-continuous work/retirement model. |
| `model_dc_multidim.py`, `egm_dc_multidim.py` | Optional retirement model with permanent income as an additional state. |
| `tools.py` | Quadrature, nonlinear grids, interpolation, and other shared numerical utilities. |

## Buffer-stock workflow

For the life-cycle model, use the following order:

```python
model = model_bufferstock()
model.life_cycle_setup()

# Change economic parameters here.
model.par.beta = 0.98

model.create_grids()
model.solve()
model.simulate()
```

Change parameters **after** `life_cycle_setup()` and **before** `create_grids()`. Shock nodes, probabilities, borrowing limits, and asset grids are derived from the calibration and must be rebuilt after changing parameters such as `sigma_xi`, `sigma_psi`, or `lambdaa`.

## Numerical and economic interpretation

- EGM and time iteration generally use different native state grids. Compare their policy functions only after interpolating them onto a common grid.
- The flat-path experiment in notebook 02 is a diagnostic with deterministic income and $\beta R=1$. It is deliberately separate from the stochastic economic baseline.
- In the liquidity-constrained region, $A_t=M_t-C_t=0$, so the consumption policy lies on the 45-degree line $C_t=M_t$.
- The grid-refinement experiment shows that time iteration also becomes extremely accurate with a sufficiently fine grid; EGM reaches high accuracy with fewer policy points.
- In the retirement model, `z_plus=0` means **working next period**, while `z_plus=1` means **retired next period**. Retirement is absorbing.
- `sigma_eta` is the scale of the taste shocks. Setting it to zero gives a sharp maximum over choices; positive values smooth the decision through the logsum.

## Parameter experiments

Notebook 03 contains an optional laboratory for varying:

- `beta`: patience and asset accumulation;
- `rho`: risk aversion and the inverse of the intertemporal elasticity of substitution;
- `sigma_psi`: permanent-income risk and precautionary saving.

Change one parameter at a time initially. Use common random numbers when comparing simulated profiles, and rebuild the grids before solving.

## Additional lecture material

- [Lecture slides](../../13_dcegm.pdf)
- [MATLAB demonstration (`run_dse.m`)](../../code/matlab/run_dse.m)

The MATLAB demonstration covers many of the same numerical comparisons. The Python notebooks are intended for hands-on experimentation and extension.
