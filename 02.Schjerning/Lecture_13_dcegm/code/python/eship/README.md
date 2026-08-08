# Entrepreneurship with DC-EGM

This folder contains a small, self-contained lifecycle version of the entrepreneurship model in Chapter 2 of Schjerning's thesis. It is separate from the exercise classes and initially omits human capital so the numerical idea remains visible.

The household begins a period with cash-on-hand `m` and occupation `z` (worker or entrepreneur), and chooses saving `a` and next period's occupation `d`:

\[
V_t(m,z)=\max_d\max_a\left\{u(c)+\beta E[V_{t+1}(m',d)]\right\},
\qquad c=m-a/R-F(z,d).
\]

Next cash is `a + wage income` for a worker and `a + business profit` for an entrepreneur. Business capital is a static choice,

\[
k(a)=\min\left\{\lambda a,\left(\alpha\theta/r\right)^{1/(1-\alpha)}\right\}.
\]

## What the code does

For each possible next occupation, `egm.solve_choice`:

1. evaluates expected marginal continuation value on an asset grid;
2. inverts the Euler equation to obtain consumption and endogenous cash-on-hand;
3. applies an upper-envelope scan within that occupation.

`model.solve` then compares the two choice-specific values. Thus the implementation is standard EGM inside each branch and DC-EGM across worker/entrepreneur branches. Set `par.sigma_eta=0` for an exact maximum or a positive value for iid extreme-value taste shocks and smooth logit probabilities.

The baseline is deliberately a teaching calibration rather than an exact thesis replication. In particular, `beta=0.96` creates enough lifecycle saving for workers to cross the entry threshold. With the fixed random seed, simulated households enter at different ages after favorable wage histories and later exit as the finite horizon shortens. The entry-cost experiment compares `F_entry=0` with `F_entry=0.20`.

## Files

- `01_entrepreneurship_dcegm.ipynb`: main walkthrough and parameter laboratory.
- `model.py`: exercise-style class with `setup`, `create_grids`, `solve`, and `simulate`.
- `egm.py`: the short DC-EGM step, technology, and transitions.
- `tools.py`: grids, quadrature, interpolation, and the max/logsum operator.
- `run.py`: minimal command-line smoke run.
- `test_model.py`: feasibility, entry/exit, hysteresis, and taste-shock checks.

Run the notebook from this directory, or use:

```bash
python run.py
python -m unittest test_model.py
```
