"""Finite-horizon occupation and saving model, organized like the exercises."""

from types import SimpleNamespace
import numpy as np
try: from . import egm, tools
except ImportError: import egm, tools

WORKER, ENTREPRENEUR = 0, 1


class model_eship:
    """Thesis-inspired lifecycle model without human capital."""

    def __init__(self):
        self.par, self.sol, self.sim = SimpleNamespace(), SimpleNamespace(), SimpleNamespace()
        self.setup()

    def setup(self):
        """Set economic parameters and numerical defaults."""
        par = self.par
        # Lifecycle, preferences, and liquid saving
        par.T, par.age_min = 35, 25
        par.beta, par.R, par.r, par.rho = 0.96, 1.04, 0.04, 1.5
        # Worker income and entrepreneurial technology/risk
        par.wage, par.sigma_w = 1.0, 0.10
        par.theta, par.alpha, par.sigma_e, par.lambd = 0.80, 0.33, 0.30, 1.0
        # Switching costs and optional extreme-value taste shocks
        par.entry_cost, par.exit_cost, par.sigma_eta = 0.20, 0.0, 0.0
        # Numerical grids and simulation
        par.m_max, par.a_max, par.Nm, par.Na, par.Nxi = 14.0, 14.0, 240, 220, 5
        par.simN, par.seed = 20_000, 2026

    def create_grids(self):
        """Construct common cash/asset grids and income-shock quadrature."""
        par = self.par
        par.grid_m = tools.nonlinspace(1e-4, par.m_max, par.Nm, 1.80)
        par.grid_a = tools.nonlinspace(1e-5, par.a_max, par.Na, 1.80)
        par.xi_w, par.xi_w_w = tools.lognormal_gauss_hermite(par.sigma_w, par.Nxi)
        par.xi_e, par.xi_e_w = tools.lognormal_gauss_hermite(par.sigma_e, par.Nxi)

    def solve(self):
        """Backward induction: EGM within each occupation, then a discrete max/logsum."""
        par, sol = self.par, self.sol
        shape, choices = (par.T, 2, par.Nm), (par.T, 2, 2, par.Nm)
        sol.v = np.full(shape, np.nan); sol.mu = np.full(shape, np.nan)
        sol.c = np.full(shape, np.nan); sol.a = np.full(shape, np.nan); sol.d = np.zeros(shape, dtype=int)
        sol.prob = np.full(choices, np.nan); sol.v_choice = np.full(choices, np.nan)
        sol.c_choice = np.full(choices, np.nan); sol.a_choice = np.full(choices, np.nan)
        sol.m_endo = np.full((par.T-1, 2, 2, par.Na), np.nan); sol.c_endo = np.full_like(sol.m_endo, np.nan)

        # Terminal period: consume all cash, independently of current occupation.
        for z in range(2):
            sol.c[-1, z] = par.grid_m; sol.a[-1, z] = 0
            sol.v[-1, z] = egm.utility(par.grid_m, par); sol.mu[-1, z] = egm.marg_utility(par.grid_m, par)

        for t in reversed(range(par.T-1)):
            for z in range(2):
                for d in range(2):
                    m_raw, c_raw, v, c, a = egm.solve_choice(z, d, t, sol, par)
                    sol.m_endo[t, z, d], sol.c_endo[t, z, d] = m_raw, c_raw
                    sol.v_choice[t, z, d], sol.c_choice[t, z, d], sol.a_choice[t, z, d] = v, c, a
                sol.v[t, z], sol.prob[t, z] = tools.logsum(sol.v_choice[t, z], par.sigma_eta)
                sol.d[t, z] = np.argmax(sol.v_choice[t, z], axis=0)
                sol.c[t, z] = np.sum(sol.prob[t, z]*sol.c_choice[t, z], axis=0)
                sol.a[t, z] = np.sum(sol.prob[t, z]*sol.a_choice[t, z], axis=0)
                sol.mu[t, z] = np.sum(sol.prob[t, z]*egm.marg_utility(sol.c_choice[t, z], par), axis=0)

    def simulate(self):
        """Simulate lifecycle choices; sigma_eta>0 draws from the logit probabilities."""
        par, sol, sim = self.par, self.sol, self.sim
        rng = np.random.default_rng(par.seed)
        sim.m = np.full((par.T, par.simN), np.nan); sim.c = np.full_like(sim.m, np.nan)
        sim.a = np.full_like(sim.m, np.nan); sim.z = np.zeros((par.T, par.simN), dtype=int)
        sim.d = np.zeros_like(sim.z); sim.entry = np.zeros_like(sim.z, dtype=bool); sim.exit = np.zeros_like(sim.entry)
        sim.m[0] = rng.lognormal(np.log(1.6)-0.5*0.55**2, 0.55, par.simN)
        sw = np.sqrt(np.log1p(par.sigma_w**2)); se = np.sqrt(np.log1p(par.sigma_e**2))

        for t in range(par.T):
            for z in range(2):
                I = sim.z[t] == z
                if not np.any(I): continue
                m = sim.m[t, I]
                if t == par.T-1: sim.c[t, I] = m; continue
                p_e = tools.interp(par.grid_m, sol.prob[t, z, ENTREPRENEUR], m)
                d = (rng.random(np.sum(I)) < p_e).astype(int); sim.d[t, I] = d
                for choice in range(2):
                    J = d == choice
                    if not np.any(J): continue
                    idx = np.flatnonzero(I)[J]
                    sim.c[t, idx] = tools.interp(par.grid_m, sol.c_choice[t, z, choice], m[J])
                    sim.a[t, idx] = tools.interp(par.grid_m, sol.a_choice[t, z, choice], m[J])
                sim.entry[t, I] = (z == WORKER) & (d == ENTREPRENEUR)
                sim.exit[t, I] = (z == ENTREPRENEUR) & (d == WORKER)
            if t == par.T-1: break
            xi_w = np.exp(sw*rng.standard_normal(par.simN)-sw**2/2)
            xi_e = np.exp(se*rng.standard_normal(par.simN)-se**2/2)
            a, d = sim.a[t], sim.d[t]; sim.z[t+1] = d
            k = egm.capital(a, par)
            sim.m[t+1] = np.where(d == WORKER, a+par.wage*xi_w, a+par.theta*k**par.alpha*xi_e-par.r*k)

    def threshold(self, t, current_occupation):
        """Interpolated first crossing where v_entrepreneur-v_worker changes from negative to positive."""
        m = self.par.grid_m
        diff = self.sol.v_choice[t, current_occupation, ENTREPRENEUR]-self.sol.v_choice[t, current_occupation, WORKER]
        if diff[0] >= 0: return m[0]
        crossings = np.flatnonzero((diff[:-1] < 0) & (diff[1:] >= 0))
        if crossings.size == 0: return np.nan
        i = crossings[0]
        return m[i]-diff[i]*(m[i+1]-m[i])/(diff[i+1]-diff[i])
