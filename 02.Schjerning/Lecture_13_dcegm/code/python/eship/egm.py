"""The DC-EGM step for the finite-horizon entrepreneurship model."""

import numpy as np
try: from . import tools
except ImportError: import tools

WORKER, ENTREPRENEUR = 0, 1


def utility(c, par):
    c = np.maximum(c, 1e-12)
    return np.log(c) if par.rho == 1 else c**(1-par.rho)/(1-par.rho)


def marg_utility(c, par): return np.maximum(c, 1e-12)**(-par.rho)
def inv_marg_utility(mu, par): return np.maximum(mu, 1e-300)**(-1/par.rho)


def switch_cost(z, d, par):
    if z == WORKER and d == ENTREPRENEUR: return par.entry_cost
    if z == ENTREPRENEUR and d == WORKER: return par.exit_cost
    return 0.0


def capital(a, par):
    """Static capital choice, constrained by business collateral."""
    k_star = (par.alpha*par.theta/par.r)**(1/(1-par.alpha))
    return np.minimum(par.lambd*np.maximum(a, 0.0), k_star)


def transition(a, d, shocks, par, derivative=False):
    """M+ = a+w*eps_W or a+theta*k(a)^alpha*eps_E-r*k(a); optionally return dM+/da."""
    a = np.asarray(a)
    if d == WORKER:
        m_plus = a[..., None]+par.wage*shocks
        return (m_plus, np.ones_like(m_plus)) if derivative else m_plus
    k = capital(a, par); m_plus = a[..., None]+par.theta*k[..., None]**par.alpha*shocks-par.r*k[..., None]
    if not derivative: return m_plus
    k_star = (par.alpha*par.theta/par.r)**(1/(1-par.alpha))
    dk = par.lambd*(par.lambd*a < k_star)
    dm = 1+dk[..., None]*(par.theta*par.alpha*np.maximum(k[..., None], 1e-12)**(par.alpha-1)*shocks-par.r)
    return m_plus, dm


def continuation(a, d, t, sol, par, derivative=False):
    """Expected continuation value and marginal value conditional on occupation d."""
    shocks, weights = (par.xi_w, par.xi_w_w) if d == WORKER else (par.xi_e, par.xi_e_w)
    if derivative:
        m_plus, dm = transition(a, d, shocks, par, True)
        mu_plus = tools.interp(par.grid_m, sol.mu[t+1, d], m_plus)
        return np.sum(weights*mu_plus*dm, axis=-1)
    m_plus = transition(a, d, shocks, par)
    v_plus = tools.interp(par.grid_m, sol.v[t+1, d], m_plus)
    return np.sum(weights*v_plus, axis=-1)


def value(m, c, z, d, t, sol, par):
    """Choice-specific value used by the upper-envelope scan."""
    a = par.R*(m-c-switch_cost(z, d, par))
    return utility(c, par)+par.beta*continuation(a, d, t, sol, par)


def solve_choice(z, d, t, sol, par):
    """One EGM step, followed by the upper envelope within occupation d."""
    a_raw = par.grid_a
    emu = continuation(a_raw, d, t, sol, par, derivative=True)
    c_raw = inv_marg_utility(par.beta*par.R*emu, par)
    m_raw = c_raw+a_raw/par.R+switch_cost(z, d, par)

    # Borrowing-constrained candidate: a=0 and c=m-cost.
    cost = switch_cost(z, d, par); m = par.grid_m
    c = np.maximum(m-cost, 1e-12); a = np.zeros_like(m)
    v = np.where(m > cost, value(m, c, z, d, t, sol, par), -1e12)

    # Every adjacent pair of endogenous points is a candidate. Comparing their
    # values is the upper-envelope step that retains folds and secondary kinks.
    for i in range(par.Na-1):
        lo, hi = sorted((m_raw[i], m_raw[i+1])); I = (m >= lo) & (m <= hi)
        if not np.any(I) or m_raw[i] == m_raw[i+1]: continue
        c_i = c_raw[i]+(m[I]-m_raw[i])*(c_raw[i+1]-c_raw[i])/(m_raw[i+1]-m_raw[i])
        a_i = par.R*(m[I]-c_i-cost); v_i = value(m[I], c_i, z, d, t, sol, par)
        better = (a_i >= -1e-10) & (v_i > v[I])
        idx = np.flatnonzero(I)[better]; c[idx], a[idx], v[idx] = c_i[better], np.maximum(a_i[better], 0), v_i[better]

    # Extend the final increasing segment if the endogenous grid ends early.
    i = np.argmax(m_raw); j = max(i-1, 0); I = m > m_raw[i]
    if np.any(I) and m_raw[i] != m_raw[j]:
        c_i = c_raw[i]+(m[I]-m_raw[i])*(c_raw[i]-c_raw[j])/(m_raw[i]-m_raw[j])
        a_i = par.R*(m[I]-c_i-cost); v_i = value(m[I], c_i, z, d, t, sol, par)
        better = (a_i >= 0) & (v_i > v[I]); idx = np.flatnonzero(I)[better]
        c[idx], a[idx], v[idx] = c_i[better], a_i[better], v_i[better]
    return m_raw, c_raw, v, c, a
