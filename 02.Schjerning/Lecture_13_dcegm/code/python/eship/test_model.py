"""Small numerical checks for the lifecycle entrepreneurship example."""

import unittest
import numpy as np
import egm
import tools
from model import ENTREPRENEUR, WORKER, model_eship


class EntrepreneurshipTests(unittest.TestCase):
    def solve(self, sigma_eta=0.0):
        model = model_eship(); model.par.simN = 4_000; model.par.sigma_eta = sigma_eta
        model.create_grids(); model.solve(); model.simulate()
        return model

    def test_feasibility_and_transitions(self):
        model = self.solve(); p, s = model.par, model.sol
        for z in range(2):
            for d in range(2):
                cost = egm.switch_cost(z, d, p)
                used = s.c_choice[:-1, z, d]+s.a_choice[:-1, z, d]/p.R+cost
                grid = np.broadcast_to(p.grid_m, used.shape); feasible = grid > cost
                self.assertLess(np.max(np.abs(used[feasible]-grid[feasible])), 2e-8)
        self.assertGreater(model.sim.entry.sum(), 100)
        self.assertGreater(model.sim.exit.sum(), 100)
        self.assertGreater(np.sum(model.sim.entry[1:]), 50)

    def test_entry_cost_creates_hysteresis(self):
        model = self.solve()
        for t in [0, 10, 20, 30]:
            self.assertGreater(model.threshold(t, WORKER), model.threshold(t, ENTREPRENEUR))
            threshold = model.threshold(t, WORKER)
            difference = model.sol.v_choice[t,WORKER,ENTREPRENEUR]-model.sol.v_choice[t,WORKER,WORKER]
            self.assertAlmostEqual(float(tools.interp(model.par.grid_m, difference, threshold)), 0.0, places=10)

    def test_taste_shock_hook(self):
        model = self.solve(sigma_eta=0.05); prob = model.sol.prob[:-1]
        self.assertTrue(np.allclose(np.sum(prob, axis=2), 1.0))
        self.assertTrue(np.any((prob > 0.01) & (prob < 0.99)))

    def test_dense_search_does_not_materially_improve_egm(self):
        model = self.solve(); p, s = model.par, model.sol; gains = []
        for cash in np.linspace(0.4, 8.0, 16):
            a = np.linspace(0, p.R*(cash-p.entry_cost-1e-6), 1_200)
            c = cash-p.entry_cost-a/p.R
            v = egm.utility(c, p)+p.beta*egm.continuation(a, ENTREPRENEUR, 10, s, p)
            gains.append(np.max(v)-tools.interp(p.grid_m, s.v_choice[10,WORKER,ENTREPRENEUR], cash))
        self.assertLess(np.max(gains), 0.01)


if __name__ == '__main__': unittest.main()
