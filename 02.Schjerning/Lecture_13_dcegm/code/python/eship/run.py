"""Solve and simulate the lifecycle entrepreneurship example."""

import numpy as np
from model import model_eship


if __name__ == '__main__':
    model = model_eship()
    model.create_grids(); model.solve(); model.simulate()
    par, sim = model.par, model.sim
    print(f'Entrants: {sim.entry.sum():,}; exits: {sim.exit.sum():,}')
    print(f'Peak entrepreneur share: {np.mean(sim.z, axis=1).max():.1%}')
    for age in [25, 35, 45, 55]:
        t = age-par.age_min
        print(f'Age {age}: entry threshold = {model.threshold(t, 0):.2f}, continuation threshold = {model.threshold(t, 1):.2f}')
