import os
from pathlib import Path

import porespy as ps

from latteasy._native import build_two_phase_solver, find_two_phase_executable
from latteasy.two_phase import LattEasyTwoPhaseSimulation

shape = (48, 48, 48)
porosity = 0.65
blobiness = 2
seed = 7
num_cores = 1
non_wetting_fraction = 0.30

os.chdir(Path(__file__).resolve().parent)

geometry = ps.generators.blobs(
    shape=list(shape),
    porosity=porosity,
    blobiness=blobiness,
    seed=seed,
).astype(bool)

try:
    solver = find_two_phase_executable()
except FileNotFoundError:
    solver = build_two_phase_solver()

simulation = LattEasyTwoPhaseSimulation(
    geometry,
    non_wetting_fraction=non_wetting_fraction,
    cpus=num_cores,
    solver_path=solver,
    convergence_iter=500,
    max_iterations=10000,
    vtk_iter=1000,
)
summary = simulation.run_sim()

print(f"Seed: {seed}")
print(f"Cores: {num_cores}")
print(f"Non-wetting fraction: {non_wetting_fraction:.2f}")
print(f"Fluid 1 capillary number: {summary.get('capillary_number_fluid1', float('nan')):.5g}")
print(f"Fluid 2 capillary number: {summary.get('capillary_number_fluid2', float('nan')):.5g}")
print(f"Fluid 1 mean x-velocity: {summary.get('velocity_fluid1', float('nan')):.5g} [l.u.]")
print(f"Fluid 2 mean x-velocity: {summary.get('velocity_fluid2', float('nan')):.5g} [l.u.]")
print(f"Simulation folder: {Path(simulation.folder_path).resolve()}")
print(f"Solver log: {Path(simulation.log_path).resolve()}")
