import os
from pathlib import Path

import porespy as ps

from latteasy._native import (
    build_solver,
    build_two_phase_solver,
    find_solver_executable,
    find_two_phase_executable,
)
from latteasy.two_phase import LattEasyUnsteadyRelativePermeability

shape = (64, 64, 64)
porosity = 0.65
blobiness = 2
seed = 7
num_cores = os.cpu_count() or 1
num_pressure_steps = 6

os.chdir(Path(__file__).resolve().parent)

geometry = ps.generators.blobs(
    shape=list(shape),
    porosity=porosity,
    blobiness=blobiness,
    seed=seed,
).astype(bool)

try:
    two_phase_solver = find_two_phase_executable()
except FileNotFoundError:
    two_phase_solver = build_two_phase_solver()

try:
    single_phase_solver = find_solver_executable()
except FileNotFoundError:
    single_phase_solver = build_solver()

simulation = LattEasyUnsteadyRelativePermeability(
    geometry,
    cpus=num_cores,
    two_phase_solver_path=two_phase_solver,
    single_phase_solver_path=single_phase_solver,
    num_pressure_steps=num_pressure_steps,
    minimum_radius=3,
    convergence_iter=250,
    max_iterations=20000,
    vtk_iter=0,
    relperm_max_iterations=250000,
)
results = simulation.run_relperm()

print(f"Seed: {seed}")
print(f"Cores: {num_cores}")
print(f"Pressure steps: {num_pressure_steps}")
print(
    f"Absolute permeability: {results['absolute_permeability']:.5g} [l.u.^2]"
)
print(f"Simulation folder: {Path(results['folder_path']).resolve()}")
print(f"Two-phase log: {Path(results['two_phase_log']).resolve()}")
print(f"Results table: {Path(results['table_path']).resolve()}")
print(f"Summary plot: {Path(results['plot_path']).resolve()}")
print(f"Final-state preview: {Path(results['preview_path']).resolve()}")
