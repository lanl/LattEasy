import os
from pathlib import Path

import porespy as ps

from latteasy.preprocessing.IO_tools import LattEasySimulation

shape = (128, 128, 128)
porosity = 0.60
blobiness = 2
seed = 7
num_cores = 4

os.chdir(Path(__file__).resolve().parent)

geometry = ps.generators.blobs(
    shape=list(shape),
    porosity=porosity,
    blobiness=blobiness,
    seed=seed,
).astype(bool)

simulation = LattEasySimulation(geometry, cpus=num_cores)
permeability = simulation.run_sim()

print(f"Seed: {seed}")
print(f"Cores: {num_cores}")
print(f"Permeability: {permeability:.5g} [l.u.^2]")
print(f"Simulation folder: {Path(simulation.folder_path).resolve()}")
