import os
from pathlib import Path
import shutil
from latteasy.preprocessing.foam_tools import FoamSimulation

here = Path(__file__).resolve().parent
os.chdir(here)

geometry_file = here/"input/h18_1constriction_channel_400_120_18.dat"
domain_size = (400, 120, 18)
num_cores = 128

simulation = FoamSimulation(
    geometry_file, domain_size, cpus=num_cores, mpi_launcher="ibrun",
    params={"geometry": {"fluidPoolHeight_LB": 18}, "Nucleation": {"distribution": "list", "numberOfBubbles": 2, "radius": 10, "shift": 20, "packingOffset": 3},
     "numerics": {"g_LB": -1e-4, "maxIter": 20000,}, "fluid": {"contactAngle": 0.0 }, "output": {"outIter": 500, "save_it": 100000},
    },  
)
for ax in "xyz":
    shutil.copy(f"nucleation_{ax}_points.txt", simulation.out_path)

history_log = simulation.run_sim()

print(f"Cores: {num_cores}")
print(f"Nuclei placed: {len(simulation.nuclei)}")
print(f"Bubble history log: {Path(history_log).resolve()}")
print(f"Simulation folder: {Path(simulation.folder_path).resolve()}")
