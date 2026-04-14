import porespy as ps
from latteasy.preprocessing.IO_tools import LattEasySimulation


im = ps.generators.blobs(shape=[128, 128, 128], porosity=0.6, blobiness=2)
writer = LattEasySimulation(im)
perm = writer.run_sim()
print(f"Permeability: {perm}")

