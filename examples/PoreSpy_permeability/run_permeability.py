import porespy as ps
from latteasy.preprocessing.IO_tools import write_MPLBM


im = ps.generators.blobs(shape=[128, 128, 128], porosity=0.6, blobiness=2)
writer = write_MPLBM(im)
perm = writer.run_sim()
print(f"Permeability: {perm}")


