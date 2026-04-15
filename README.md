# LattEasy 🤌

LattEasy is a small, approachable toolkit for running Lattice Boltzmann
simulations from 3D pore geometries using the Palabos backend.


## Quickstart

`uv` is required for the Python workflow.

### 1. Sync the project environment

```bash
uv sync
source .venv/bin/activate
```

This creates or updates the local project environment. It does not try to
compile the native solver during sync.

### 2. Check your machine

```bash
latteasy doctor
```

`latteasy doctor` tells you whether Python dependencies, CMake, MPI, and the
native solver are ready.

### 3. Build the bundled solver

```bash
latteasy build
```

This step needs:

- a C++ compiler
- `cmake`
- MPI development libraries and a launcher such as `mpirun` or `mpiexec`

Typical installs:

- Debian and Ubuntu: `sudo apt-get install build-essential cmake libopenmpi-dev openmpi-bin`
- macOS with Homebrew: `brew install cmake open-mpi`

Use the system `cmake` executable. The Python `cmake` package is not enough.

### 4. Run the first simulation

```bash
latteasy demo
```

This runs a small built-in straight-channel permeability example with gentle
defaults and writes output into `sims/pore_<n>/`.

After the run you should see:

- a permeability value in the terminal
- solver logs in `sims/pore_<n>/perm.txt`
- simulation outputs in `sims/pore_<n>/output/`

## Examples

Run the built-in demo from Python:

```python
from latteasy.demo import run_demo

result = run_demo()
print(result.permeability)
print(result.folder_path)
```

There is also a minimal script at `examples/first_simulation.py`.

For the standalone gray-fracture example:

```bash
uv sync
source .venv/bin/activate
python examples/gray_sponge_permeability/run_permeability.py
```

This keeps the usual permeability workflow, but swaps in a separate
`gray_permeability` solver and inserts a seeded gray fracture through the
middle of the sample before the run. It now runs both the no-fracture and
fracture cases, then writes a side-by-side velocity comparison plot into the
fracture run folder as `output/velocity_compare_vs_no_fracture.png`.

For a small PoreSpy-based two-phase steady-state example:

```bash
uv sync
source .venv/bin/activate
python examples/porespy_two_phase/run_steady_state.py
```

This builds the `ShanChen` solver if needed, generates a seeded blobs geometry,
initializes a simple wetting/non-wetting configuration, and runs one steady
two-phase case with beginner-friendly defaults.

For a small PoreSpy-based unsteady relative permeability example:

```bash
uv sync
source .venv/bin/activate
python examples/porespy_unsteady_relperm/run_relperm.py
```

This uses pressure-step drainage in the `ShanChen` solver, then runs the
single-phase permeability solver on the wetting and non-wetting masks from each
saved state. It writes a CSV table plus a combined capillary-pressure /
relative-permeability plot into `examples/porespy_unsteady_relperm/sims/`.

## Command Line

```bash
latteasy doctor
latteasy build
latteasy demo
latteasy demo --help
```

## Repository layout

```text
latteasy/         Python package
examples/         small runnable examples
src/              native C++ solver sources and bundled Palabos archive
```
