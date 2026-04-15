# LattEasy 🤌

LattEasy is a small, approachable toolkit for running Lattice Boltzmann
simulations from 3D pore geometries using the Palabos backend.


## Quickstart

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install the Python package

```bash
pip install -e .
```

This installs the Python interface and CLI. It does not try to compile the
native solver during installation.

### 3. Check your machine

```bash
latteasy doctor
```

`latteasy doctor` tells you whether Python dependencies, CMake, MPI, and the
native solver are ready.

### 4. Build the bundled solver

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

### 5. Run the first simulation

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
pip install -e ".[examples]"
python examples/gray_sponge_permeability/run_permeability.py
```

This keeps the usual permeability workflow, but swaps in a separate
`gray_permeability` solver and inserts a seeded gray fracture through the
middle of the sample before the run. It now runs both the no-fracture and
fracture cases, then writes a side-by-side velocity comparison plot into the
fracture run folder as `output/velocity_compare_vs_no_fracture.png`.

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

