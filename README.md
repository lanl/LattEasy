# LattEasy

LattEasy provides preprocessing and postprocessing utilities for running
lattice Boltzmann simulations with external C++ solvers.  The original
C++ sources are included in the `src/` directory.

## Installation

Clone this repository and install it with `pip`:

```
git clone https://github.com/youruser/LattEasy.git
cd LattEasy
pip install .
```

On Debian-based systems you can get the required build tools with:

```
sudo apt-get install build-essential cmake make libopenmpi-dev openmpi-bin
```
On macOS with Homebrew you can install them via:
```
brew install cmake open-mpi
```

The `cmake` command must come from your system package manager (not the
Python `cmake` module), otherwise installation will fail with
`ModuleNotFoundError: No module named 'cmake'`.

`pip install .` will unpack the Palabos sources and build the C++
permeability solver.  A C++17 compiler, `cmake`, `make` and an MPI
implementation must be available on your system.

## Usage

```python
from latteasy.preprocessing import geometry
```

The `examples/` directory contains small demos such as
`PoreSpy_permeability/run_permeability.py`.
## Instructions### For my cluster```bash module load cmake &&module load mpich/3.3.2/gcc-7.5.0 &&```