"""Small, friendly demo helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._native import find_mpi_launcher, find_solver_executable
from .preprocessing.IO_tools import LattEasySimulation


@dataclass
class DemoResult:
    folder_path: Path
    permeability: float


def make_channel_geometry(shape=(32, 32, 64), radius=None):
    """Create a simple straight channel that is stable and fast to run."""
    x_size, y_size, z_size = shape
    if min(shape) < 16:
        raise ValueError("The demo geometry needs each dimension to be at least 16 voxels.")

    if radius is None:
        radius = max(4, min(x_size, y_size) // 4)
    if radius <= 0:
        raise ValueError("Channel radius must be positive.")
    if radius >= min(x_size, y_size) / 2:
        raise ValueError("Channel radius must leave some solid wall around the pore space.")

    x = np.arange(x_size) - (x_size - 1) / 2
    y = np.arange(y_size) - (y_size - 1) / 2
    xx, yy = np.meshgrid(x, y, indexing="ij")
    channel = (xx**2 + yy**2) <= radius**2

    pore = np.zeros((x_size, y_size, z_size), dtype=np.uint8)
    pore[:, :, :] = channel[:, :, None]
    return pore


def run_demo(shape=(32, 32, 64), radius=None, cpus=1):
    """Run the built-in straight-channel permeability demo."""
    find_solver_executable()

    launcher = find_mpi_launcher()
    if launcher is None:
        raise RuntimeError(
            "No MPI launcher was found. Install MPI and make sure `mpirun` or `mpiexec` is on your PATH."
        )

    pore = make_channel_geometry(shape=shape, radius=radius)
    writer = LattEasySimulation(pore, cpus=cpus)
    permeability = writer.run_sim(mpi_procs=cpus)
    return DemoResult(folder_path=Path(writer.folder_path).resolve(), permeability=permeability)
