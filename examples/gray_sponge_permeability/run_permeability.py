from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import porespy as ps

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from latteasy._native import (
    build_gray_permeability_solver,
    find_gray_permeability_executable,
)
from latteasy.preprocessing.IO_tools import LattEasySimulation


ENCODED_FLUID = 2608
ENCODED_GRAY_FRACTURE = 2611


def geometry_file_for_run(run_dir: Path) -> Path:
    return next((run_dir / "input").glob("*.dat"))


def velocity_file_for_run(run_dir: Path) -> Path:
    return next((run_dir / "output").glob("*_vel.dat"))


def parse_geometry_shape(geometry_file: Path) -> tuple[int, int, int]:
    match = re.search(r"_(\d+)_(\d+)_(\d+)\.dat$", geometry_file.name)
    if match is None:
        raise RuntimeError(f"Could not parse geometry size from `{geometry_file.name}`.")
    return tuple(int(value) for value in match.groups())


def add_random_gray_fracture(run_dir: Path, seed: int = 1) -> int:
    geometry_file = geometry_file_for_run(run_dir)
    nx, ny, nz = parse_geometry_shape(geometry_file)
    geometry = np.fromfile(geometry_file, dtype=np.int16).reshape((nx, ny, nz))

    rng = np.random.default_rng(seed)
    y_mid = ny // 2
    z_center = nz // 2
    z_centers = np.empty(nx, dtype=np.int32)
    z_centers[0] = z_center

    for i in range(1, nx):
        z_centers[i] = np.clip(z_centers[i - 1] + rng.integers(-1, 2), 3, nz - 4)

    changed = 0
    for x in range(2, nx - 2):
        y_half = 2 + int(rng.integers(0, 2))
        z_half = 3 + int(rng.integers(0, 3))
        y0 = max(1, y_mid - y_half)
        y1 = min(ny - 1, y_mid + y_half + 1)
        z0 = max(1, z_centers[x] - z_half)
        z1 = min(nz - 1, z_centers[x] + z_half + 1)

        fracture_view = geometry[x, y0:y1, z0:z1]
        rock_mask = fracture_view != ENCODED_FLUID
        changed += int(rock_mask.sum())
        fracture_view[rock_mask] = ENCODED_GRAY_FRACTURE

    geometry.astype(np.int16).tofile(geometry_file)
    return changed


def load_velocity_field(run_dir: Path) -> np.ndarray:
    geometry_file = geometry_file_for_run(run_dir)
    nx, ny, nz = parse_geometry_shape(geometry_file)
    velocity_file = velocity_file_for_run(run_dir)
    velocity = np.fromfile(velocity_file, sep=" ")
    return velocity.reshape((nx, ny, nz, 3))


def load_fracture_mask(run_dir: Path) -> np.ndarray:
    geometry_file = geometry_file_for_run(run_dir)
    nx, ny, nz = parse_geometry_shape(geometry_file)
    geometry = np.fromfile(geometry_file, dtype=np.int16).reshape((nx, ny, nz))
    return geometry == ENCODED_GRAY_FRACTURE


def write_velocity_comparison_plot(
    baseline_run_dir: Path, fracture_run_dir: Path, output_file: Path
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline_velocity = load_velocity_field(baseline_run_dir)
    fracture_velocity = load_velocity_field(fracture_run_dir)
    fracture_mask = load_fracture_mask(fracture_run_dir)

    baseline_ux = baseline_velocity[..., 0]
    fracture_ux = fracture_velocity[..., 0]
    delta_ux = fracture_ux - baseline_ux

    mid_y = baseline_ux.shape[1] // 2
    baseline_slice = baseline_ux[:, mid_y, :].T
    fracture_slice = fracture_ux[:, mid_y, :].T
    delta_slice = delta_ux[:, mid_y, :].T
    fracture_slice_mask = fracture_mask[:, mid_y, :].T

    ux_vmax = max(float(baseline_slice.max()), float(fracture_slice.max()))
    delta_lim = max(abs(float(delta_slice.min())), abs(float(delta_slice.max())))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    image = axes[0, 0].imshow(
        baseline_slice, origin="lower", cmap="viridis", vmin=0.0, vmax=ux_vmax, aspect="auto"
    )
    axes[0, 0].set_title("No fracture: $u_x$ at middle y-slice")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("z")
    fig.colorbar(image, ax=axes[0, 0], label="$u_x$ [l.u.]")

    image = axes[0, 1].imshow(
        fracture_slice, origin="lower", cmap="viridis", vmin=0.0, vmax=ux_vmax, aspect="auto"
    )
    axes[0, 1].set_title("Gray fracture: $u_x$ at same slice")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("z")
    fig.colorbar(image, ax=axes[0, 1], label="$u_x$ [l.u.]")

    image = axes[1, 0].imshow(
        delta_slice,
        origin="lower",
        cmap="coolwarm",
        vmin=-delta_lim,
        vmax=delta_lim,
        aspect="auto",
    )
    axes[1, 0].set_title("Exact difference: $u_x$(fracture) - $u_x$(no fracture)")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("z")
    fig.colorbar(image, ax=axes[1, 0], label="$\\Delta u_x$ [l.u.]")

    image = axes[1, 1].imshow(
        fracture_slice_mask, origin="lower", cmap="Oranges", vmin=0, vmax=1, aspect="auto"
    )
    axes[1, 1].set_title("Gray fracture cells on that slice")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("z")
    fig.colorbar(image, ax=axes[1, 1], label="fracture mask")

    fig.suptitle("Gray sponge permeability: exact velocity comparison", fontsize=16)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=180)
    plt.close(fig)
    return output_file


def run_case(image: np.ndarray, solver: Path, add_fracture: bool) -> tuple[Path, float, int]:
    writer = LattEasySimulation(image, solver_path=solver)
    fracture_voxels = 0
    if add_fracture:
        fracture_voxels = add_random_gray_fracture(Path(writer.folder_path))
    permeability = writer.run_sim()
    return Path(writer.folder_path), permeability, fracture_voxels


def main():
    np.random.seed(0)
    im = ps.generators.blobs(shape=[128, 128, 128], porosity=0.6, blobiness=2)

    try:
        solver = find_gray_permeability_executable()
    except FileNotFoundError:
        solver = build_gray_permeability_solver()

    print("Running no-fracture reference case...")
    baseline_run_dir, baseline_perm, _ = run_case(im, solver, add_fracture=False)
    print("Running gray-fracture case...")
    fracture_run_dir, fracture_perm, fracture_voxels = run_case(im, solver, add_fracture=True)
    comparison_plot = write_velocity_comparison_plot(
        baseline_run_dir,
        fracture_run_dir,
        fracture_run_dir / "output" / "velocity_compare_vs_no_fracture.png",
    )

    print(f"No-fracture permeability: {baseline_perm}")
    print(f"Gray fracture voxels: {fracture_voxels}")
    print(f"Fracture permeability: {fracture_perm}")
    print(f"Baseline files: {baseline_run_dir}")
    print(f"Fracture files: {fracture_run_dir}")
    print(f"Velocity comparison plot: {comparison_plot}")


if __name__ == "__main__":
    main()
