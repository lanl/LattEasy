import os
import re
from pathlib import Path
from subprocess import run

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt as edist

from latteasy._native import (
    build_runtime_env,
    find_mpi_launcher,
    find_solver_executable,
    find_two_phase_executable,
)
from latteasy.preprocessing.IO_tools import (
    create_folder,
    create_single_phase_input_file,
    erase_regions,
    read_permeability,
)


def read_two_phase_summary(file_path):
    text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    summary = {}
    patterns = {
        "velocity_fluid1": r"Average x-velocity for fluid1 \[l\.u\.\] =\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        "velocity_fluid2": r"Average x-velocity for fluid2 \[l\.u\.\] =\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        "capillary_number_fluid1": r"Capillary number fluid1 =\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        "capillary_number_fluid2": r"Capillary number fluid2 =\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        "runtime_seconds": r"Simulation took seconds:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    }

    for key, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            summary[key] = float(matches[-1])

    return summary


def read_pressure_steps(file_path):
    text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(
        r"Pressure difference =\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        text,
    )
    return np.array([float(value) for value in matches], dtype=float)


def check_two_phase_install():
    try:
        exe = find_two_phase_executable()
    except FileNotFoundError:
        raise RuntimeError(
            "Two-phase solver not found. Build it before running this example."
        ) from None

    print(f"\nTwo-phase solver was found: {exe}\n")
    return exe


def check_single_phase_install():
    try:
        exe = find_solver_executable()
    except FileNotFoundError:
        raise RuntimeError(
            "Single-phase solver not found. Build it before running this example."
        ) from None

    print(f"\nSingle-phase solver was found: {exe}\n")
    return exe


def create_numbered_folder(prefix):
    for i in range(10000000000):
        path = Path(f"{prefix}_{i}")
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue


def create_two_phase_simulation_matrix(
    rock,
    non_wetting_fraction=0.0,
    buffer_layers=0,
):
    if non_wetting_fraction < 0 or non_wetting_fraction > 1:
        raise ValueError("`non_wetting_fraction` must be between 0 and 1.")
    if buffer_layers < 0:
        raise ValueError("`buffer_layers` must be zero or positive.")

    if buffer_layers:
        rock = np.pad(rock, [(buffer_layers, buffer_layers), (0, 0), (0, 0)])

    sim_matrix = edist(rock)

    sim_matrix[0, :, :] = 1
    sim_matrix[:, 0, :] = 1
    sim_matrix[:, :, 0] = 1
    sim_matrix[-1, :, :] = 1
    sim_matrix[:, -1, :] = 1
    sim_matrix[:, :, -1] = 1

    sim_matrix[rock == 0] = 0
    sim_matrix[(sim_matrix > 0) & (sim_matrix < 2)] = 1
    sim_matrix[sim_matrix > 1] = 2

    pore_mask = sim_matrix == 0
    non_wetting_voxels = int(round(pore_mask.sum() * non_wetting_fraction))

    if non_wetting_voxels > 0:
        distance_to_solid = edist(np.where(rock == 1, 0, 1))
        pore_scores = distance_to_solid[pore_mask]
        pore_indices = np.argwhere(pore_mask)
        order = np.argsort(pore_scores)[::-1]
        selected = pore_indices[order[:non_wetting_voxels]]
        sim_matrix[selected[:, 0], selected[:, 1], selected[:, 2]] = 3

    sim_matrix = sim_matrix.astype(np.int16)
    sim_matrix[sim_matrix == 0] = 2608
    sim_matrix[sim_matrix == 1] = 2609
    sim_matrix[sim_matrix == 2] = 2610
    sim_matrix[sim_matrix == 3] = 2611
    return sim_matrix


def create_single_phase_simulation_matrix(rock, buffer_layers=2):
    if buffer_layers < 0:
        raise ValueError("`buffer_layers` must be zero or positive.")

    if buffer_layers:
        rock = np.pad(rock, [(buffer_layers, buffer_layers), (0, 0), (0, 0)])

    sim_matrix = edist(rock)

    sim_matrix[0, :, :] = 1
    sim_matrix[:, 0, :] = 1
    sim_matrix[:, :, 0] = 1
    sim_matrix[-1, :, :] = 1
    sim_matrix[:, -1, :] = 1
    sim_matrix[:, :, -1] = 1

    sim_matrix[rock == 0] = 0
    sim_matrix[(sim_matrix > 0) & (sim_matrix < 2)] = 1
    sim_matrix[sim_matrix > 1] = 2

    sim_matrix = sim_matrix.astype(np.int16)
    sim_matrix[sim_matrix == 0] = 2608
    sim_matrix[sim_matrix == 1] = 2609
    sim_matrix[sim_matrix == 2] = 2610
    return sim_matrix


def create_two_phase_input_file(
    input_file_name,
    geometry_file_name,
    domain_size,
    output_folder,
    periodic,
    rho_f1,
    rho_f2,
    force_f1,
    force_f2,
    Gc,
    omega_f1,
    omega_f2,
    G_ads_f1_s1,
    G_ads_f1_s2,
    G_ads_f1_s3,
    G_ads_f1_s4,
    convergence,
    convergence_iter,
    max_iterations,
    gif_iter,
    vtk_iter,
    rho_f2_vtk,
    print_geom,
    print_stl,
    fluid_from_geom=True,
    fluid1_box=(1, 1, 1, 1, 1, 1),
    fluid2_box=(1, 1, 1, 1, 1, 1),
    pressure_bc=False,
    rho_f1_i=None,
    rho_f2_i=None,
    num_pc_steps=0,
    min_radius=1,
    rho_d=0.06,
):
    x_size, y_size, z_size = domain_size
    periodic_x, periodic_y, periodic_z = periodic
    fluid1_x1, fluid1_x2, fluid1_y1, fluid1_y2, fluid1_z1, fluid1_z2 = fluid1_box
    fluid2_x1, fluid2_x2, fluid2_y1, fluid2_y2, fluid2_z1, fluid2_z2 = fluid2_box

    if rho_f1_i is None:
        rho_f1_i = rho_f1
    if rho_f2_i is None:
        rho_f2_i = rho_f2

    with open(input_file_name, "w", encoding="utf-8") as file:
        file.write('<?xml version="1.0" ?>\n\n')

        file.write("<geometry>\n")
        file.write(f"\t<file_geom> input/{geometry_file_name} </file_geom>\n")
        file.write(
            f"\t<size> <x> {x_size} </x> <y> {y_size} </y> <z> {z_size} </z> </size>\n"
        )
        file.write("\t<per>\n")
        file.write(
            f"\t\t<fluid1> <x> {periodic_x} </x> <y> {periodic_y} </y> <z> {periodic_z} </z> </fluid1>\n"
        )
        file.write(
            f"\t\t<fluid2> <x> {periodic_x} </x> <y> {periodic_y} </y> <z> {periodic_z} </z> </fluid2>\n"
        )
        file.write("\t</per>\n")
        file.write("</geometry>\n\n")

        file.write("<init>\n")
        file.write(f"\t<fluid_from_geom> {str(fluid_from_geom)} </fluid_from_geom>\n")
        file.write("\t<fluid1>\n")
        file.write(
            f"\t\t<x1> {fluid1_x1} </x1> <y1> {fluid1_y1} </y1> <z1> {fluid1_z1} </z1>\n"
        )
        file.write(
            f"\t\t<x2> {fluid1_x2} </x2> <y2> {fluid1_y2} </y2> <z2> {fluid1_z2} </z2>\n"
        )
        file.write("\t</fluid1>\n")
        file.write("\t<fluid2>\n")
        file.write(
            f"\t\t<x1> {fluid2_x1} </x1> <y1> {fluid2_y1} </y1> <z1> {fluid2_z1} </z1>\n"
        )
        file.write(
            f"\t\t<x2> {fluid2_x2} </x2> <y2> {fluid2_y2} </y2> <z2> {fluid2_z2} </z2>\n"
        )
        file.write("\t</fluid2>\n")
        file.write("</init>\n\n")

        file.write("<fluids>\n")
        file.write(f"\t<Gc> {Gc} </Gc>\n")
        file.write(f"\t<omega_f1> {omega_f1} </omega_f1>\n")
        file.write(f"\t<omega_f2> {omega_f2} </omega_f2>\n")
        file.write(f"\t<force_f1> {force_f1} </force_f1>\n")
        file.write(f"\t<force_f2> {force_f2} </force_f2>\n")
        file.write(f"\t<G_ads_f1_s1> {G_ads_f1_s1} </G_ads_f1_s1>\n")
        file.write(f"\t<G_ads_f1_s2> {G_ads_f1_s2} </G_ads_f1_s2>\n")
        file.write(f"\t<G_ads_f1_s3> {G_ads_f1_s3} </G_ads_f1_s3>\n")
        file.write(f"\t<G_ads_f1_s4> {G_ads_f1_s4} </G_ads_f1_s4>\n")
        file.write(f"\t<rho_f1> {rho_f1} </rho_f1>\n")
        file.write(f"\t<rho_f2> {rho_f2} </rho_f2>\n")
        file.write(f"\t<pressure_bc> {str(pressure_bc)} </pressure_bc>\n")
        file.write(f"\t<rho_f1_i> {rho_f1_i} </rho_f1_i>\n")
        file.write(f"\t<rho_f2_i> {rho_f2_i} </rho_f2_i>\n")
        file.write(f"\t<num_pc_steps> {num_pc_steps} </num_pc_steps>\n")
        file.write(f"\t<min_radius> {min_radius} </min_radius>\n")
        file.write(f"\t<rho_d> {rho_d} </rho_d>\n")
        file.write("</fluids>\n\n")

        file.write("<output>\n")
        file.write(f"\t<out_folder> {output_folder} </out_folder>\n")
        file.write(f"\t<convergence> {convergence} </convergence>\n")
        file.write(f"\t<it_max> {max_iterations} </it_max>\n")
        file.write(f"\t<it_conv> {convergence_iter} </it_conv>\n")
        file.write(f"\t<it_gif> {gif_iter} </it_gif>\n")
        file.write(f"\t<rho_vtk> {rho_f2_vtk} </rho_vtk>\n")
        file.write(f"\t<it_vtk> {vtk_iter} </it_vtk>\n")
        file.write(f"\t<print_geom> {print_geom} </print_geom>\n")
        file.write(f"\t<print_stl> {print_stl} </print_stl>\n")
        file.write("</output>\n")


def run_native_solver(cmd, cwd, log_path):
    env = build_runtime_env()
    with open(log_path, "w", encoding="utf-8") as file_handle:
        completed = run(
            cmd,
            cwd=cwd,
            env=env,
            stdout=file_handle,
            stderr=file_handle,
        )

    if completed.returncode != 0:
        raise RuntimeError(f"Simulation failed. Check `{log_path}` for details.")


def run_single_phase_case(
    case_path,
    geometry_name,
    geometry_matrix,
    solver_path,
    mpi_procs,
    pressure,
    max_iterations,
    convergence,
    save_vtks,
):
    create_folder(str(case_path))
    create_folder(str(case_path / "input"))
    create_folder(str(case_path / "output"))

    geometry_path = case_path / "input" / f"{geometry_name}.dat"
    geometry_matrix.tofile(geometry_path)

    create_single_phase_input_file(
        case_path / "permeability.xml",
        geometry_name,
        geometry_matrix.shape,
        ["false", "false", "false"],
        ["input/", "output/"],
        [1, pressure, max_iterations, convergence],
        "true" if save_vtks else "false",
    )

    mpi_launcher = find_mpi_launcher()
    if mpi_launcher is None:
        raise RuntimeError(
            "No MPI launcher was found. Install MPI and make sure `mpirun` or `mpiexec` is on your PATH."
        )

    cmd = [mpi_launcher, "-n", str(mpi_procs), str(solver_path), "permeability.xml"]
    log_path = case_path / "perm.txt"
    run_native_solver(cmd, case_path, log_path)

    result_path = case_path / "output" / "relPerm&vels.txt"
    return read_permeability(result_path), log_path


def load_density_state(file_path, shape):
    data = np.loadtxt(file_path, dtype=float)
    if data.size != np.prod(shape):
        raise ValueError(f"Unexpected density field size in `{file_path}`.")
    return data.reshape(shape)


def split_fluid_masks(density, rho_f1, pore_mask):
    threshold = rho_f1 - 1.0
    non_wetting_mask = (density >= threshold) & pore_mask
    wetting_mask = (density > 0) & (density < threshold) & pore_mask
    return wetting_mask, non_wetting_mask


def write_relperm_plot(output_path, wetting_saturation, capillary_pressure, krw, krnw):
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    axes[0].plot(wetting_saturation, capillary_pressure, "-o", color="#0f766e", lw=2)
    axes[0].set_ylabel("Capillary pressure [l.u.]")
    axes[0].grid(alpha=0.2)

    axes[1].plot(wetting_saturation, krw, "-o", color="#2563eb", lw=2, label="Krw")
    axes[1].plot(wetting_saturation, krnw, "-o", color="#dc2626", lw=2, label="Krnw")
    axes[1].set_xlabel("Wetting saturation")
    axes[1].set_ylabel("Relative permeability")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_state_preview(output_path, state_mask):
    mid_slice = state_mask.shape[1] // 2
    preview = state_mask[:, mid_slice, :]

    fig, ax = plt.subplots(figsize=(6, 4))
    image = ax.imshow(preview.T, origin="lower", cmap="viridis")
    ax.set_title("Final wetting / non-wetting state")
    ax.set_xlabel("Flow direction")
    ax.set_ylabel("Z")
    cbar = fig.colorbar(image, ax=ax, shrink=0.9)
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(["rock", "wetting", "non-wetting"])
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


class LattEasyTwoPhaseSimulation:
    def __init__(
        self,
        pore_obj,
        non_wetting_fraction=0.30,
        cpus=1,
        solver_path=None,
        rho_f1=2.0,
        rho_f2=2.0,
        force_f1=1e-4,
        force_f2=1e-4,
        Gc=0.9,
        omega_f1=1.0,
        omega_f2=1.0,
        G_ads_f1_s1=-0.4,
        G_ads_f1_s2=0.0,
        G_ads_f1_s3=0.0,
        G_ads_f1_s4=0.0,
        convergence=1e-4,
        convergence_iter=500,
        max_iterations=10000,
        gif_iter=0,
        vtk_iter=1000,
        rho_f2_vtk=False,
        print_geom=True,
        print_stl=False,
    ):
        self.lbm_loc = str(solver_path) if solver_path is not None else str(check_two_phase_install())
        self.cpus = cpus
        self.xml_name = "steady_state.xml"

        create_folder("sims")
        self.folder_path = create_numbered_folder("sims/two_phase")
        create_folder(str(self.folder_path / "input"))
        create_folder(str(self.folder_path / "output"))

        rock = 1 - np.transpose(pore_obj, (2, 0, 1))
        rock = erase_regions(rock)
        sim_matrix = create_two_phase_simulation_matrix(rock, non_wetting_fraction)

        pore_size = sim_matrix.shape
        geometry_name = f"pore_{pore_size[0]}_{pore_size[1]}_{pore_size[2]}.dat"
        geometry_path = self.folder_path / "input" / geometry_name
        sim_matrix.tofile(geometry_path)

        create_two_phase_input_file(
            self.folder_path / self.xml_name,
            geometry_name,
            pore_size,
            "output/",
            (True, True, True),
            rho_f1,
            rho_f2,
            force_f1,
            force_f2,
            Gc,
            omega_f1,
            omega_f2,
            G_ads_f1_s1,
            G_ads_f1_s2,
            G_ads_f1_s3,
            G_ads_f1_s4,
            convergence,
            convergence_iter,
            max_iterations,
            gif_iter,
            vtk_iter,
            rho_f2_vtk,
            print_geom,
            print_stl,
        )

        self.log_path = self.folder_path / "two_phase.txt"

    def run_sim(self, mpi_procs=None):
        if mpi_procs is None:
            mpi_procs = self.cpus

        mpi_launcher = find_mpi_launcher()
        if mpi_launcher is None:
            raise RuntimeError(
                "No MPI launcher was found. Install MPI and make sure `mpirun` or `mpiexec` is on your PATH."
            )

        cmd = [mpi_launcher, "-n", str(mpi_procs), str(self.lbm_loc), self.xml_name]
        print("Running:", " ".join(cmd))

        run_native_solver(cmd, self.folder_path, self.log_path)
        return read_two_phase_summary(self.log_path)


class LattEasyUnsteadyRelativePermeability:
    def __init__(
        self,
        pore_obj,
        cpus=4,
        two_phase_solver_path=None,
        single_phase_solver_path=None,
        buffer_layers=4,
        rho_f1=2.0,
        rho_f2=2.0,
        rho_d=0.06,
        minimum_radius=3,
        num_pressure_steps=6,
        Gc=0.9,
        omega_f1=1.0,
        omega_f2=1.0,
        G_ads_f1_s1=-0.4,
        G_ads_f1_s2=0.0,
        G_ads_f1_s3=0.0,
        G_ads_f1_s4=0.0,
        convergence=1e-4,
        convergence_iter=500,
        max_iterations=30000,
        gif_iter=0,
        vtk_iter=5000,
        rho_f2_vtk=False,
        print_geom=True,
        print_stl=False,
        relperm_pressure=0.0005,
        relperm_max_iterations=500000,
        relperm_convergence=1e-6,
        relperm_save_vtks=False,
    ):
        self.two_phase_solver = (
            str(two_phase_solver_path)
            if two_phase_solver_path is not None
            else str(check_two_phase_install())
        )
        self.single_phase_solver = (
            str(single_phase_solver_path)
            if single_phase_solver_path is not None
            else str(check_single_phase_install())
        )
        self.cpus = cpus
        self.buffer_layers = buffer_layers
        self.rho_f1 = rho_f1
        self.relperm_pressure = relperm_pressure
        self.relperm_max_iterations = relperm_max_iterations
        self.relperm_convergence = relperm_convergence
        self.relperm_save_vtks = relperm_save_vtks
        self.xml_name = "unsteady_relperm.xml"

        create_folder("sims")
        self.folder_path = create_numbered_folder("sims/unsteady_relperm")
        self.input_path = self.folder_path / "input"
        self.output_path = self.folder_path / "output"
        self.relperm_path = self.folder_path / "relperm"
        create_folder(str(self.input_path))
        create_folder(str(self.output_path))
        create_folder(str(self.relperm_path))

        self.rock = 1 - np.transpose(pore_obj, (2, 0, 1))
        self.rock = erase_regions(self.rock)
        self.pore_mask = self.rock == 0

        self.two_phase_matrix = create_two_phase_simulation_matrix(
            self.rock,
            non_wetting_fraction=0.0,
            buffer_layers=buffer_layers,
        )
        self.two_phase_shape = self.two_phase_matrix.shape
        self.two_phase_geometry_name = (
            f"pore_{self.two_phase_shape[0]}_{self.two_phase_shape[1]}_{self.two_phase_shape[2]}.dat"
        )
        (self.input_path / self.two_phase_geometry_name).write_bytes(
            self.two_phase_matrix.tobytes()
        )

        fluid1_box = (1, 2, 1, self.two_phase_shape[1], 1, self.two_phase_shape[2])
        fluid2_box = (3, self.two_phase_shape[0], 1, self.two_phase_shape[1], 1, self.two_phase_shape[2])

        create_two_phase_input_file(
            self.folder_path / self.xml_name,
            self.two_phase_geometry_name,
            self.two_phase_shape,
            "output/",
            (False, False, False),
            rho_f1,
            rho_f2,
            0.0,
            0.0,
            Gc,
            omega_f1,
            omega_f2,
            G_ads_f1_s1,
            G_ads_f1_s2,
            G_ads_f1_s3,
            G_ads_f1_s4,
            convergence,
            convergence_iter,
            max_iterations,
            gif_iter,
            vtk_iter,
            rho_f2_vtk,
            print_geom,
            print_stl,
            fluid_from_geom=False,
            fluid1_box=fluid1_box,
            fluid2_box=fluid2_box,
            pressure_bc=True,
            rho_f1_i=rho_f1,
            rho_f2_i=rho_f2,
            num_pc_steps=num_pressure_steps,
            min_radius=minimum_radius,
            rho_d=rho_d,
        )

        self.log_path = self.folder_path / "two_phase.txt"
        self.table_path = self.relperm_path / "relperm_table.csv"
        self.plot_path = self.relperm_path / "pc_relperm_curve.png"
        self.preview_path = self.relperm_path / "final_state.png"

    def run_two_phase(self, mpi_procs=None):
        if mpi_procs is None:
            mpi_procs = self.cpus

        mpi_launcher = find_mpi_launcher()
        if mpi_launcher is None:
            raise RuntimeError(
                "No MPI launcher was found. Install MPI and make sure `mpirun` or `mpiexec` is on your PATH."
            )

        cmd = [mpi_launcher, "-n", str(mpi_procs), self.two_phase_solver, self.xml_name]
        print("Running:", " ".join(cmd))

        run_native_solver(cmd, self.folder_path, self.log_path)
        return read_two_phase_summary(self.log_path)

    def run_relperm(self, mpi_procs=None):
        if mpi_procs is None:
            mpi_procs = self.cpus

        two_phase_summary = self.run_two_phase(mpi_procs=mpi_procs)

        rho_files = sorted(self.output_path.glob("rho_f1_*.dat"))
        if not rho_files:
            raise RuntimeError(
                "The two-phase run finished without writing any `rho_f1_*.dat` states."
            )

        capillary_pressure = read_pressure_steps(self.output_path / "output.dat")
        if capillary_pressure.size != len(rho_files):
            raise RuntimeError(
                "The pressure schedule and saved density states do not match."
            )

        absolute_matrix = create_single_phase_simulation_matrix(self.rock)
        absolute_perm, absolute_log = run_single_phase_case(
            self.relperm_path / "absolute",
            "absolute_perm",
            absolute_matrix,
            self.single_phase_solver,
            mpi_procs,
            self.relperm_pressure,
            self.relperm_max_iterations,
            self.relperm_convergence,
            self.relperm_save_vtks,
        )

        wetting_saturation = []
        krw = []
        krnw = []
        wetting_perm = []
        non_wetting_perm = []
        final_state = None

        for run_index, rho_file in enumerate(rho_files):
            density = load_density_state(rho_file, self.two_phase_shape)
            if self.buffer_layers:
                density = density[self.buffer_layers:-self.buffer_layers, :, :]
            wetting_mask, non_wetting_mask = split_fluid_masks(
                density,
                self.rho_f1,
                self.pore_mask,
            )

            wetting_geometry = np.where(wetting_mask, 0, 1).astype(np.uint8)
            non_wetting_geometry = np.where(non_wetting_mask, 0, 1).astype(np.uint8)

            wetting_case = self.relperm_path / f"wetting_{run_index:03d}"
            non_wetting_case = self.relperm_path / f"non_wetting_{run_index:03d}"

            k_w, _ = run_single_phase_case(
                wetting_case,
                f"wetting_{run_index:03d}",
                create_single_phase_simulation_matrix(wetting_geometry),
                self.single_phase_solver,
                mpi_procs,
                self.relperm_pressure,
                self.relperm_max_iterations,
                self.relperm_convergence,
                self.relperm_save_vtks,
            )
            k_w = max(k_w, 0.0)
            k_nw, _ = run_single_phase_case(
                non_wetting_case,
                f"non_wetting_{run_index:03d}",
                create_single_phase_simulation_matrix(non_wetting_geometry),
                self.single_phase_solver,
                mpi_procs,
                self.relperm_pressure,
                self.relperm_max_iterations,
                self.relperm_convergence,
                self.relperm_save_vtks,
            )
            k_nw = max(k_nw, 0.0)

            wetting_perm.append(k_w)
            non_wetting_perm.append(k_nw)
            wetting_saturation.append(wetting_mask.sum() / self.pore_mask.sum())
            krw.append(k_w / absolute_perm if absolute_perm else np.nan)
            krnw.append(k_nw / absolute_perm if absolute_perm else np.nan)

            final_state = np.zeros_like(self.rock, dtype=int)
            final_state[self.rock == 1] = 0
            final_state[wetting_mask] = 1
            final_state[non_wetting_mask] = 2

        wetting_saturation = np.array(wetting_saturation, dtype=float)
        krw = np.array(krw, dtype=float)
        krnw = np.array(krnw, dtype=float)
        wetting_perm = np.array(wetting_perm, dtype=float)
        non_wetting_perm = np.array(non_wetting_perm, dtype=float)

        table = np.column_stack(
            [
                np.arange(len(rho_files), dtype=int),
                capillary_pressure,
                wetting_saturation,
                np.full(len(rho_files), absolute_perm),
                wetting_perm,
                non_wetting_perm,
                krw,
                krnw,
            ]
        )
        header = (
            "run,capillary_pressure_lu,wetting_saturation,"
            "absolute_permeability_lu2,wetting_permeability_lu2,"
            "non_wetting_permeability_lu2,krw,krnw"
        )
        np.savetxt(
            self.table_path,
            table,
            delimiter=",",
            header=header,
            comments="",
        )
        write_relperm_plot(
            self.plot_path,
            wetting_saturation,
            capillary_pressure,
            krw,
            krnw,
        )
        if final_state is not None:
            write_state_preview(self.preview_path, final_state)

        return {
            "two_phase": two_phase_summary,
            "absolute_permeability": absolute_perm,
            "wetting_saturation": wetting_saturation,
            "capillary_pressure": capillary_pressure,
            "krw": krw,
            "krnw": krnw,
            "table_path": self.table_path,
            "plot_path": self.plot_path,
            "preview_path": self.preview_path,
            "folder_path": self.folder_path,
            "two_phase_log": self.log_path,
            "absolute_log": absolute_log,
        }
