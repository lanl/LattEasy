"""Command-line entry points for LattEasy."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys

from ._native import build_solver, find_cmake, find_mpi_launcher, find_solver_executable


def _package_present(module_name):
    return importlib.util.find_spec(module_name) is not None


def _print_status_rows(rows):
    label_width = max(len(label) for label, _, _ in rows)
    state_width = max(len(state) for _, state, _ in rows)
    for label, state, detail in rows:
        print(f"{label:<{label_width}}  {state:<{state_width}}  {detail}")


def _doctor(_args):
    cmake = find_cmake()
    solver_missing = False
    rows = [
        ("python", "ok", sys.version.split()[0]),
        ("numpy", "ok" if _package_present("numpy") else "missing", "array math"),
        ("pandas", "ok" if _package_present("pandas") else "missing", "result parsing"),
        ("scipy", "ok" if _package_present("scipy") else "missing", "distance transforms"),
        ("skimage", "ok" if _package_present("skimage") else "missing", "connected-region cleanup"),
        ("cmake", "ok" if cmake else "missing", cmake or "install CMake to build the solver"),
    ]

    mpi_launcher = find_mpi_launcher()
    rows.append(
        (
            "mpi",
            "ok" if mpi_launcher else "missing",
            mpi_launcher or "install an MPI runtime to run the solver",
        )
    )

    try:
        solver = find_solver_executable()
    except FileNotFoundError:
        solver_missing = True
        rows.append(("solver", "missing", "run `latteasy build`"))
    else:
        rows.append(("solver", "ok", str(solver)))

    print("LattEasy doctor")
    print("")
    _print_status_rows(rows)
    print("")

    missing_python = [
        name
        for name in ("numpy", "pandas", "scipy", "skimage")
        if not _package_present(name)
    ]
    if missing_python:
        print("Next step: install the Python dependencies with `pip install -e .`.")
    elif cmake is None or mpi_launcher is None:
        missing_tools = []
        if cmake is None:
            missing_tools.append("CMake")
        if mpi_launcher is None:
            missing_tools.append("MPI")
        print(f"Next step: install {', '.join(missing_tools)}, then run `latteasy build`.")
    elif solver_missing:
        print("Next step: run `latteasy build`, then try `latteasy demo`.")
    else:
        print("Next step: try `latteasy demo`.")
    return 0


def _build(args):
    try:
        solver = build_solver(jobs=args.jobs)
    except subprocess.CalledProcessError:
        print(
            "Native build failed. Confirm CMake, a working C++ compiler, and MPI development libraries are installed.",
            file=sys.stderr,
        )
        return 1
    except (FileNotFoundError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Solver built successfully: {solver}")
    print("Next step: run `latteasy demo`.")
    return 0


def _demo(args):
    missing_python = [
        name
        for name in ("numpy", "pandas", "scipy", "skimage")
        if not _package_present(name)
    ]
    if missing_python:
        print(
            "Python dependencies are missing. Install them with `pip install -e .` before running the demo.",
            file=sys.stderr,
        )
        return 1

    try:
        find_solver_executable()
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if find_mpi_launcher() is None:
        print(
            "No MPI launcher was found. Install MPI and make sure `mpirun` or `mpiexec` is on your PATH.",
            file=sys.stderr,
        )
        return 1

    try:
        from .demo import run_demo
    except ImportError as exc:
        print(f"Failed to import demo helpers: {exc}", file=sys.stderr)
        return 1

    try:
        result = run_demo(
            shape=(args.x, args.y, args.z),
            radius=args.radius,
            cpus=args.cpus,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Demo completed. Permeability: {result.permeability}")
    print(f"Simulation files: {result.folder_path}")
    print(f"VTK output: {result.folder_path / 'output'}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="latteasy",
        description="Simple tools for preparing and running approachable LBM simulations.",
    )
    subparsers = parser.add_subparsers(dest="command")

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Check whether the current machine is ready for a first simulation.",
    )
    doctor_parser.set_defaults(func=_doctor)

    build_cmd = subparsers.add_parser(
        "build",
        help="Build the bundled single-phase solver from this repository checkout.",
    )
    build_cmd.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Maximum number of parallel compile jobs. Defaults to your CPU count.",
    )
    build_cmd.set_defaults(func=_build)

    demo_parser = subparsers.add_parser(
        "demo",
        help="Run a small built-in permeability demo with friendly defaults.",
    )
    demo_parser.add_argument("--x", type=int, default=32, help="Domain size in x.")
    demo_parser.add_argument("--y", type=int, default=32, help="Domain size in y.")
    demo_parser.add_argument("--z", type=int, default=64, help="Domain size in z.")
    demo_parser.add_argument(
        "--radius",
        type=int,
        default=None,
        help="Channel radius. The default chooses a safe size automatically.",
    )
    demo_parser.add_argument(
        "--cpus",
        type=int,
        default=1,
        help="Number of MPI ranks to launch. Defaults to 1 for the gentlest first run.",
    )
    demo_parser.set_defaults(func=_demo)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return args.func(args)
