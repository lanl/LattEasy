"""Helpers for locating and building the native LBM solver."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
SOURCE_ROOT = PROJECT_ROOT / "src"
PALABOS_ARCHIVE = SOURCE_ROOT / "palabos.zip"
PALABOS_DIR = SOURCE_ROOT / "palabos-master"
SINGLE_PHASE_DIR = SOURCE_ROOT / "single_phase"
BUILD_DIR = SINGLE_PHASE_DIR / "build"
MULTI_PHASE_DIR = SOURCE_ROOT / "multi_phase"
MULTI_PHASE_BUILD_DIR = MULTI_PHASE_DIR / "build"
GRAY_SINGLE_PHASE_DIR = SOURCE_ROOT / "gray_single_phase"
GRAY_SINGLE_PHASE_BUILD_DIR = GRAY_SINGLE_PHASE_DIR / "build"
COMMON_TOOL_DIRS = (
    Path("/opt/homebrew/bin"),
    Path("/usr/local/bin"),
)
WRAPPER_DIR = Path(tempfile.gettempdir()) / "latteasy-tool-wrappers"


def find_system_tool(*names):
    """Find a tool on PATH or in common package-manager install locations."""
    for name in names:
        location = shutil.which(name)
        if location:
            return location

    for directory in COMMON_TOOL_DIRS:
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return str(candidate)
    return None


def build_runtime_env(base_env=None):
    """Return an environment with common package-manager bin dirs prepended to PATH."""
    env = dict(os.environ if base_env is None else base_env)
    path_parts = env.get("PATH", "").split(os.pathsep) if env.get("PATH") else []
    prepend = []
    convert_wrapper = ensure_imagemagick_wrapper()
    if convert_wrapper is not None:
        prepend.append(str(convert_wrapper.parent))
    prepend.extend(str(directory) for directory in COMMON_TOOL_DIRS if directory.is_dir())
    merged = []
    for entry in prepend + path_parts:
        if entry and entry not in merged:
            merged.append(entry)
    env["PATH"] = os.pathsep.join(merged)
    return env


def ensure_imagemagick_wrapper():
    """Create a small `convert` wrapper that forwards to ImageMagick 7 `magick`."""
    if os.name == "nt":
        return None

    magick = find_system_tool("magick")
    if magick is None:
        return None

    WRAPPER_DIR.mkdir(parents=True, exist_ok=True)
    wrapper = WRAPPER_DIR / "convert"
    script = f'#!/bin/sh\nexec "{magick}" "$@"\n'
    if not wrapper.exists() or wrapper.read_text() != script:
        wrapper.write_text(script)
        wrapper.chmod(0o755)
    return wrapper


def find_cmake():
    """Return the CMake executable path, including common Homebrew locations."""
    return find_system_tool("cmake")


def solver_binary_name(stem="permeability"):
    """Return the platform-specific solver filename."""
    suffix = ".exe" if os.name == "nt" else ""
    return f"{stem}{suffix}"


def packaged_solver_path(stem="permeability"):
    return PACKAGE_ROOT / "bin" / solver_binary_name(stem)


def built_solver_path(source_dir=SINGLE_PHASE_DIR, stem="permeability"):
    return source_dir / solver_binary_name(stem)


def find_solver_executable(stem="permeability", source_dir=SINGLE_PHASE_DIR):
    """Return the first available compiled solver executable."""
    candidates = [
        packaged_solver_path(stem),
        built_solver_path(source_dir, stem),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No compiled `{stem}` solver was found. Run the matching build step from the repository root."
    )


def find_mpi_launcher():
    """Return an available MPI launcher."""
    return find_system_tool("mpirun", "mpiexec")


def ensure_palabos_sources():
    """Unpack Palabos when only the bundled archive is available."""
    if PALABOS_DIR.is_dir():
        return PALABOS_DIR
    if not PALABOS_ARCHIVE.is_file():
        raise FileNotFoundError(
            "Bundled Palabos sources were not found. Run this command from a LattEasy source checkout."
        )
    with zipfile.ZipFile(PALABOS_ARCHIVE) as archive:
        archive.extractall(SOURCE_ROOT)
    return PALABOS_DIR


def read_cmake_cache_value(cache_path, key):
    """Return a value from CMakeCache.txt if it exists."""
    if not cache_path.is_file():
        return None

    prefix = f"{key}:"
    with cache_path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            if line.startswith(prefix):
                return line.partition("=")[2].strip()
    return None


def ensure_fresh_cmake_build_dir(build_dir, source_dir):
    """Delete a stale CMake build tree if it points at another checkout."""
    cache_path = build_dir / "CMakeCache.txt"
    if not cache_path.is_file():
        return

    cached_source = read_cmake_cache_value(cache_path, "CMAKE_HOME_DIRECTORY")
    cached_build = read_cmake_cache_value(cache_path, "CMAKE_CACHEFILE_DIR")
    expected_source = str(source_dir.resolve())
    expected_build = str(build_dir.resolve())

    if cached_source == expected_source and (cached_build is None or cached_build == expected_build):
        return

    shutil.rmtree(build_dir, ignore_errors=True)


def build_solver(jobs=None):
    """Build the bundled single-phase solver and copy it into the package."""
    if not SINGLE_PHASE_DIR.is_dir():
        raise FileNotFoundError(
            "Native solver sources were not found. Run this command from a LattEasy source checkout."
        )

    cmake = find_cmake()
    if cmake is None:
        raise RuntimeError(
            "CMake is required to build the solver. Install it and run `latteasy build` again."
        )

    ensure_palabos_sources()
    ensure_fresh_cmake_build_dir(BUILD_DIR, SINGLE_PHASE_DIR)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    configure_cmd = [
        cmake,
        "-S",
        str(SINGLE_PHASE_DIR),
        "-B",
        str(BUILD_DIR),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CXX_STANDARD=17",
    ]
    env = build_runtime_env()
    subprocess.check_call(configure_cmd, env=env)

    build_cmd = [cmake, "--build", str(BUILD_DIR), "--config", "Release"]
    if jobs is None:
        jobs = os.cpu_count() or 1
    if jobs > 0:
        build_cmd.extend(["-j", str(jobs)])
    subprocess.check_call(build_cmd, env=env)

    solver = built_solver_path(SINGLE_PHASE_DIR)
    if not solver.is_file():
        raise FileNotFoundError(
            f"Build finished, but `{solver.name}` was not created where expected."
        )

    package_target = packaged_solver_path()
    package_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(solver, package_target)
    return package_target


def find_gray_permeability_executable():
    """Return the compiled single-phase gray permeability executable."""
    return find_solver_executable("gray_permeability", GRAY_SINGLE_PHASE_DIR)


def find_two_phase_executable():
    """Return the compiled two-phase Shan-Chen executable."""
    return find_solver_executable("ShanChen", MULTI_PHASE_DIR)


def build_two_phase_solver(jobs=None):
    """Build the bundled two-phase solver and copy it into the package."""
    if not MULTI_PHASE_DIR.is_dir():
        raise FileNotFoundError(
            "Two-phase solver sources were not found. Run this command from a LattEasy source checkout."
        )

    cmake = find_cmake()
    if cmake is None:
        raise RuntimeError(
            "CMake is required to build the two-phase solver. Install it and run the example again."
        )

    ensure_palabos_sources()
    ensure_fresh_cmake_build_dir(MULTI_PHASE_BUILD_DIR, MULTI_PHASE_DIR)
    MULTI_PHASE_BUILD_DIR.mkdir(parents=True, exist_ok=True)

    configure_cmd = [
        cmake,
        "-S",
        str(MULTI_PHASE_DIR),
        "-B",
        str(MULTI_PHASE_BUILD_DIR),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CXX_STANDARD=17",
    ]
    env = build_runtime_env()
    subprocess.check_call(configure_cmd, env=env)

    build_cmd = [cmake, "--build", str(MULTI_PHASE_BUILD_DIR), "--config", "Release"]
    if jobs is None:
        jobs = os.cpu_count() or 1
    if jobs > 0:
        build_cmd.extend(["-j", str(jobs)])
    subprocess.check_call(build_cmd, env=env)

    solver = built_solver_path(MULTI_PHASE_DIR, "ShanChen")
    if not solver.is_file():
        raise FileNotFoundError(
            f"Build finished, but `{solver.name}` was not created where expected."
        )

    package_target = packaged_solver_path("ShanChen")
    package_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(solver, package_target)
    return package_target


def build_gray_permeability_solver(jobs=None):
    """Build the single-phase gray permeability solver and copy it into the package."""
    if not GRAY_SINGLE_PHASE_DIR.is_dir():
        raise FileNotFoundError(
            "Gray single-phase solver sources were not found. Run this command from a LattEasy source checkout."
        )

    cmake = find_cmake()
    if cmake is None:
        raise RuntimeError(
            "CMake is required to build the gray permeability solver. Install it and run the example again."
        )

    ensure_palabos_sources()
    ensure_fresh_cmake_build_dir(GRAY_SINGLE_PHASE_BUILD_DIR, GRAY_SINGLE_PHASE_DIR)
    GRAY_SINGLE_PHASE_BUILD_DIR.mkdir(parents=True, exist_ok=True)

    configure_cmd = [
        cmake,
        "-S",
        str(GRAY_SINGLE_PHASE_DIR),
        "-B",
        str(GRAY_SINGLE_PHASE_BUILD_DIR),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CXX_STANDARD=17",
    ]
    env = build_runtime_env()
    subprocess.check_call(configure_cmd, env=env)

    build_cmd = [cmake, "--build", str(GRAY_SINGLE_PHASE_BUILD_DIR), "--config", "Release"]
    if jobs is None:
        jobs = os.cpu_count() or 1
    if jobs > 0:
        build_cmd.extend(["-j", str(jobs)])
    subprocess.check_call(build_cmd, env=env)

    solver = built_solver_path(GRAY_SINGLE_PHASE_DIR, "gray_permeability")
    if not solver.is_file():
        raise FileNotFoundError(
            f"Build finished, but `{solver.name}` was not created where expected."
        )

    package_target = packaged_solver_path("gray_permeability")
    package_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(solver, package_target)
    return package_target
