from pathlib import Path
import os
import subprocess
import shutil
import zipfile
from setuptools import setup, find_packages
from setuptools.command.install import install

def build_cpp():
    """Unpack Palabos and compile the permeability solver."""
    src_dir = Path(__file__).parent / "src"
    zip_path = src_dir / "palabos.zip"
    palabos_dir = src_dir / "palabos-master"

    build_dir = src_dir / "single_phase" / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    # Ensure required build tools exist before invoking them
    if shutil.which("cmake") is None:
        raise RuntimeError("cmake is required to build the C++ solver; install it via your package manager")
    if shutil.which("make") is None:
        raise RuntimeError("make is required to build the C++ solver")

    # Force a modern C++ standard to avoid build failures with recent
    # compilers.  Palabos requires at least C++17.
    subprocess.check_call(["cmake", "-DCMAKE_CXX_STANDARD=17", ".."], cwd=build_dir)
    subprocess.check_call(["make", f"-j{os.cpu_count() or 1}",], cwd=build_dir)

    # Copy the resulting executable into latteasy/bin so it gets packaged
    exe = src_dir / "single_phase" / "permeability"
    bin_dir = Path(__file__).parent / "latteasy" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exe, bin_dir / exe.name)

class InstallWithCPP(install):
    """Compile the C++ solver during installation."""

    def run(self):
        build_cpp()
        super().run()

setup(
    name="latteasy",
    version="0.1.0",
    description="Utilities for Lattice Boltzmann preprocessing and postprocessing",
    author="LattEasy Developers",
    packages=find_packages(include=["latteasy", "latteasy.*"]),
    python_requires=">=3.8",
    include_package_data=True,
    package_data={"latteasy": ["bin/*"]},
    cmdclass={"install": InstallWithCPP},
)
