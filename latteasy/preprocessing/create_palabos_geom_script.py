import subprocess
import os

from .create_geom_for_palabos import *
# from create_palabos_input_file import *
from .parse_input_file import *
from pathlib import Path

# Directory containing this Python script
preprocessing_dir = Path(__file__).resolve().parent

input_file = preprocessing_dir / 'input.yml'

def create_palabos_geom_script():
    # 1. Read the updated input.yml
    inputs = parse_input_file(input_file)

    if inputs["simulation type"] == "2-phase":
        raise KeyError(
            "Simulation type set to 2-phase; please change to 1-phase."
        )

    sim_directory = inputs["input output"]["simulation directory"]

    # 2. Generate Palabos geometry
    print("Creating efficient geometry for Palabos...")
    create_geom_for_palabos(inputs)

    print("Palabos geometry generation completed.")


if __name__ == "__main__":
    create_palabos_geom_script()
