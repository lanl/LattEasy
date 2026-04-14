from latteasy.demo import run_demo


if __name__ == "__main__":
    result = run_demo()
    print(f"Permeability: {result.permeability}")
    print(f"Simulation files: {result.folder_path}")
