import json
from pathlib import Path

from src.pipeline import run_pipeline

INPUT_PATH = "data/raw/project1_real_data.csv"


def main():
    final_output = run_pipeline(INPUT_PATH, on_step=print)
    print("\nDone.")
    print(f"Final output saved to: {final_output['_output_path']}")


if __name__ == "__main__":
    main()
