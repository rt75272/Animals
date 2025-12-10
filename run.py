# run.py
import argparse
import sys
from pathlib import Path

# Add root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

def main():
    parser = argparse.ArgumentParser(description="Run stages of the animal classification project.")
    parser.add_argument(
        "stage",
        choices=["prepare_data", "train", "run_app"],
        help="The stage to run."
    )
    args = parser.parse_args()

    if args.stage == "prepare_data":
        print("Running data preparation...")
        from src.data_preparation import prepare
        prepare.prepare_data()
        print("Data preparation complete.")
    elif args.stage == "train":
        print("Running model training...")
        from src.training import train
        train.train_model()
        print("Model training complete.")
    elif args.stage == "run_app":
        print("Running the web application...")
        from src.webapp.app import app, initialize_app
        initialize_app()
        app.run(host='0.0.0.0', port=5000, debug=True)
