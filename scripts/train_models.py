import argparse
from backend.app.data import load_blast_csv
from backend.app.models import train_regressor

parser = argparse.ArgumentParser(description="Train a separate AI model against a measured blast outcome.")
parser.add_argument("--csv", required=True)
parser.add_argument("--target", required=True, help="Example: PPV_actual or X80_actual")
parser.add_argument("--model-dir", default="models")
args = parser.parse_args()

df = load_blast_csv(args.csv)
print(train_regressor(df, args.target, args.model_dir))
