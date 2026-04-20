\
import argparse
import pandas as pd
import joblib

from pubg_lib import train_pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path to train.csv")
    parser.add_argument("--model_out", default="artifacts/model.joblib", help="Output model path")
    parser.add_argument("--model_name", default="Ridge", help="Model name (default: Ridge)")
    parser.add_argument("--valid_size", type=float, default=0.2, help="Validation split ratio")
    args = parser.parse_args()

    train_df = pd.read_csv(args.train)
    result = train_pipeline(train_df, model_name=args.model_name, valid_size=args.valid_size)
    print(f"Validation MAE: {result.valid_mae:.5f}")

    joblib.dump(result.pipe, args.model_out)
    print(f"Saved model: {args.model_out}")

if __name__ == "__main__":
    main()
