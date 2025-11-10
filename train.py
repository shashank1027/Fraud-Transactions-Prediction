import os
import pandas as pd
from sklearn.ensemble import IsolationForest
from joblib import dump

def main():
    input_dir = os.environ["SM_CHANNEL_TRAINING"]

    output_dir = os.environ["SM_MODEL_DIR"]

    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]
    df = pd.concat([
        pd.read_parquet(f) if f.endswith(".parquet") else pd.read_csv(f)
        for f in files
    ])

    model = IsolationForest(
        n_estimators=150,
        contamination=0.05,
        random_state=42
    )

    model.fit(df)
    dump(model, os.path.join(output_dir, "model.joblib"))
    print(" Model saved:", os.path.join(output_dir, "model.joblib"))

if __name__ == "__main__":
    main()
