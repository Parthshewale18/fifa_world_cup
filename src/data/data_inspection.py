from pathlib import Path
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def inspect_csv(path: Path) -> None:
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"  COULD NOT READ {path.name}: {e}")
        return

    print(f"\n--- {path.relative_to(RAW_DIR)} ---")
    print(f"  shape: {df.shape}")
    print(f"  columns: {list(df.columns)}")
    print(f"  sample row:\n{df.head(1).to_dict(orient='records')}")


def main() -> None:
    csv_files = sorted(RAW_DIR.rglob("*.csv"))
    if not csv_files:
        print(f"No CSVs found under {RAW_DIR}.")
        return

    print(f"Found {len(csv_files)} CSV files under {RAW_DIR}\n")
    for path in csv_files:
        inspect_csv(path)


if __name__ == "__main__":
    main()