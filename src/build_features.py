# src/build_features.py

import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

DATA_DIR = "data"
OUTPUT_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_FEATURES = os.path.join(OUTPUT_DIR, "features.npy")


def find_input_csv() -> str:
    """
    Tự tìm file dữ liệu sạch.
    """
    candidates = [
        os.path.join(DATA_DIR, "processed", "wines_clean.csv"),
        os.path.join(DATA_DIR, "clean", "wines_clean.csv"),
        os.path.join(DATA_DIR, "clean", "wines_client.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            print(f"[INFO] Found input CSV at: {p}")
            return p

    raise FileNotFoundError("Không tìm thấy file dữ liệu sạch.")


def main():
    # =========================
    # 1. Load dữ liệu
    # =========================
    input_csv = find_input_csv()
    df = pd.read_csv(input_csv)

    if "name" not in df.columns:
        raise ValueError("Dataset phải có cột name.")

    # Chuẩn hoá tên cột
    rename_map = {}
    if "price_final" in df.columns and "price" not in df.columns:
        rename_map["price_final"] = "price"
    if "alcohol_final" in df.columns and "alcohol" not in df.columns:
        rename_map["alcohol_final"] = "alcohol"
    if "volume_final" in df.columns and "volume_ml" not in df.columns:
        rename_map["volume_final"] = "volume_ml"

    if rename_map:
        df = df.rename(columns=rename_map)

    numeric_cols = [c for c in ["price", "alcohol", "volume_ml"] if c in df.columns]
    categorical_cols = [c for c in ["country", "grape"] if c in df.columns]

    print("=== BUILD FEATURES ===")
    print(f"Loaded {len(df)} rows")
    print("Numeric:", numeric_cols)
    print("Categorical:", categorical_cols)

    # =========================
    # 2. Chuẩn hoá Unknown
    # =========================
    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")
        df[col] = df[col].replace(
            {"": "Unknown", "NA": "Unknown", "nan": "Unknown", "NaN": "Unknown"}
        )

    # =========================
    # 3. Pipeline
    # =========================
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )

    # sklearn >= 1.2 → dùng sparse_output
    categorical_pipeline = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,   # <--- FIX LỖI Ở ĐÂY
        dtype=float,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    # =========================
    # 4. Fit + Transform
    # =========================
    X = preprocessor.fit_transform(df)
    print("X shape before unknown-fix :", X.shape)

    # =========================
    # 5. Neutral hóa Unknown
    # =========================
    if categorical_cols:
        ohe = preprocessor.named_transformers_["cat"]
        categories = ohe.categories_

        numeric_count = len(numeric_cols)
        col_start = numeric_count

        for i, cat_list in enumerate(categories):
            cat_list = list(cat_list)
            size = len(cat_list)

            if "Unknown" in cat_list:
                pos = cat_list.index("Unknown")
                col_unknown = col_start + pos

                X[:, col_unknown] = 0.0  # Neutral vector
                print(
                    f"[Neutralized] Unknown in column '{categorical_cols[i]}' → feature col {col_unknown}"
                )

            col_start += size

    print("X shape after unknown-fix :", X.shape)

    # =========================
    # 6. Save output
    # =========================
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(OUTPUT_FEATURES, X)
    print(f"[DONE] Saved features → {OUTPUT_FEATURES}")


if __name__ == "__main__":
    main()
