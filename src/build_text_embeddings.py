# src/build_text_embeddings.py

"""
Sinh text embedding cho từng sản phẩm rượu vang, để dùng cho
Content-Based Recommender "xịn" hơn:

- Text đầu vào: name + grape + country (có thể mở rộng thêm mô tả sau này).
- Model: paraphrase-multilingual-MiniLM-L12-v2 (hỗ trợ tiếng Việt + tiếng Anh).
- Output: data/processed/text_embeddings.npy (shape: [n_items, d_embed]).
"""

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
OUTPUT_DIR = os.path.join(DATA_DIR, "processed")
EMB_PATH = os.path.join(OUTPUT_DIR, "text_embeddings.npy")


def find_input_csv() -> str:
    candidates = [
        os.path.join(DATA_DIR, "processed", "wines_clean.csv"),
        os.path.join(DATA_DIR, "clean", "wines_clean.csv"),
        os.path.join(DATA_DIR, "clean", "wines_client.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            print(f"[Embeddings] Using dataset: {p}")
            return p
    raise FileNotFoundError("Không tìm thấy file dữ liệu sạch để sinh embedding.")


def main():
    csv_path = find_input_csv()
    df = pd.read_csv(csv_path)

    if "name" not in df.columns:
        raise ValueError("Dataset phải có cột 'name'.")

    # Chuẩn hoá các trường text
    for col in ["grape", "country"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
        else:
            df[col] = "Unknown"

    # Text mô tả sản phẩm dùng cho embedding
    # (sau này em có cột 'description' thì gộp thêm vào chuỗi này)
    texts = []
    for _, row in df.iterrows():
        name = str(row.get("name", ""))
        grape = str(row.get("grape", "Unknown"))
        country = str(row.get("country", "Unknown"))

        text = f"{name}. Giống nho: {grape}. Xuất xứ: {country}."
        texts.append(text)

    print(f"[Embeddings] Tổng số sản phẩm: {len(texts)}")

    # Model đa ngôn ngữ (hỗ trợ tiếng Việt)
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    print(f"[Embeddings] Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    # Encode theo batch
    print("[Embeddings] Encoding...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=False,  # sẽ tự chuẩn hoá sau ở recommender
    )

    print("[Embeddings] Embeddings shape:", embeddings.shape)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(EMB_PATH, embeddings)
    print(f"[DONE] Saved text embeddings → {EMB_PATH}")


if __name__ == "__main__":
    main()
