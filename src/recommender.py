# src/recommender.py

import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = "data"
FEATURES_PATH = os.path.join(DATA_DIR, "processed", "features.npy")
TEXT_EMB_PATH = os.path.join(DATA_DIR, "processed", "text_embeddings.npy")


def find_input_csv() -> str:
    candidates = [
        os.path.join(DATA_DIR, "processed", "wines_clean.csv"),
        os.path.join(DATA_DIR, "clean", "wines_clean.csv"),
        os.path.join(DATA_DIR, "clean", "wines_client.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            print(f"[Recommender] Using dataset: {p}")
            return p
    raise FileNotFoundError("Không tìm thấy file dữ liệu sạch cho recommender.")


class WineRecommender:
    """
    Hệ khuyến nghị lọc theo nội dung (Content-Based Filtering) NÂNG CAO:

    - Vector cấu trúc (structural features): price, alcohol, volume_ml,
      country, grape (đã MinMax + OneHot + xử lý Unknown trung tính).
    - Text embedding: mã hoá name + grape + country bằng SentenceTransformer.
    - Hợp nhất 2 loại feature thành 1 item profile mạnh mẽ hơn.
    - Độ tương đồng giữa 2 sản phẩm = Cosine Similarity giữa 2 vector profile.
    """

    def __init__(self):
        # 1. Dataset gốc
        csv_path = find_input_csv()
        self.df = pd.read_csv(csv_path)
        if "name" not in self.df.columns:
            raise ValueError("Dataset phải có cột 'name'.")

        # 2. Features cấu trúc (numeric + categorical)
        if not os.path.exists(FEATURES_PATH):
            raise FileNotFoundError(
                f"Không tìm thấy {FEATURES_PATH}. Hãy chạy: python -m src.build_features"
            )
        X_struct = np.load(FEATURES_PATH).astype(float)

        if X_struct.shape[0] != len(self.df):
            raise ValueError(
                f"Số dòng features ({X_struct.shape[0]}) != số sản phẩm ({len(self.df)}). "
                "Hãy kiểm tra lại preprocess + build_features."
            )

        # Chuẩn hoá L2 cho từng vector
        struct_norms = np.linalg.norm(X_struct, axis=1, keepdims=True)
        struct_norms[struct_norms == 0] = 1.0
        self.X_struct_norm = X_struct / struct_norms

        # 3. Text embeddings (có thể chưa tồn tại → fallback struct-only)
        self.have_text = False
        if os.path.exists(TEXT_EMB_PATH):
            X_text = np.load(TEXT_EMB_PATH).astype(float)
            if X_text.shape[0] == len(self.df):
                text_norms = np.linalg.norm(X_text, axis=1, keepdims=True)
                text_norms[text_norms == 0] = 1.0
                self.X_text_norm = X_text / text_norms
                self.have_text = True
                print("[Recommender] Loaded text embeddings:", X_text.shape)
            else:
                print(
                    "[Recommender][WARN] text_embeddings.npy không khớp số sản phẩm. "
                    "Tạm thời bỏ qua text embedding."
                )

        self.names = self.df["name"].astype(str).tolist()

        # 4. Hợp nhất 2 vector: Z = [w_struct * X_struct_norm, w_text * X_text_norm]
        if self.have_text:
            w_struct = 0.6
            w_text = 0.4
            Z = np.concatenate(
                [self.X_struct_norm * w_struct, self.X_text_norm * w_text], axis=1
            )
        else:
            Z = self.X_struct_norm

        norms = np.linalg.norm(Z, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.Z_norm = Z / norms

    # ----------------- TIỆN ÍCH NỘI BỘ -----------------

    def _row_to_dict(self, idx: int, similarity: float | None = None) -> dict:
        row = self.df.iloc[int(idx)]

        def safe_float(val):
            try:
                if pd.isna(val):
                    return None
                return float(val)
            except Exception:
                return None

        return {
            "index": int(idx),
            "name": str(row.get("name", "")),
            "url": row.get("url", ""),
            "image_url": row.get("image_url", ""),
            "country": row.get("country", "Unknown"),
            "grape": row.get("grape", "Unknown"),
            "price": safe_float(row.get("price", None)),
            "alcohol": safe_float(row.get("alcohol", None)),
            "volume_ml": safe_float(row.get("volume_ml", None)),
            "similarity": float(similarity) if similarity is not None else None,
        }

    def _find_index_by_name(self, query: str) -> int | None:
        """
        Tìm index sản phẩm gần với tên query nhất.
        Bước 1: exact / contains theo string.
        (Đủ tốt vì UI của em đã có gợi ý tên.)
        """
        if not query:
            return None

        q = query.strip().lower()
        names = self.df["name"].astype(str)
        lower = names.str.lower()

        exact = np.where(lower == q)[0]
        if len(exact) > 0:
            return int(exact[0])

        contains = np.where(lower.str.contains(q))[0]
        if len(contains) > 0:
            return int(contains[0])

        # fallback: lấy sản phẩm đầu tiên
        return 0

    # ----------------- API CHO APP -----------------

    def recommend_similar(self, query: str, top_k: int = 6):
        """
        Gợi ý các sản phẩm tương tự dựa trên tên sản phẩm (query).
        Trả về:
          - results: list[dict] các sản phẩm gợi ý
          - base: dict sản phẩm gốc
        """
        idx = self._find_index_by_name(query)
        if idx is None:
            return [], None

        base_vec = self.Z_norm[idx : idx + 1]  # shape (1, d)
        sims = cosine_similarity(base_vec, self.Z_norm)[0]  # shape (N,)

        # Không gợi ý chính nó
        sims[idx] = -1.0

        top_k = max(1, min(top_k, len(sims) - 1))
        top_idx = np.argpartition(-sims, top_k)[:top_k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        results = [self._row_to_dict(i, sims[i]) for i in top_idx]
        base = self._row_to_dict(idx, 1.0)

        return results, base

    def get_popular(self, top_k: int = 8):
        """
        Gợi ý "có thể bạn sẽ thích".
        Nếu có các cột rating / num_purchases thì ưu tiên,
        còn không thì tạm sort theo price giảm dần.
        """
        df = self.df.copy()

        score_cols = [
            c
            for c in ["num_purchases", "purchase_count", "rating", "num_reviews"]
            if c in df.columns
        ]

        if score_cols:
            df["_score"] = 0.0
            for c in score_cols:
                df["_score"] += df[c].fillna(0)
            df_sorted = df.sort_values("_score", ascending=False)
        else:
            if "price" in df.columns:
                df_sorted = df.sort_values("price", ascending=False)
            else:
                df_sorted = df

        idxs = df_sorted.head(top_k).index.tolist()
        return [self._row_to_dict(i, None) for i in idxs]


if __name__ == "__main__":
    rec = WineRecommender()
    print("Tổng số sản phẩm:", len(rec.df))
    results, base = rec.recommend_similar(rec.df["name"].iloc[0], top_k=5)
    print("Base:", base["name"])
    for r in results:
        print("  ->", r["name"], "| sim =", f"{r['similarity']:.3f}")
