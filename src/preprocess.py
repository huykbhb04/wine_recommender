# src/preprocess.py

import pandas as pd
import numpy as np
import re
import os

INPUT = "data/raw/wines_raw.csv"
OUTPUT = "data/clean/wines_clean.csv"


def parse_price(text):
    """Tách giá tiền từ chuỗi phức tạp."""
    if pd.isna(text):
        return np.nan

    text = str(text)

    # Tìm phần giá theo pattern tiền VNĐ (bất kỳ dạng nào)
    price_match = re.search(r"([\d\.\,]+)\s*₫", text)
    if not price_match:
        return np.nan

    price_str = price_match.group(1)
    price_str = price_str.replace(".", "").replace(",", "")
    try:
        return float(price_str)
    except:
        return np.nan


def parse_vintage(text):
    """Tách năm sản xuất từ chuỗi như:
       '2019,2022 5.990.000 ₫'
       '2021 14.000.000 ₫'
       '2018'
       Hoặc không có năm -> NaN
    """
    if pd.isna(text):
        return np.nan

    text = str(text)

    # Lấy tất cả năm dạng 4 chữ số 19xx – 20xx
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", text)
    if not years:
        return np.nan

    # Nếu có nhiều năm, lấy năm đầu tiên (luôn đúng với format web)
    return float(years[0])


def parse_alcohol(text):
    """Tách nồng độ cồn '14.5%' → 14.5"""
    if pd.isna(text):
        return np.nan
    text = str(text).replace("%", "").strip()
    try:
        return float(text)
    except:
        return np.nan


def parse_volume(text):
    """'750ML' → 750"""
    if pd.isna(text):
        return np.nan
    text = text.upper().replace("ML", "")
    try:
        return float(text)
    except:
        return np.nan


def main():
    if not os.path.exists(INPUT):
        raise FileNotFoundError(f"Không tìm thấy file: {INPUT}")

    df = pd.read_csv(INPUT)

    # --- TIỀN XỬ LÝ ---
    df["price"] = df["price_raw"].apply(parse_price)
    df["vintage"] = df["price_raw"].apply(parse_vintage)

    df["alcohol"] = df["alcohol_raw"].apply(parse_alcohol)
    df["volume_ml"] = df["volume_raw"].apply(parse_volume)

    # Chuẩn hoá missing
    df["grape"] = df["grape"].fillna("Unknown")
    df["country"] = df["country"].fillna("Unknown")

    # Xuất file
    os.makedirs("data/clean", exist_ok=True)
    df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    print(f"[DONE] Saved cleaned data → {OUTPUT}")
    print(df.head())


if __name__ == "__main__":
    main()
