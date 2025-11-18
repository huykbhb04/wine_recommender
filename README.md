# 📘 HỆ THỐNG GỢI Ý RƯỢU VANG 

## 1. Giới thiệu tổng quan

Hệ thống được xây dựng nhằm thu thập dữ liệu rượu vang từ website thương
mại điện tử, xử lý và chuẩn hóa dữ liệu, tạo vector đặc trưng (features)
cho từng sản phẩm và xây dựng hệ khuyến nghị theo nội dung
(Content-Based Filtering) kết hợp Text Embedding như các hệ thống
recommender của Tiki/Shopee.

Sản phẩm cuối gồm: - Bộ dữ liệu sạch gồm hơn 2300 sản phẩm rượu vang -
Hệ thống gợi ý dựa trên đặc trưng cấu trúc + ngôn ngữ (embedding) - Ứng
dụng web (Streamlit UI) đẹp và thân thiện

------------------------------------------------------------------------

## 2. Công nghệ sử dụng

### **Ngôn ngữ & Framework**

-   Python 3.10+
-   Streamlit (giao diện web)
-   Playwright (web scraping)
-   Pandas / NumPy (xử lý dữ liệu)
-   scikit-learn (PCA, Scaling, OneHotEncoder)
-   sentence-transformers (text embedding)

### **Thư viện chính**

``` bash
playwright
pandas
numpy
scikit-learn
sentence-transformers
tqdm
streamlit
```

------------------------------------------------------------------------

## 3. Pipeline dữ liệu

### **3.1 Web Scraping (collect_data.py)**

-   Sử dụng Playwright + asyncio để cào song song (concurrency).
-   Thu thập các trường:
    -   `url`, `name`, `price`, `alcohol`, `volume`, `grape`, `country`,
        `image_url`.
-   Tự động phân trang từ trang 1 → 97.
-   Loại bỏ lỗi định dạng (ví dụ: `2019,2022` tách sai về giá).

------------------------------------------------------------------------

### **3.2 Tiền xử lý dữ liệu (preprocess.py)**

-   Loại bỏ ký tự ("₫", "%", "ML").
-   Chuẩn hóa trường:
    -   `price → float`
    -   `alcohol → float`
    -   `volume_ml → float`
-   Xử lý thiếu:
    -   Numeric → Median Imputation
    -   Categorical → `"Unknown"`

------------------------------------------------------------------------

### **3.3 Xây dựng đặc trưng (build_features.py)**

Bao gồm:

#### **3.3.1 Numeric features**

-   Price
-   Alcohol
-   Volume (ml)

Áp dụng: - MinMaxScaler

#### **3.3.2 Categorical features**

-   Country
-   Grape

Áp dụng: - OneHotEncoder - Unknown được mã hóa trung tính (không chi
phối mô hình)

------------------------------------------------------------------------

### **3.4 Text Embedding (build_text_embeddings.py)**

Sử dụng mô hình:

    sentence-transformers/all-MiniLM-L6-v2

Embedding các trường: - name - grape - country

Tạo vector độ dài 384 chiều cho mỗi sản phẩm.

------------------------------------------------------------------------

## 4. Thiết kế hệ khuyến nghị

### **4.1 Content-Based Filtering (theo giáo trình)**

Tìm độ tương đồng giữa các sản phẩm dựa trên đặc điểm của chính sản
phẩm.

### **4.2 Vector đặc trưng hợp nhất**

Z = \[0.6 × Structural Features\] + \[0.4 × Text Embeddings\]

### **4.3 Độ đo tương đồng**

-   Cosine Similarity\
-   Giá trị càng gần 1 → càng giống nhau.

### **4.4 Xử lý Unknown thông minh**

Unknown được mã hóa bằng vector trung lập (vì Unknown không phải đặc
tính thật của sản phẩm).

------------------------------------------------------------------------

## 5. Ứng dụng Web (Streamlit UI)

Bao gồm 3 chức năng chính:

### **5.1 Trang chủ --- Gợi ý nổi bật**

-   Top sản phẩm giá trị cao, nhiều tiêu chí nổi bật.
-   Hiển thị dạng lưới đẹp (3 sản phẩm / hàng).

### **5.2 Tìm kiếm sản phẩm**

-   Tự động gợi ý tên + hình ảnh (autocomplete 5 sản phẩm gần nhất).
-   Nhấn Enter → hiển thị danh sách sản phẩm khớp từ khóa.

### **5.3 Trang chi tiết sản phẩm**

-   Hiển thị ảnh + thông tin đầy đủ.
-   Dưới đó: "Các sản phẩm tương tự"
-   Có thể click tiếp từng sản phẩm → chuyển trang.

------------------------------------------------------------------------

## 6. Cấu trúc thư mục dự án

    wine_recommender/
    │── app.py
    │── requirements.txt
    │── src/
    │   ├── collect_data.py
    │   ├── preprocess.py
    │   ├── build_features.py
    │   ├── build_text_embeddings.py
    │   └── recommender.py
    │
    ├── data/
    │   ├── raw/
    │   │   └── wines_raw.csv
    │   ├── clean/
    │   │   └── wines_clean.csv
    │   └── processed/
    │       ├── features.npy
    │       └── text_embeddings.npy

------------------------------------------------------------------------

## 7. Hướng dẫn cài đặt & cấu hình

### **7.1 Cài Python**

Tải Python 3.10--3.12 từ python.org.

### **7.2 Cài Playwright**

``` bash
pip install playwright
playwright install
```

### **7.3 Cài các thư viện khác**

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 8. Hướng dẫn chạy toàn bộ hệ thống

### **8.1 Bước 1 --- Thu thập dữ liệu**

``` bash
python -m src.collect_data
```

### **8.2 Bước 2 --- Tiền xử lý**

``` bash
python -m src.preprocess
```

### **8.3 Bước 3 --- Xây dựng đặc trưng**

``` bash
python -m src.build_features
```

### **8.4 Bước 4 --- Tạo text embedding**

``` bash
python -m src.build_text_embeddings
```

### **8.5 Bước 5 --- Chạy giao diện web**

``` bash
streamlit run app.py
```

------------------------------------------------------------------------



