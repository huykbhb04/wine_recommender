# app.py
import streamlit as st
from src.recommender import WineRecommender

st.set_page_config(
    page_title="Wine Recommender",
    page_icon="🍷",
    layout="wide",
)

# ================== CSS TÔNG MÀU HÀI HOÀ ==================
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #f7f3f0 0%, #fdfaf7 100%);
    }
    .wine-card {
        padding: 1rem 1.1rem;
        border-radius: 1rem;
        background-color: #ffffff;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
        border: 1px solid #f0e1d6;
    }
    .wine-title {
        font-weight: 800;
        font-size: 1.15rem;
        margin-top: 0.4rem;
        margin-bottom: 0.25rem;
        color: #ffae42;  /* màu vàng cam sáng, nổi trên nền tối */
        text-shadow: 0 0 4px rgba(0,0,0,0.6);
    }
    .wine-meta {
       font-size: 0.95rem;
        color: #f0f0f0;  /* sáng hơn hẳn */
        margin-bottom: 0.2rem;
    }
    .wine-price {
        font-size: 1.05rem;
        font-weight: 800;
        color: #ff6b3d; /* giống tone màu giá bạn đang thấy */
        margin-top: 0.2rem;
        margin-bottom: 0.3rem;
        text-shadow: 0 0 4px rgba(0,0,0,0.7);
    }
    .similarity-badge {
        font-size: 0.95rem;              /* to hơn */
        font-weight: 700;                /* đậm */
        padding: 0.35rem 0.75rem;        /* dày hơn, nhìn sang hơn */
        border-radius: 999px;
        background-color: #ffecd9;       /* nền sáng */
        color: #b44025;                  /* đỏ rượu vang đậm */
        border: 1px solid #e8b095;       /* viền nhẹ */
        display: inline-block;
        margin-top: 0.5rem;
        text-shadow: 0 0 3px rgba(0,0,0,0.2);
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #4c1c2f;
        margin-bottom: 0.5rem;
    }
    .section-subtitle {
        font-size: 0.9rem;
        color: #777;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ========== LOAD RECOMMENDER ==========

@st.cache_resource
def load_recommender():
    return WineRecommender()

rec = load_recommender()

# ========== SESSION STATE KHỞI TẠO ==========
if "selected_product_name" not in st.session_state:
    st.session_state["selected_product_name"] = None
if "search_query" not in st.session_state:
    st.session_state["search_query"] = ""


def choose_product(name: str):
    """Chọn 1 sản phẩm làm base & clear search (dùng cho mọi nơi)."""
    st.session_state["selected_product_name"] = name
    st.session_state["search_query"] = ""  # clear ô tìm kiếm


def clear_all():
    """Về trang chủ: xoá lựa chọn & clear search."""
    st.session_state["selected_product_name"] = None
    st.session_state["search_query"] = ""


# ========== HÀM RENDER CARD / GRID ==========

def render_product_card(prod: dict, show_similarity: bool = False, button_prefix: str = ""):
    st.markdown("<div class='wine-card'>", unsafe_allow_html=True)

    # ẢNH Ở TRÊN
    img = prod.get("image_url")
    if img:
        st.image(img, use_container_width=True)
    else:
        st.image("https://placehold.co/400x500?text=No+Image", use_container_width=True)

    # TÊN SẢN PHẨM – NỔI BẬT
    st.markdown(
        f"<div class='wine-title'>{prod.get('name','(No name)')}</div>",
        unsafe_allow_html=True,
    )

    # DÒNG 1: country + grape
    meta_line_1 = []
    if prod.get("country"):
        meta_line_1.append(f"🌍 <b>{prod['country']}</b>")
    if prod.get("grape"):
        meta_line_1.append(f"🍇 <b>{prod['grape']}</b>")
    if meta_line_1:
        st.markdown(
            "<div class='wine-meta'>" + " &nbsp;•&nbsp; ".join(meta_line_1) + "</div>",
            unsafe_allow_html=True,
        )

    # DÒNG 2: alcohol + volume
    meta_line_2 = []
    alcohol = prod.get("alcohol")
    volume = prod.get("volume_ml")
    if alcohol is not None:
        try:
            meta_line_2.append(f"🍷 {float(alcohol):.1f}%")
        except Exception:
            meta_line_2.append(f"🍷 {alcohol}%")
    if volume is not None:
        meta_line_2.append(f"📦 {volume:.0f} ml")
    if meta_line_2:
        st.markdown(
            "<div class='wine-meta'>" + " &nbsp;•&nbsp; ".join(meta_line_2) + "</div>",
            unsafe_allow_html=True,
        )

    # GIÁ – RẤT NỔI
    price = prod.get("price")
    if price is not None:
        st.markdown(
            f"<div class='wine-price'>💰 {price:,.0f} ₫</div>",
            unsafe_allow_html=True,
        )

    # ĐỘ TƯƠNG ĐỒNG (nếu có)
    if show_similarity and prod.get("similarity") is not None:
        sim = prod["similarity"]
        st.markdown(
            f"<div class='similarity-badge'>Độ tương đồng: {sim:.3f}</div>",
            unsafe_allow_html=True,
        )

    # Link & nút chọn
    if prod.get("url"):
        st.markdown(f"[🔗 Xem chi tiết trên website]({prod['url']})")

    st.button(
        "Xem sản phẩm này & gợi ý tương tự",
        key=f"{button_prefix}_{prod['index']}",
        on_click=choose_product,
        args=(prod["name"],),
    )

    st.markdown("</div>", unsafe_allow_html=True)


def render_product_grid(products, show_similarity: bool = False, button_prefix: str = ""):
    """Hiển thị sản phẩm dạng grid: tối đa 3 sản phẩm / hàng."""
    if not products:
        st.info("Không có sản phẩm để hiển thị.")
        return

    cols_per_row = min(3, len(products))
    for i in range(0, len(products), cols_per_row):
        row = products[i : i + cols_per_row]
        cols = st.columns(len(row))
        for col, prod in zip(cols, row):
            with col:
                render_product_card(
                    prod,
                    show_similarity=show_similarity,
                    button_prefix=button_prefix,
                )


# ========== SIDEBAR (TÌM KIẾM + GỢI Ý 5 SẢN PHẨM) ==========

with st.sidebar:
    st.header("🔍 Tìm kiếm & cấu hình")

    # Ô tìm kiếm: binding trực tiếp vào session_state["search_query"]
    st.text_input(
        "Nhập tên (hoặc một phần tên) chai rượu rồi nhấn Enter",
        key="search_query",
    )

    query_current = st.session_state["search_query"].strip()

    # GỢI Ý TỐI ĐA 5 SẢN PHẨM: TÊN + HÌNH ẢNH
    if query_current and getattr(rec, "names", None):
        q_lower = query_current.lower()
        match_indices = [
            i for i, name in enumerate(rec.names)
            if q_lower in str(name).lower()
        ][:5]  # chỉ lấy tối đa 5 gợi ý

        if match_indices:
            st.markdown("**Gợi ý sản phẩm:**")
            with st.container():
                for idx in match_indices:
                    row = rec.df.iloc[idx]
                    name = row.get("name", "(No name)")
                    img_url = row.get("image_url", None)

                    c1, c2 = st.columns([1, 3])
                    with c1:
                        if img_url:
                            st.image(img_url, width=50)
                        else:
                            st.image("https://placehold.co/50x70?text=No+Img", width=50)
                    with c2:
                        st.button(
                            name,
                            key=f"suggest_{idx}",
                            on_click=choose_product,
                            args=(name,),
                        )

    top_k = st.slider(
        "Số sản phẩm tương tự muốn hiển thị",
        min_value=3,
        max_value=15,
        value=6,
        step=3,
    )
    st.caption("• Mỗi hàng tối đa 3 sản phẩm.")

    st.markdown("---")
    st.button("🧹 Về trang chủ (xoá lựa chọn & tìm kiếm)", on_click=clear_all)


# ========== NỘI DUNG CHÍNH ==========

st.title("🍷 Hệ thống gợi ý rượu vang")

selected_name = st.session_state["selected_product_name"]
query_current = st.session_state["search_query"].strip()

# =============== CASE 1: ĐÃ CHỌN 1 SẢN PHẨM (BASE) ===============
if selected_name:
    results, base = rec.recommend_similar(selected_name, top_k=top_k)

    if base is None:
        st.error("Không tìm thấy sản phẩm tương ứng. Vui lòng thử lại hoặc chọn sản phẩm khác.")
    else:
        st.markdown(
            "<div class='section-title'>🍇 Sản phẩm đang xem</div>",
            unsafe_allow_html=True,
        )

        # Sản phẩm gốc
        base_cols = st.columns([1, 2])
        with base_cols[0]:
            render_product_card(base, show_similarity=False, button_prefix="base_view")
        with base_cols[1]:
            st.write("")

        st.write("---")
        st.markdown(
            "<div class='section-title'>✨ Các sản phẩm tương tự</div>",
            unsafe_allow_html=True,
        )
        render_product_grid(results, show_similarity=True, button_prefix="sim_view")

# =============== CASE 2: CHƯA CHỌN SẢN PHẨM, NHƯNG CÓ TỪ KHOÁ TÌM KIẾM ===============
elif query_current:
    q_lower = query_current.lower()
    match_indices = [
        i for i, name in enumerate(rec.names)
        if q_lower in str(name).lower()
    ]

    if not match_indices:
        st.warning("Không tìm thấy sản phẩm nào chứa từ khoá bạn nhập.")
    else:
        st.markdown(
            "<div class='section-title'>🔎 Kết quả tìm kiếm</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='section-subtitle'>Chọn một sản phẩm bất kỳ để xem chi tiết và gợi ý tương tự.</div>",
            unsafe_allow_html=True,
        )

        products = [rec._row_to_dict(i) for i in match_indices]
        render_product_grid(products, show_similarity=False, button_prefix="search_result")

# =============== CASE 3: TRANG CHỦ (KHÔNG TÌM KIẾM, KHÔNG CHỌN SẢN PHẨM) ===============
else:
    st.markdown(
        "<div class='section-title'>💡 Có thể bạn sẽ thích</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='section-subtitle'>Một số sản phẩm gợi ý ban đầu (dựa trên phổ biến hoặc chọn ngẫu nhiên).</div>",
        unsafe_allow_html=True,
    )

    popular_list = rec.get_popular(top_k=top_k)
    render_product_grid(popular_list, show_similarity=False, button_prefix="homepage")
