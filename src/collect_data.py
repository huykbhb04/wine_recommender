# src/collect_data.py
import asyncio
from pathlib import Path
import pandas as pd
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
import re
import time
from typing import List, Optional

# ================== CẤU HÌNH CƠ BẢN ==================

# Trang danh sách sản phẩm chính
BASE_LIST_URL = "https://grandcru.vn/san-pham/"
# Mẫu URL phân trang: /san-pham/page/2/, /san-pham/page/3/, ...
PAGED_LIST_URL = "https://grandcru.vn/san-pham/page/{page}/"

# Selector tìm link sản phẩm trong danh sách
PRODUCT_LINK_SELECTOR = "a.woocommerce-LoopProduct-link"

RAW_DATA_PATH = Path("data/raw/wines_raw.csv")
MAX_CONCURRENT_REQUESTS = 3   # giảm bớt để đỡ nặng mạng & ít timeout hơn
MAX_PAGES = 97               # giới hạn số trang tối đa

PRODUCT_PAGE_TIMEOUT = 60000  # 60s


# ================== HÀM PHỤ ==================


async def get_product_links_from_category(page, category_url: str) -> List[str]:
    """
    Lấy tất cả link sản phẩm từ 1 trang danh mục.
    Trả về list URL. Nếu không tìm thấy sản phẩm nào → [].
    """
    print(f"[CATEGORY] {category_url}")
    try:
        await page.goto(category_url, wait_until="domcontentloaded", timeout=20000)
    except PlaywrightTimeout:
        print(f"[TIMEOUT] Category page: {category_url} -> skip this page")
        return []

    elements = await page.query_selector_all(PRODUCT_LINK_SELECTOR)
    print(f"  -> Found {len(elements)} product elements by selector '{PRODUCT_LINK_SELECTOR}'")

    links = set()
    for el in elements:
        href = await el.get_attribute("href")
        if href:
            if href.startswith("/"):
                href = "https://grandcru.vn" + href
            links.add(href)

    print(f"  -> Unique links: {len(links)}")
    return list(links)


def _extract_after_label(full_text: str, label: str) -> Optional[str]:
    """
    Từ full text của trang, lấy giá trị ngay sau 1 label.
    Ví dụ:

        Giống Nho
        Fiano, Chardonnay
        Dung tích
        750ML
        Nồng độ
        13.5%

    pattern: label + xuống dòng + dòng tiếp theo
    """
    pattern = rf"{re.escape(label)}\s+([^\n]+)"
    m = re.search(pattern, full_text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def _extract_price(full_text: str) -> Optional[str]:
    """
    Lấy cụm giá có ký tự ₫ trong body text.
    Ví dụ: '1.859.000 ₫'
    """
    matches = re.findall(r"([\d\.\s]+₫)", full_text)
    if not matches:
        return None
    # Lấy phần tử cuối cùng (thường là giá hiện tại)
    return matches[-1].strip()


async def parse_product(context, url: str, sem: asyncio.Semaphore) -> dict:
    """
    Truy cập trang chi tiết sản phẩm và lấy dữ liệu bằng cách đọc full body text.
    Nếu trang bị timeout → log và trả về bản ghi rỗng cho URL đó.
    """
    async with sem:
        print(f"[PRODUCT] {url}")
        page = await context.new_page()
        try:
            try:
                await page.goto(
                    url,
                    wait_until="domcontentloaded",  # nhẹ hơn networkidle
                    timeout=PRODUCT_PAGE_TIMEOUT,
                )
            except PlaywrightTimeout:
                print(f"[TIMEOUT] Product page: {url} -> skip this product")
                return {
                    "url": url,
                    "name": None,
                    "price_raw": None,
                    "alcohol_raw": None,
                    "volume_raw": None,
                    "vintage_raw": None,
                    "grape": None,
                    "country": None,
                    "image_url": None,
                }

            # Tên sản phẩm
            name_el = await page.query_selector("h1.product_title")
            if name_el:
                name = (await name_el.inner_text()).strip()
            else:
                name = None

            # Lấy body text
            body_text = await page.inner_text("body")
            raw_body = body_text
            body_text_compact = re.sub(r"\s+", " ", body_text)

            price_raw = _extract_price(body_text_compact)
            grape = _extract_after_label(raw_body, "Giống Nho")
            volume_raw = _extract_after_label(raw_body, "Dung tích")
            alcohol_raw = _extract_after_label(raw_body, "Nồng độ")
            country = _extract_after_label(raw_body, "Xuất xứ")

            # Ảnh sản phẩm
            img_el = await page.query_selector(
                ".woocommerce-product-gallery__image img, .wp-post-image"
            )
            image_url = None
            if img_el:
                image_url = await img_el.get_attribute("src")

            return {
                "url": url,
                "name": name,
                "price_raw": price_raw,
                "alcohol_raw": alcohol_raw,
                "volume_raw": volume_raw,
                "vintage_raw": None,  # Site ít show niên vụ → tạm None
                "grape": grape,
                "country": country,
                "image_url": image_url,
            }
        finally:
            await page.close()


# ================== HÀM CHÍNH ==================


async def collect_all():
    start_time = time.time()
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()

        page = await context.new_page()
        all_links = set()

        # 1) Trang đầu tiên: /san-pham/
        first_links = await get_product_links_from_category(page, BASE_LIST_URL)
        all_links.update(first_links)

        # 2) Các trang tiếp theo: /san-pham/page/2/, /page/3/, ...
        page_index = 2
        while page_index <= MAX_PAGES:
            url = PAGED_LIST_URL.format(page=page_index)
            links = await get_product_links_from_category(page, url)

            # Nếu không còn sản phẩm, coi như hết trang
            if not links:
                print(f"[STOP] Page {page_index} không còn sản phẩm. Dừng crawl phân trang.")
                break

            all_links.update(links)
            page_index += 1

        await page.close()

        all_links = list(all_links)
        print(f"[TOTAL PRODUCT LINKS] {len(all_links)}")

        if not all_links:
            print("[WARNING] Không tìm thấy link sản phẩm nào. Kiểm tra lại selector và URL.")
            await browser.close()
            pd.DataFrame([]).to_csv(RAW_DATA_PATH, index=False, encoding="utf-8-sig")
            print(f"[DONE] Saved EMPTY CSV to {RAW_DATA_PATH}")
            return

        print("[INFO] Ví dụ vài link đầu:")
        for link in all_links[:5]:
            print("   ", link)

        # 3) Crawl chi tiết từng sản phẩm song song (có hạn chế bằng semaphore)
        sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        tasks = [parse_product(context, url, sem) for url in all_links]
        results = await asyncio.gather(*tasks)

        await browser.close()

    # 4) Lọc bỏ các record hoàn toàn rỗng (URL bị timeout)
    df = pd.DataFrame(results)
    non_empty_mask = df["name"].notna() | df["price_raw"].notna()
    df = df[non_empty_mask]

    df.to_csv(RAW_DATA_PATH, index=False, encoding="utf-8-sig")
    print(f"[DONE] Saved {len(df)} rows to {RAW_DATA_PATH}")
    print(f"Time: {time.time() - start_time:.1f} seconds")


if __name__ == "__main__":
    asyncio.run(collect_all())
