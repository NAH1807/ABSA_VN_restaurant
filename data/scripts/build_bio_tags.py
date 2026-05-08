import os
import re
import pandas as pd

from underthesea import word_tokenize


# =========================
# PATHS
# =========================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

INPUT_DIR = os.path.join(
    SCRIPT_DIR,
    "..",
    "processed",
    "sentiment_classification"
)

OUTPUT_DIR = os.path.join(
    SCRIPT_DIR,
    "..",
    "processed",
    "aspect_extraction"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


ASPECT_KEYWORDS = {
    "FOOD": ["đồ ăn", "món ăn", "thức ăn", "đồ uống", "món", "nước",
          "đồ", "mồi", "cơm", "bún", "phở", "hủ tiếu", "lẩu", "bbq", "nướng",
          "buffet", "trà sữa", "cafe", "cà phê", "trà", "sinh tố",
          "pizza", "hamburger", "gà", "heo", "bò", "hải sản",
          "ngon", "dở", "mặn", "ngọt", "chua", "cay", "nhạt", "béo",
          "menu", "portion", "khẩu phần", "size", "topping", "ăn", "uống"],

    "SERVICE": ["phục vụ", "nhân viên", "ship", "order",
            "phục vụ viên", "bồi bàn", "staff", "waiter", "waitress",
            "quản lý", "thu ngân",
            "giao hàng", "shipper", "delivery",
            "đặt món", "lên món", "mang món", "book bàn", "đặt bàn",
            "thái độ", "hỗ trợ", "tư vấn", "take care", "chăm sóc khách hàng"],

    "PRICE": ["giá", "tiền", "rẻ", "đắt",
          "giá cả", "chi phí", "bill", "hóa đơn",
          "combo", "khuyến mãi", "voucher", "discount", "sale",
          "mắc", "bình dân", "hợp lý", "giá sinh viên", "phải chăng",
          "cao", "thấp",
          "thanh toán", "cash", "momo", "chuyển khoản"],

    "AMBIENCE": ["không gian", "view", "decor", "quán",
             "bàn ghế", "máy lạnh", "điều hòa", "ánh sáng", "âm nhạc", "mùi", "wifi",
             "trang trí", "setup", "concept", "background",
             "đông", "ồn", "yên tĩnh", "sạch", "dơ", "thoáng", "chật",
             "checkin", "sống ảo", "view đẹp"],

    "LOCATION": ["vị trí", "địa điểm",
             "trung tâm", "mặt tiền", "gần", "xa", "đường", "hẻm",
             "gửi xe", "bãi xe", "parking",
             "dễ tìm", "khó tìm", "thuận tiện",
             "google map", "map"]
}


def tokenize(text):
    return word_tokenize(text, format="word")

# =========================
# FIND SPAN (IMPORTANT FIX)
# =========================

def find_span(tokens, keyword):
    """Return (start, end) index of keyword in tokens"""
    keyword_tokens = word_tokenize(keyword, format="word")

    n = len(tokens)
    m = len(keyword_tokens)

    for i in range(n - m + 1):
        if [t.lower() for t in tokens[i:i+m]] == [k.lower() for k in keyword_tokens]:
            return i, i + m - 1

    return None

# =========================
# BIO TAGGING
# =========================

def create_bio(tokens, spans):
    labels = ["O"] * len(tokens)

    for start, end in spans:
        labels[start] = "B-ASP"
        for i in range(start + 1, end + 1):
            labels[i] = "I-ASP"

    return labels

# =========================
# PROCESS
# =========================

def process_file(file_name):
    path = os.path.join(INPUT_DIR, file_name)
    df = pd.read_csv(path)

    total_row = len(df)
    kept_row = 0

    rows = []

    for _, row in df.iterrows():
        sentence = row["sentence"]
        aspect = row["aspect"]

        tokens = tokenize(sentence)

        spans = []

        for keyword in ASPECT_KEYWORDS.get(aspect.upper(), []):
            span = find_span(tokens, keyword)
            if span:
                spans.append(span)

        # nếu không có aspect match → bỏ
        if not spans:
            continue

        labels = create_bio(tokens, spans)

        rows.append({
            "tokens": tokens,
            "labels": labels
        })

        kept_row += 1

    print(f"[{file_name}] total_row = {total_row} | changed_row = {total_row - kept_row}")

    return rows

# =========================
# SAVE
# =========================

def save(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in rows:
            for t, l in zip(item["tokens"], item["labels"]):
                f.write(f"{t}\t{l}\n")
            f.write("\n")

# =========================
# MAIN
# =========================

def main():
    for split in ["train", "dev", "test"]:
        rows = process_file(f"{split}.csv")
        save(rows, os.path.join(OUTPUT_DIR, f"{split}.txt"))
        print(f"Saved {split}")

if __name__ == "__main__":
    main()