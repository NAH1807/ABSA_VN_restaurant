import os
import sys
import torch

from transformers import AutoTokenizer

# Add parent directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.predictor import (
    ABSAPredictor
)

from models.sentiment_model import (
    SentimentClassifier
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# =========================
# LABELS
# =========================

ID2LABEL = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}


# =========================
# ASPECT MAPPING
# =========================

ASPECT_MAPPING = {
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


# =========================
# PIPELINE
# =========================

class ABSAPipeline:
    def __init__(self):
        # Aspect extractor
        self.aspect_predictor = (
            ABSAPredictor()
        )
        # Tokenizer
        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                "vinai/phobert-base"
            )
        )
        # Sentiment model
        self.sentiment_model = (
            SentimentClassifier()
        )
        # Resolve sentiment checkpoint path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = os.path.join(
            script_dir,
            "..",
            "..",
            "checkpoints",
            "sentiment_model",
            "best_model.pt"
        )

        self.sentiment_model.load_state_dict(
            torch.load(
                checkpoint_path,
                map_location=DEVICE
            )
        )

        self.sentiment_model.to(DEVICE)
        self.sentiment_model.eval()

    # ======================
    # MAP ASPECT TERM
    # ======================

    def map_aspect(
        self,
        aspect_term
    ):
        aspect_term = (
            aspect_term.lower()
        )
        for aspect, keywords in (
            ASPECT_MAPPING.items()
        ):
            if aspect_term in keywords:

                return aspect

        return "OTHER"

    # ======================
    # PREDICT SENTIMENT
    # ======================

    def predict_sentiment(
        self,
        sentence,
        aspect
    ):
        text = (
            f"{aspect} [SEP] {sentence}"
        )
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        input_ids = encoding[
            "input_ids"
        ].to(DEVICE)

        attention_mask = encoding[
            "attention_mask"
        ].to(DEVICE)

        with torch.no_grad():

            logits = self.sentiment_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        prediction = torch.argmax(
            logits,
            dim=1
        ).item()

        return ID2LABEL[prediction]

    # ======================
    # FULL PIPELINE
    # ======================

    def predict(
        self,
        sentence
    ):
        results = []
        # Step 1:
        # Extract aspects
        aspect_terms = (
            self.aspect_predictor.predict(
                sentence
            )
        )

        # Remove duplicates
        aspect_terms = list(
            set(aspect_terms)
        )
        # Step 2:
        # Predict sentiment
        for term in aspect_terms:

            aspect = self.map_aspect(
                term
            )

            sentiment = (
                self.predict_sentiment(
                    sentence,
                    aspect
                )
            )
            results.append({
                "aspect": aspect,
                "term": term,
                "sentiment": sentiment
            })

        return results


# =========================
# TEST
# =========================

if __name__ == "__main__":
    pipeline = ABSAPipeline()
    sentence = (
        "Đồ ăn ngon nhưng phục vụ chậm"
    )
    results = pipeline.predict(
        sentence
    )
    print("\nABSA Results:\n")
    for item in results:
        print(item)