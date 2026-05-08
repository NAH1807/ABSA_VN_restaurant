# ABSA Vietnamese Restaurant Reviews

Dự án **Phân Tích Cảm Xúc Theo Khía Cạnh (ABSA) cho Tiếng Việt** cho tiếng Việt trong lĩnh vực đánh giá nhà hàng.  
Mục tiêu là phân tích cảm xúc theo từng khía cạnh (aspect) thay vì chỉ phân loại cảm xúc tổng thể của câu.

---

# 1. Giới thiệu đề tài

Trong thực tế, một câu đánh giá có thể chứa nhiều cảm xúc khác nhau đối với từng khía cạnh.

Ví dụ:

> "Đồ ăn ngon nhưng phục vụ chậm"

Phân tích đúng:
- FOOD → Positive
- SERVICE → Negative

---

### Mục tiêu bài toán
- Xác định các **aspect (khía cạnh)** trong câu
- Gán **sentiment (cảm xúc)** cho từng aspect
- Xử lý ngôn ngữ tiếng Việt trong domain nhà hàng

---

# 2. Dataset sử dụng

## VLSP 2018 ABSA Restaurant Dataset

- Dataset được sử dụng từ cuộc thi **VLSP 2018 - Aspect Based Sentiment Analysis tiếng Việt**.
- Link tải Dataset: https://github.com/ds4v/absa-vlsp-2018/tree/main/datasets

---

### Đặc điểm dataset:
- Ngôn ngữ: Tiếng Việt
- Domain: Nhà hàng (restaurant reviews)
- Dạng dữ liệu: câu + nhãn aspect + sentiment

---

### Các aspect chính:
- FOOD (Đồ ăn)
- SERVICE (Dịch vụ)
- PRICE (Giá cả)
- AMBIENCE (Không gian)
- LOCATION (Vị trí)

---

### Sentiment labels:
- Positive
- Negative
- Neutral

---

# 3. Kiến trúc mô hình

Dự án sử dụng mô hình kết hợp 3 thành phần:

---

## PhoBERT (Encoder chính)

- Mô hình Transformer pretrained cho tiếng Việt
- Hiểu ngữ nghĩa và ngữ cảnh tốt hơn BERT đa ngôn ngữ
- Làm embedding đầu vào cho mô hình

---

## BiLSTM (Sequence Modeling)

- Học quan hệ tuần tự giữa các token
- Kết hợp thông tin trái → phải và phải → trái
- Giúp cải thiện việc nhận diện aspect trong câu

---

## CRF (Conditional Random Field)

- Giải mã chuỗi nhãn BIO
- Đảm bảo output hợp lệ (không sai cấu trúc tag)
- Tối ưu nhãn cuối cùng theo toàn chuỗi

---

## Kiến trúc tổng thể
<img width="857" height="522" alt="image" src="https://github.com/user-attachments/assets/e8e70e3d-ff77-4e7f-8f71-bf9890f8e685" />

---
# 4. Cài đặt thư viện: 
pip install -r requirement.txt

---

# 5. Cấu trúc thư mục:
data: chứa dataset (train, dev, test) và các file sau khi đã xử lí dữ liệu
src: chưa file code model, train model.
demo: chứa phần demo cho bài toán


