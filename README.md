# 🇻🇳 Aspect-Based Sentiment Analysis (ABSA) for Vietnamese Restaurant Reviews

## 📌 Giới thiệu

Dự án này xây dựng hệ thống **Aspect-Based Sentiment Analysis (ABSA)** cho tiếng Việt, tập trung vào bài toán **trích xuất khía cạnh (Aspect Extraction)** từ các câu đánh giá nhà hàng.

Mục tiêu của hệ thống là xác định các **khía cạnh (aspects)** xuất hiện trong câu (ví dụ: *giá cả, chất lượng món ăn, dịch vụ...*) dựa trên mô hình học sâu.

---

## 🧠 Kiến trúc mô hình

Mô hình được đề xuất theo kiến trúc:

<img width="857" height="522" alt="image" src="https://github.com/user-attachments/assets/cc4d0906-b3c4-4c5e-8275-501d966d97ff" />


### 🔷 Vai trò từng thành phần:

- **PhoBERT**:  
  Mô hình ngôn ngữ tiền huấn luyện cho tiếng Việt, giúp trích xuất biểu diễn ngữ nghĩa mạnh từ câu đầu vào.

- **BiLSTM (Bidirectional LSTM)**:  
  Học mối quan hệ tuần tự giữa các token theo cả hai chiều, giúp hiểu rõ ngữ cảnh trong câu.

- **CRF (Conditional Random Field)**:  
  Tối ưu chuỗi nhãn BIO, đảm bảo tính hợp lệ của đầu ra (giảm lỗi gán nhãn rời rạc)
