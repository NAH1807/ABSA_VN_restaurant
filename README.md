# ABSA Vietnamese Restaurant - Aspect-Based Sentiment Analysis

Dự án phân tích cảm xúc dựa trên khía cạnh (Aspect-Based Sentiment Analysis - ABSA) cho các đánh giá nhà hàng Việt Nam. Hệ thống này không chỉ xác định cảm xúc chung mà còn phân tích chi tiết cảm xúc theo từng khía cạnh cụ thể như chất lượng thức ăn, dịch vụ, giá cả, không gian, v.v.

## 📋 Giới Thiệu về Dataset

### Nguồn dữ liệu
Dataset được thu thập từ các đánh giá thực tế của khách hàng trên các nền tảng review nhà hàng Việt Nam, bao gồm:
- Các bình luận và đánh giá từ người dùng
- Rating (từ 1-5 sao) liên quan đến từng khía cạnh
- Nhận xét tự do về trải nghiệm tại nhà hàng

### Đặc điểm Dataset
- **Ngôn ngữ**: Tiếng Việt
- **Miền ứng dụng**: Đánh giá nhà hàng
- **Kích thước**: Chứa hàng nghìn bình luận có annotation
- **Cấu trúc**: Dữ liệu được gán nhãn với các khía cạnh và cảm xúc tương ứng

### Các Khía Cạnh (Aspects) Chính
- 🍽️ **Chất lượng thức ăn** (Food Quality): Hương vị, độ tươi sống, chất lượng nguyên liệu
- 👨‍💼 **Dịch vụ** (Service): Thái độ nhân viên, tốc độ phục vụ, sự chuyên nghiệp
- 💰 **Giá cả** (Price): Mức giá phù hợp, chất lượng so với giá
- 🏢 **Không gian/Môi trường** (Ambiance): Trang trí, độ sạch sẽ, tiếng ồn
- 🚗 **Đỗ xe** (Parking): Điều kiện đỗ xe, an toàn
- 🕐 **Thời gian chờ đợi** (Wait Time): Tốc độ phục vụ, thời gian chờ

### Phân loại Cảm xúc (Sentiments)
- **Positive** (Tích cực) 😊: Khách hàng thích, hài lòng
- **Negative** (Tiêu cực) 😞: Khách hàng không hài lòng, có than phiền
- **Neutral** (Trung tính) 😐: Mô tả khách quan, không có cảm xúc rõ rệt

## 🏗️ Cấu Trúc Chung Model

### Kiến trúc Tổng Quát

```
┌─────────────────────────────────────────────────────┐
│         Input: Review Text (Tiếng Việt)             │
└────────────────────┬────────────────────────────────┘
                     │
        ┌────────────▼─────────────┐
        │   Text Preprocessing     │
        │  - Tokenization          │
        │  - Lowercasing           │
        │  - Remove special chars   │
        └────────────┬─────────────┘
                     │
        ┌────────────▼──────────────┐
        │   Word Embedding Layer    │
        │  (Word2Vec/FastText/BERT) │
        └────────────┬──────────────┘
                     │
        ┌────────────▼──────────────────────┐
        │   Feature Extraction              │
        │  - Encoder (LSTM/BiLSTM/Transformer)
        │  - Context Understanding         │
        └────────────┬──────────────────────┘
                     │
        ┌────────────▼──────────────────────┐
        │   Aspect Detection Layer          │
        │  - Identify aspects in text       │
        │  - Aspect-specific features      │
        └────────────┬──────────────────────┘
                     │
        ┌────────────▼──────────────────────┐
        │   Sentiment Classification        │
        │  - Per-aspect sentiment score     │
        │  - Output layer (3 classes)       │
        └────────────┬──────────────────────┘
                     │
        ┌────────────▼──────────────────────┐
        │         Output Results             │
        │  - Aspect labels                 │
        │  - Sentiment predictions         │
        │  - Confidence scores             │
        └──────────────────────────────────┘
```

### Các Thành Phần Chính

1. **Preprocessing Module**
   - Làm sạch văn bản (remove noise, special characters)
   - Tokenization: tách từ
   - Chuẩn hóa (lowercasing, accent normalization)

2. **Embedding Layer**
   - Chuyển đổi từ thành vector số
   - Các tùy chọn: Word2Vec, FastText, BERT, PhoBERT (cho tiếng Việt)

3. **Encoder Layer**
   - BiLSTM, Transformer hoặc CNN
   - Học các đặc trưng ngữ cảnh từ văn bản

4. **Aspect Extraction Module**
   - Xác định các khía cạnh xuất hiện trong review
   - Sử dụng attention mechanism để focus vào khía cạnh cụ thể

5. **Sentiment Classifier**
   - Phân loại cảm xúc cho mỗi khía cạnh
   - Đầu ra: [Negative, Neutral, Positive]

## 🔑 Các Khái Niệm Chính

### 1. **ABSA - Aspect-Based Sentiment Analysis**
Phương pháp phân tích cảm xúc ở mức độ chi tiết hơn so với Sentiment Analysis thông thường:
- **Sentiment Analysis thông thường**: "Review này tích cực hay tiêu cực?"
- **ABSA**: "Khía cạnh nào là tích cực? Khía cạnh nào là tiêu cực?"

**Ví dụ:**
```
Review: "Thức ăn rất ngon nhưng dịch vụ kém và giá hơi đắt"

Sentiment Analysis: Cảm xúc = Hỗn hợp (Mixed)

ABSA:
├─ Thức ăn: Positive ✓
├─ Dịch vụ: Negative ✗
└─ Giá cả: Negative ✗
```

### 2. **Aspect Term Extraction**
Quá trình xác định các từ/cụm từ đề cập đến các khía cạnh trong review:
- **Input**: "Cơm chiên rất ngon nhưng chỗ ngồi hơi chật"
- **Output**: ["cơm chiên", "chỗ ngồi"]

### 3. **Aspect Category Detection**
Gán nhãn khía cạnh cho mỗi aspect term:
- "cơm chiên" → Food/Thức ăn
- "chỗ ngồi" → Ambiance/Không gian

### 4. **Opinion Expression**
Xác định các từ/cụm từ biểu thị cảm xúc liên quan đến khía cạnh:
- "rất ngon" (tích cực)
- "hơi chật" (tiêu cực)

### 5. **Attention Mechanism**
Cơ chế để model tập trung vào các phần quan trọng của text:
```
Ví dụ: "Cơm chiên rất ngon"
              ↑ ↑ ↑ ↑ ↑
           Trọng số attention
```
Model học để tập trung cao vào "rất ngon" khi phân tích khía cạnh "cơm chiên".

### 6. **BiLSTM - Bidirectional Long Short-Term Memory**
Mạng neural xử lý chuỗi văn bản theo cả hai hướng:
```
Forward:  "Thức ăn" ─→ "rất" ─→ "ngon" ─→
Backward: "Thức ăn" ←─ "rất" ←─ "ngon" ←─

Output: Vector kết hợp cả hai hướng
```

### 7. **Embedding Vector**
Cách biểu diễn từ dưới dạng vector số:
```
Ví dụ: "ngon" = [0.2, -0.5, 0.8, 0.1, ...]
       "tệ"  = [0.1, -0.6, 0.7, -0.2, ...]
       
Các từ tương tự sẽ có vector gần nhau
```

### 8. **F1-Score, Precision, Recall**
Các metric đánh giá hiệu suất model:
- **Precision**: Trong những gì model dự đoán là tích cực, có bao nhiêu % đúng?
- **Recall**: Trong tất cả cảm xúc tích cực thực tế, model tìm ra bao nhiêu %?
- **F1-Score**: Điểm cân bằng giữa Precision và Recall

## 📊 Cấu Trúc Thư Mục

```
ABSA_VN_restaurant/
├── README.md                 # Tài liệu này
├── data/
│   ├── raw/                 # Dữ liệu thô
│   ├── processed/           # Dữ liệu đã xử lý
│   └── annotations/         # Các tệp ghi chú
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_training.ipynb
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── utils.py
│   └── evaluate.py
├── models/                  # Các model đã lưu
├── results/                 # Kết quả và visualizations
└── requirements.txt
```

## 🚀 Bắt Đầu

### Yêu cầu
```bash
pip install -r requirements.txt
```

### Công nghệ Sử Dụng
- **Python 3.8+**
- **TensorFlow/PyTorch**: Deep Learning Framework
- **NLTK/spaCy**: NLP Library cho tiếng Việt
- **Scikit-learn**: Machine Learning utilities
- **Pandas/NumPy**: Data processing
- **Jupyter**: Notebook environment

---

**Tác giả**: NAH1807  
**Ngôn ngữ**: Python, Jupyter Notebook  
**Cấp độ**: Advanced NLP Project
