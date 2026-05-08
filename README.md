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
