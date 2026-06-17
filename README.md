# Fetal Health Classification - Phân loại tình trạng sức khỏe thai nhi

Dự án bài tập lớn môn Học máy (Kỳ 2, 2025-2026). Dự án tập trung vào việc xây dựng, huấn luyện và so sánh các mô hình Machine Learning để thực hiện phân loại đa lớp (Multi-class Classification) cho tình trạng sức khỏe thai nhi dựa trên dữ liệu Cardiotocogram (CTG).

## 1. Giảng viên hướng dẫn & Thành viên nhóm
**Giảng viên hướng dẫn:** [Cao Văn Chung](http://mim.hus.vnu.edu.vn/en/staff/chungcv) - Khoa Toán - Cơ - Tin học, Trường Đại học Khoa học Tự nhiên, ĐHQGHN.

| Thành viên | Vai trò chính |
| :--- | :--- |
| **Đinh Bảo Triết** (Leader) | Data Preprocessing, SVM, Logistic Regression, Tổng hợp báo cáo |
| **Đỗ Minh Hoàng** | Dimensionality Reduction (PCA/LDA), SoftMax Regression |
| **Vũ Nhân Tông** | Clustering (K-Means), Naive Bayes |

## 2. Giới thiệu dự án
Trong lĩnh vực y tế dự phòng, việc theo dõi nhịp tim thai nhi là một trong những chỉ số quan trọng nhất để đánh giá sức khỏe thai nhi trong quá trình mang thai và chuyển dạ. Dự án này sử dụng tập dữ liệu Cardiotocogram (CTG) để xây dựng hệ thống phân loại tình trạng sức khỏe thai nhi thành 3 nhóm: 
- **Normal (Bình thường)**
- **Suspect (Nghi ngờ)**
- **Pathological (Bệnh lý)**

Mục tiêu cốt lõi của dự án là phát triển các mô hình Machine Learning có khả năng nhận diện chính xác các dấu hiệu bất thường, từ đó hỗ trợ ra quyết định lâm sàng, giúp bác sĩ tầm soát sớm các rủi ro y khoa thông qua phân tích dữ liệu biểu đồ CTG. Chúng tôi đặc biệt chú trọng vào việc xử lý **Class Imbalance** (mất cân bằng dữ liệu) để đảm bảo mô hình không bỏ sót các trường hợp bệnh lý nguy hiểm.

## 3. Cấu trúc thư mục
```text
fetal-health-classification/
├── data/
│   └── fetal_health.csv        # Tập dữ liệu gốc từ Kaggle
├── docs/
│   ├── report.pdf              # Báo cáo chi tiết dự án
│   └── presentation.pdf        # Slide thuyết trình
├── results/
│   ├── figures/                # Các biểu đồ và Confusion Matrix
│   └── model_weights/          # Các file mô hình đã huấn luyện (.pkl)
├── src/
│   └── source.ipynb            # Notebook chính chứa toàn bộ mã nguồn
├── .gitignore                  # Cấu hình bỏ qua file rác và file nặng
├── requirements.txt            # Danh sách các thư viện cần cài đặt
└── README.md                   # Tài liệu hướng dẫn dự án

```

## 4. Quy trình thực hiện (Methodology)

1. **Data Preprocessing:** Làm sạch dữ liệu, xử lý nhiễu và thực hiện chia tập dữ liệu (Stratified Data Splitting).
2. **Feature Engineering:** Chuẩn hóa dữ liệu bằng `StandardScaler`, giảm chiều dữ liệu bằng `PCA` (giữ lại 95% phương sai) và `LDA`.
3. **Modeling:**
* **Baseline Models:** Naive Bayes và SoftMax Regression.
* **Optimized Model (SVM):** Để giải quyết bài toán mất cân bằng dữ liệu (**Class Imbalance**), lớp `Pathological` có tỷ lệ rất thấp, chúng tôi đã tinh chỉnh tham số `class_weight`: `class_weight={1.0: 1, 2.0: 3, 3.0: 7}`.


4. **Clustering:** Sử dụng `K-Means (k=4)` để phân tích cấu trúc không gian dữ liệu.

## 5. Kết quả thực nghiệm (SVM Triage)

Việc tinh chỉnh `class_weight` đã giúp cải thiện đáng kể độ nhạy (Sensitivity) của mô hình đối với các nhóm bệnh lý.

| Metrics (Class 2 & 3) | SVM Baseline (Trước) | SVM Final (Sau) | Cải thiện |
| --- | --- | --- | --- |
| **Recall (Class 2)** | 0.59 | **0.85** | +0.26 |
| **Recall (Class 3)** | 0.77 | **0.94** | +0.17 |
| **F1-Score (Class 2)** | 0.67 | **0.72** | +0.05 |
| **F1-Score (Class 3)** | 0.84 | **0.91** | +0.07 |

**Hiệu năng trên toàn bộ các Class:**

| Class | Precision | Recall | F1-Score | Support |
| --- | --- | --- | --- | --- |
| 1.0 (`Normal`) | 0.98 | 0.91 | 0.95 | 494 |
| 2.0 (`Suspect`) | 0.63 | 0.85 | 0.72 | 88 |
| 3.0 (`Pathological`) | 0.88 | 0.94 | 0.91 | 52 |
| **Accuracy (Tổng)** |  |  | **90.69%** | **634** |

**Discussion:** Việc tăng trọng số giúp tăng mạnh **Recall** cho các ca bệnh, giảm thiểu False Negatives, dù có đánh đổi nhẹ về Precision.

## 6. Cài đặt và Thực thi

* **Bước 1: Clone dự án về máy:**

```bash
git clone https://github.com/dbaotriett/fetal-health-classification.git
cd fetal-health-classification

```

* **Bước 2: Cài đặt thư viện:**

```bash
pip install -r requirements.txt

```

* **Bước 3: Thực thi dự án:**
Mở `src/source.ipynb` và chọn **"Restart & Run All"** để tái lập toàn bộ kết quả.

> **Lưu ý:** Nếu gặp lỗi liên quan đến thư viện `yellowbrick`, hãy thực hiện lệnh: `pip install --upgrade yellowbrick`.

## 7. Tài liệu tham khảo

* Dataset: [Fetal Health Classification (Kaggle)](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification)
* Tài liệu học tập môn Học máy - Kỳ 2, 2025-2026.

