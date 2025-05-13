# Banking Fraud Detection System

Hệ thống phát hiện gian lận trong giao dịch ngân hàng sử dụng phương pháp học máy và học sâu.

## Tính năng

- **Tiền xử lý dữ liệu**: Xử lý dữ liệu đầu vào, chuẩn hóa và cân bằng lớp
- **Đa mô hình**: Hỗ trợ các mô hình MLP, Random Forest, XGBoost và LightGBM
- **Chỉ số đánh giá**: Đánh giá mô hình dựa trên accuracy, precision, recall và F1-score
- **Giao diện Streamlit**: Giao diện người dùng đồ họa để dự đoán gian lận

## Cài đặt

1. Sao chép repository:
```bash
git clone <repo-url>
cd banking_fraud
```

2. Cài đặt các thư viện bắt buộc:
```bash
pip install -r requirements.txt
```

3. Chuẩn bị dữ liệu:
- Đặt file dữ liệu huấn luyện vào thư mục `data/` với tên `CreditCardData.csv` (hoặc cập nhật đường dẫn trong `src/config/config.yaml`)
- Đặt file dữ liệu test vào thư mục `data/` với tên `test_data.csv` (hoặc cập nhật `test_file` trong phần `inference` của cấu hình)

## Cấu trúc dự án

```
banking_fraud/
├── data/                  # Dữ liệu gốc và đã xử lý
│   ├── CreditCardData.csv # Dữ liệu huấn luyện
│   └── test_data.csv      # Dữ liệu test (tùy chọn)
├── src/                   # Mã nguồn chính
│   ├── config/            # Cấu hình
│   ├── data_utils/        # Tiện ích xử lý dữ liệu
│   ├── metrics/           # Đo lường hiệu suất
│   ├── model/             # Định nghĩa mô hình
│   ├── pipelines/         # Pipeline xử lý dữ liệu
│   └── task/              # Tác vụ huấn luyện và dự đoán
├── checkpoints/           # Mô hình đã lưu
├── logs/                  # File log
├── predictions/           # Kết quả dự đoán
├── app.py                 # Ứng dụng Streamlit
├── main.py                # Script chính
├── test_pipeline.py       # Script kiểm tra
└── requirements.txt       # Thư viện bắt buộc
```

## Sử dụng

### Huấn luyện mô hình

Để huấn luyện mô hình:

```bash
# Huấn luyện với mô hình mặc định (MLP)
python main.py --mode train

# Huấn luyện với mô hình cụ thể
python main.py --mode train --model random_forest
python main.py --mode train --model xgboost
python main.py --mode train --model lightgbm
```

### Chạy dự đoán

Có 3 cách để thực hiện dự đoán:

1. Sử dụng file test được chỉ định trong cấu hình:
```bash
# Sử dụng file test trong cấu hình và mô hình mặc định
python main.py --mode inference

# Sử dụng file test trong cấu hình và mô hình cụ thể
python main.py --mode inference --model xgboost
```

2. Cung cấp đường dẫn file test qua tham số command line (sẽ ghi đè cấu hình):
```bash
# Chỉ định file test và mô hình
python main.py --mode inference --model random_forest --test-data path/to/your/test_data.csv
```

3. Không cung cấp file test để chạy dự đoán trên một giao dịch mẫu:
```bash
# Không chỉ định file test - sẽ sử dụng giao dịch mẫu
python main.py --mode inference --model lightgbm
```

Kết quả dự đoán sẽ được lưu vào thư mục `predictions/` và hiển thị tổng quan trên console.

### Ứng dụng Streamlit

Khởi chạy giao diện người dùng:
```bash
streamlit run app.py
```

Giao diện web cho phép bạn:
1. Chọn mô hình để sử dụng (với trạng thái xác nhận đã huấn luyện)
2. Tải lên dữ liệu để phân tích
3. Điều chỉnh ngưỡng phát hiện
4. Kiểm tra giao dịch đơn lẻ
5. Xem tổng quan và giải thích kết quả phát hiện

### Kiểm tra hệ thống

Để kiểm tra toàn bộ quy trình từ tải dữ liệu đến dự đoán:
```bash
python test_pipeline.py
```

## Cấu hình

Bạn có thể điều chỉnh cấu hình trong `src/config/config.yaml`:

- **Dữ liệu**: 
  - `data_path`: Đường dẫn dữ liệu huấn luyện
  - `processed_dir`: Thư mục lưu dữ liệu đã xử lý
  - `target_column`: Tên cột nhãn

- **Tiền xử lý**: 
  - Các tỷ lệ chia tập dữ liệu
  - Tham số SMOTE và undersampling
  - Các chức năng tạo đặc trưng

- **Mô hình**: 
  - Loại mô hình (mlp, random_forest, xgboost, lightgbm)
  - Tham số cho từng loại mô hình

- **Dự đoán**:
  - `test_file`: Đường dẫn file dữ liệu test cho dự đoán
  - `threshold`: Ngưỡng phát hiện gian lận
  - `output_dir`: Thư mục lưu kết quả dự đoán

## Metrics

Hệ thống sử dụng các chỉ số sau để đánh giá hiệu suất phát hiện gian lận:

- **Accuracy**: Tỷ lệ dự đoán chính xác
- **Precision**: Tỷ lệ TP / (TP + FP)
- **Recall**: Tỷ lệ TP / (TP + FN)
- **F1 Score**: 2 * (Precision * Recall) / (Precision + Recall)

## Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng gửi Pull Request hoặc mở Issues để cải thiện hệ thống. 