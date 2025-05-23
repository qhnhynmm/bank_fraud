# CNN Model for Banking Fraud Detection

## Tổng quan

Dự án này đã được cập nhật để hỗ trợ **Convolutional Neural Network (CNN)** model cho bài toán phát hiện gian lận ngân hàng, thay thế hoặc làm thêm lựa chọn cho MLP model ban đầu.

## Tại sao sử dụng CNN cho Fraud Detection?

### Ưu điểm của CNN so với MLP:

1. **Phát hiện Local Patterns**: CNN có thể học được các mẫu cục bộ trong dữ liệu transaction
2. **Feature Extraction tự động**: Các convolutional layers có thể tự động trích xuất các đặc trưng quan trọng
3. **Giảm overfitting**: Pooling layers và shared weights giúp giảm overfitting
4. **Hiệu quả với dữ liệu có cấu trúc**: Tốt với dữ liệu có relationship giữa các features

### Kiến trúc CNN Model:

```
Input (batch_size, features) 
    ↓
Reshape to (batch_size, 1, features)
    ↓
Conv1D + BatchNorm + ReLU + Dropout
    ↓
MaxPool1D (tùy chọn)
    ↓
Conv1D + BatchNorm + ReLU + Dropout
    ↓
MaxPool1D (tùy chọn)  
    ↓
Conv1D + BatchNorm + ReLU + Dropout
    ↓
Global Average Pooling
    ↓
Fully Connected Layers
    ↓
Output (fraud probability)
```

## Cách sử dụng

### 1. Cập nhật Configuration

Chỉnh sửa file `src/config/config.yaml`:

```yaml
model:
  type: "cnn"  # Thay đổi từ "mlp" sang "cnn"
  
  cnn:
    num_filters: [64, 128, 64]    # Số filters cho mỗi conv layer
    kernel_sizes: [3, 3, 3]       # Kích thước kernel cho mỗi conv layer  
    hidden_dims: [128, 64]        # Số neurons trong FC layers
    dropout_rate: 0.3             # Tỷ lệ dropout
```

### 2. Training CNN Model

```bash
# Train CNN model
python main.py --train

# Hoặc so sánh cả MLP và CNN
python compare_models.py
```

### 3. Inference với CNN Model

```bash
# Chạy inference với model đã train
python main.py --inference

# Test single transaction
python main.py --predict
```

## Hyperparameter Tuning

### Các tham số có thể điều chỉnh:

```yaml
cnn:
  num_filters: [32, 64, 32]       # Ít filters hơn cho dataset nhỏ
  num_filters: [128, 256, 128]    # Nhiều filters hơn cho dataset lớn
  
  kernel_sizes: [5, 3, 3]         # Kernel lớn hơn để capture wider patterns
  kernel_sizes: [3, 3, 3]         # Kernel nhỏ hơn cho local patterns
  
  hidden_dims: [256, 128, 64]     # Thêm layers cho model phức tạp hơn
  hidden_dims: [64, 32]           # Ít layers hơn cho dataset nhỏ
  
  dropout_rate: 0.2               # Giảm dropout nếu underfitting
  dropout_rate: 0.5               # Tăng dropout nếu overfitting
```

## So sánh Performance

| Model | Ưu điểm | Nhược điểm |
|-------|---------|------------|
| **MLP** | Đơn giản, nhanh train | Không học được local patterns |
| **CNN** | Học local patterns, robust | Phức tạp hơn, train lâu hơn |

## Kiểm tra Model hoạt động

```bash
# Test CNN model implementation
python test_cnn_model.py
```

## Model Files

Sau khi training, models sẽ được lưu tại:

```
checkpoints/
├── best_model_mlp.pt       # MLP model
├── best_model_cnn.pt       # CNN model  
└── best_model_*.joblib     # Traditional ML models
```

## Troubleshooting

### Lỗi thường gặp:

1. **CUDA out of memory**: Giảm `batch_size` trong config
2. **Convergence chậm**: Tăng learning rate hoặc giảm model complexity
3. **Overfitting**: Tăng `dropout_rate` hoặc thêm regularization

### Performance không tốt:

1. Thử các hyperparameters khác nhau
2. Kiểm tra data preprocessing
3. So sánh với MLP model để xem CNN có phù hợp không

## Code Structure

```
src/
├── model/
│   ├── mlp_model.py       # MLP implementation
│   ├── cnn_model.py       # CNN implementation (NEW)
│   └── ml_models.py       # Traditional ML models
├── task/
│   ├── train.py           # Training script (updated for CNN)
│   └── inference.py       # Inference script (updated for CNN)
└── config/
    └── config.yaml        # Configuration (updated for CNN)
```

## Kết luận

CNN model cung cấp một alternative approach cho fraud detection với khả năng học local patterns tốt hơn MLP. Thử nghiệm cả hai models và chọn model phù hợp nhất dựa trên performance trên validation set của bạn.

---

**Lưu ý**: Đảm bảo đã cài đặt đầy đủ dependencies trong `requirements.txt` trước khi sử dụng CNN model. 