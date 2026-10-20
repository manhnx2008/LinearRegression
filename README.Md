# House Price Prediction — Linear Regression from Scratch

Dự đoán giá nhà sử dụng mô hình **Linear Regression** được xây dựng từ đầu bằng NumPy, không dùng thư viện ML nào (scikit-learn, ...).

## Cấu trúc project

```
Machine-Learning/
├── main.py                # Entry point — load data, train, predict
├── requirements.txt       # Thư viện cần cài
├── .gitignore
├── data/
│   ├── Housing.csv        # Dataset gốc (~21k mẫu nhà)
│   └── data.py            # Load, chia train/test, chuẩn hoá dữ liệu
└── src/
    ├── model.py           # Class LinearRegression (predict)
    └── train.py           # Hàm train (Gradient Descent)
```

## Dataset

Sử dụng **King County House Sales** dataset gồm ~21.613 mẫu với các đặc trưng:

| Feature | Mô tả |
|---------|--------|
| `bedrooms` | Số phòng ngủ |
| `bathrooms` | Số phòng tắm |
| `sqft_living` | Diện tích sống (sqft) |
| `sqft_lot` | Diện tích đất (sqft) |
| `floors` | Số tầng |
| `waterfront` | View mặt nước (0/1) |
| `condition` | Tình trạng nhà (1-5) |
| `grade` | Chất lượng xây dựng (1-13) |
| ... | và nhiều đặc trưng khác |

**Target:** `price` — giá nhà (USD)

## Thuật toán

- **Model:** Linear Regression: $\hat{y} = X \cdot w + b$
- **Loss:** Mean Squared Error (MSE): $L = \frac{1}{2N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2$
- **Optimizer:** Batch Gradient Descent
  - $w \leftarrow w - \alpha \cdot \frac{1}{N} X^T (X w + b - y)$
  - $b \leftarrow b - \alpha \cdot \frac{1}{N} \sum (X w + b - y)$
- **Tiền xử lý:** Z-score normalization trên features

## Cài đặt & Chạy

```bash
# 1. Clone repo
git clone <repo-url>
cd Machine-Learning

# 2. Tạo virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Cài thư viện
pip install -r requirements.txt

# 4. Chạy
python main.py
```

## Kết quả mẫu

```
Predicted: 707466.88  |  Actual: 527000.00
Predicted: 416914.16  |  Actual: 445000.00
Predicted: 322885.41  |  Actual: 435000.00
Predicted: 1178202.38 |  Actual: 970000.00
Predicted: 468404.32  |  Actual: 345000.00
```

## Hyperparameters

| Tham số | Giá trị mặc định |
|---------|-------------------|
| Learning rate | 0.01 |
| Epochs | 1000 |
| Tolerance (early stop) | 1e-5 |
| Train/Test split | 80/20 |

