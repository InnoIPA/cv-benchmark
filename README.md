# CV-Benchmark 🔍

[![Version](https://img.shields.io/badge/version-v1.0-blue.svg)](https://github.com/your-repo/cv-benchmark)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

一個強大的計算機視覺模型性能基準測試工具，專為 YOLO 模型優化，支援固定Channel數量的多模型並行測試和詳細的性能分析。

## 📋 版本信息

- **當前版本**: v1.0
- **發布日期**: 2025-09-12
- **Python 支援**: 3.8+
- **主要功能**: 固定Channel數量的多模型並行測試、智能推薦、詳細分析

## ✨ 功能特色

- 🚀 **現代化推論**: 基於 PyTorch + Ultralytics YOLO 推論引擎
- 📊 **詳細分析**: 提供 FPS、延遲、吞吐量等關鍵性能指標
- 💻 **資源監控**: 實時監控 CPU、記憶體、GPU 使用情況
- 🎯 **檢測統計**: 追蹤每幀檢測數量和置信度，分析模型準確性
- 🧠 **智能推薦**: 基於硬體規格的自動參數優化建議
- 📄 **報告生成**: 自動生成 JSON 格式的詳細測試報告
- 🔧 **靈活配置**: 支援自定義輸入尺寸、置信度閾值等參數
- 🎯 **固定Channel策略**: 用戶設定的Channel數不會改變，用可載入的模型數量來處理所有Channel

## 🛠️ 安裝與設置

### 環境要求

- Python 3.8+
- PyTorch 2.0+
- OpenCV 4.8+
- NumPy 1.24+
- Ultralytics 8.0+

### 快速安裝（本機環境）

```bash
# 1. 克隆專案
git clone <repository-url>
cd cv-benchmark

# 2. 安裝 venv（通用）
sudo apt update
sudo apt install -y python3-venv

# 3. 創建虛擬環境
python3 -m venv venv
source venv/bin/activate

# 4. 安裝依賴套件
pip install -r requirements.txt
```

### 使用 Docker（推薦跨機器一致性）

```bash
# 1) 建置映像
docker build -t cv-benchmark:latest .

# 2) 以 GPU 執行（需要 NVIDIA 驅動與 nvidia-container-toolkit）
docker run --rm --gpus all \
  -v "$PWD/reports:/app/reports" \
  -v "$PWD/videos:/app/videos" \
  cv-benchmark:latest \
  --video videos/car.mp4 --model yolov8n.pt -n 4 -t 30

# 3) 其他範例
docker run --rm --gpus all -v "$PWD/reports:/app/reports" -v "$PWD/videos:/app/videos" \
  cv-benchmark:latest --video videos/car.mp4 --model yolov8m.pt -n 8 --auto-optimize
```

## 🚀 使用方法

### 基本使用

```bash
# 基本測試
python run-cv-benchmark.py --video videos/car.mp4 --model yolov8m.pt

# 指定Channel數量和測試時間
python run-cv-benchmark.py --video videos/car.mp4 --model yolov8m.pt -n 8 -t 30

# 固定模型數量
python run-cv-benchmark.py --video videos/car.mp4 --model yolov8m.pt -n 8 -m 4 -t 30
```

### 自動優化模式

```bash
# 自動測試不同模型數量，找到最佳平衡點
python run-cv-benchmark.py --video videos/car.mp4 --model yolov8m.pt -n 8 -t 30 --auto-optimize
```

### 參數說明

| 參數 | 簡寫 | 說明 | 預設值 |
|------|------|------|--------|
| `--video` | - | 輸入視頻路徑 | 必填 |
| `--model` | - | YOLO 模型名稱或路徑 | `yolov8n.pt` |
| `--channels` | `-n` | 固定的並行Channel數（不會改變） | 4 |
| `--models` | `-m` | 固定載入的模型數量（覆蓋自動計算） | - |
| `--auto-optimize` | - | 自動測試不同模型數量，找到最佳平衡點 | - |
| `--seconds` | `-t` | 測試持續時間（秒） | 60 |
| `--img-size` | - | 模型輸入尺寸 | 640 |
| `--conf` | - | 置信度閾值 | 0.25 |
| `--iou` | - | IoU 閾值 | 0.5 |
| `--device` | - | 設備配置 (auto, cpu, cuda) | `auto` |
| `--output` | - | 輸出報告文件 | - |

## 🎯 固定Channel策略說明

### 核心概念

這個工具採用**固定Channel數量**的策略：

- ✅ **Channel數量固定**: 用戶設定的Channel數不會改變
- ✅ **智能模型分配**: 用可載入的模型數量來處理所有Channel
- ✅ **模型共享**: 多個Channel可以共享同一個模型實例
- ✅ **詳細分配報告**: 顯示Channel與模型的對應關係
- ✅ **性能分析**: 分析模型共享對性能的影響

### 使用場景

**場景1: 理想配置**
```bash
# 請求8個Channel，硬體能載入8個模型
python run-cv-benchmark.py --video videos/car.mp4 --model yolov8n.pt -n 8
# 結果: 每個Channel都有專屬模型，性能最佳
```

**場景2: 模型共享**
```bash
# 請求16個Channel，硬體只能載入8個模型
python run-cv-benchmark.py --video videos/car.mp4 --model yolov8m.pt -n 16
# 結果: 8個模型處理16個Channel，每個模型處理2個Channel
```

**場景3: 自動優化**
```bash
# 自動測試不同模型數量，找到最佳平衡點
python run-cv-benchmark.py --video videos/car.mp4 --model yolov8m.pt -n 8 --auto-optimize
# 結果: 測試1, 2, 4, 8個模型，找到最佳配置
```

## 📁 專案結構

```
cv-benchmark/
├── run-cv-benchmark.py           # CV基準測試工具
├── requirements.txt              # 依賴包列表
├── videos/                       # 測試視頻目錄
│   ├── car.mp4                  # 車輛檢測測試視頻
│   └── 4-corner-downtown.mp4    # 街景測試視頻
├── reports/                      # 測試報告目錄
│   ├── cv_benchmark_yolov8n_4ch_20241201_143022.json
│   └── cv_optimization_yolov8m_8ch_20241201_150315.json
├── venv/                         # 虛擬環境
├── yolov8*.pt                    # YOLO 模型文件
└── README.md                     # 本文件
```

## 📊 輸出報告

### 控制台輸出

測試過程中會實時顯示：
- 每Channel的 FPS 和延遲
- 系統資源使用情況 (CPU, 記憶體, GPU)
- 檢測統計信息和置信度
- Channel與模型的分配關係

### JSON 報告

工具會自動生成帶時間戳記的 JSON 報告，存放在 `reports/` 目錄中：

#### 報告命名規則
- **基準測試報告**: `cv_benchmark_{模型名}_{Channel數}ch_{時間戳記}.json`
- **優化測試報告**: `cv_optimization_{模型名}_{Channel數}ch_{時間戳記}.json`

#### 範例檔案名
```
cv_benchmark_yolov8n_4ch_20241201_143022.json
cv_optimization_yolov8m_8ch_20241201_150315.json
```

#### 報告內容
詳細的 JSON 報告包含：

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "sdk_info": {
    "name": "Fixed Channel Multi-Model Parallel Benchmark",
    "version": "1.0.0",
    "framework": "PyTorch + Ultralytics YOLO"
  },
  "configuration": {
    "model": "yolov8m.pt",
    "requested_channels": 8,
    "actual_models": 4,
    "channels_per_model": 2.0,
    "architecture": "fixed_channel_multi_model_parallel"
  },
  "performance_metrics": {
    "fps": { 
      "average": 45.2, 
      "min": 42.1, 
      "max": 48.3,
      "total": 361.6
    },
    "latency_ms": { 
      "average": 22.1, 
      "min": 20.7, 
      "max": 23.8
    }
  },
  "hardware_analysis": {
    "channel_allocation": {
      "requested_channels": 8,
      "actual_models": 4,
      "channels_per_model": 2.0,
      "allocation_efficiency": 50.0,
      "is_ideal_config": false
    }
  },
  "optimization_recommendations": [
    "⚠️ 模型共享：每個模型處理 2.0 個Channel",
    "⚠️ 硬體限制：只能載入 4/8 個模型",
    "💡 建議：使用更小的模型 (yolov8n) 以載入更多實例"
  ]
}
```

## 🔧 高級配置

### 支援的模型

支援 YOLO 系列 PyTorch 模型：

```bash
# 預訓練模型 (自動下載)
python run-cv-benchmark.py --video test.mp4 --model yolov8n.pt
python run-cv-benchmark.py --video test.mp4 --model yolov8s.pt
python run-cv-benchmark.py --video test.mp4 --model yolov8m.pt

# 自定義模型
python run-cv-benchmark.py --video test.mp4 --model /path/to/custom_model.pt
```

### 性能優化

1. **GPU 加速**: 確保安裝了 CUDA 版本的 PyTorch
2. **Channel數量**: 根據硬體規格調整Channel數
3. **輸入尺寸**: 較小的輸入尺寸可以提升速度但可能降低準確性
4. **自動優化**: 使用 `--auto-optimize` 獲取最佳配置建議

### 批量測試

```bash
# 測試多個模型
for model in yolov8n.pt yolov8s.pt yolov8m.pt; do
    python run-cv-benchmark.py \
        --video videos/test.mp4 \
        --model $model \
        --output "reports/$(basename "$model" .pt)_report.json" \
        -n 8 -t 30
done
```

## 🐛 故障排除

### 常見問題

1. **CUDA 不可用**
   ```
   CUDA 可用: False
   ```
   - 解決方案：安裝 CUDA 版本的 PyTorch
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. **視頻無法打開**
   ```
   [Channel 0] 無法打開視頻: videos/your_video.mp4
   ```
   - 解決方案：檢查視頻文件路徑和格式是否正確

3. **模型載入失敗**
   ```
   FileNotFoundError: model not found: yolov8n.pt
   ```
   - 解決方案：模型會自動下載，確保網路連接正常

4. **記憶體不足**
   ```
   RuntimeError: CUDA out of memory
   ```
   - 解決方案：使用更小的模型或減少Channel數量

### 性能調優

- **CPU 模式**: 適合 CPU 推理，調整Channel數以匹配 CPU 核心數
- **GPU 模式**: 需要 NVIDIA GPU 和對應的 CUDA 環境
- **記憶體優化**: 減少輸入尺寸或Channel數以降低記憶體使用
- **自動優化**: 使用 `--auto-optimize` 自動調整參數

## 📈 性能基準

### 測試環境
- CPU: Intel/AMD 多核心處理器
- GPU: NVIDIA GPU (可選)
- 模型: YOLOv8 PyTorch
- 輸入尺寸: 640x640
- 框架: PyTorch + Ultralytics YOLO

### 典型性能
- **單Channel CPU**: ~30-45 FPS, 20-35ms 延遲
- **單Channel GPU**: ~60-120 FPS, 8-17ms 延遲
- **多Channel並行**: 根據硬體配置調整
- **模型共享**: 性能會根據共享比例調整

## 📝 更新日誌

### v1.0 (2024-01-01)
- 🎉 **初始發布**
- ✨ 固定Channel數量的多模型並行測試
- ✨ 智能模型分配策略
- ✨ 詳細的性能分析報告
- ✨ 自動優化功能
- ✨ 支援 YOLO PyTorch 模型
- 📊 實時資源監控
- 🎯 檢測統計和分析

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

1. Fork 專案
2. 創建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟 Pull Request

## 📄 授權

本專案採用 MIT 授權 - 查看 [LICENSE](LICENSE) 文件了解詳情。

## 🙏 致謝

- [PyTorch](https://pytorch.org/) - 深度學習框架
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - 目標檢測模型
- [OpenCV](https://opencv.org/) - 計算機視覺庫

---

**⭐ 如果這個專案對您有幫助，請給個 Star！**