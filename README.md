# 🔊 CATCH-NOISE

**リアルタイム教室騒音分類AIシステム**  
ディープラーニングベースで教室の騒音をリアルタイム分析し、学習妨害騒音を検知することで、  
学生自ら騒音環境を認知・調整できるようサポートする**自律学習環境構築プロジェクト**

<br>

## プロジェクト概要

- **期間**: 2025.06 ~
- **チーム構成**: 3名チーム
- **目標**: リアルタイムオーディオ分類による学習妨害騒音検知およびビジュアルフィードバック提供

### 主要機能
- リアルタイムオーディオ分類 (Non-Noisy / Noisy)
- 10-15ms低遅延推論で30fps+リアルタイム処理
- Gradioベースの直感的Web UI
- Jetson Nanoエッジデバイスデプロイ
- 3つのモード提供 (図書館、会議、休憩時間)

### 主要成果
| 指標 | 値 |
|------|-----|
| Test Accuracy | **92.3%** |
| リアルタイム精度 | **91.5%** |
| 推論速度 | **10-15ms** |
| F1 Score | **0.916** |
| モデル最適化 | **12.5倍向上** |

<br>

## チームメンバー紹介

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/jammmin02.png" width="100px;" alt="パク・ジョンミン"/><br />
      <sub><b>パク・ジョンミン</b></sub><br />
      チームリーダー<br />
      <a href="https://github.com/jammmin02" target="_blank">@jammmin02</a>
    </td>
    <td align="center">
      <img src="https://github.com/HyoChan1117.png" width="100px;" alt="キム・ヒョチャン"/><br />
      <sub><b>キム・ヒョチャン</b></sub><br />
      チームメンバー<br />
      <a href="https://github.com/HyoChan1117" target="_blank">@HyoChan1117</a>
    </td>
    <td align="center">
      <img src="https://github.com/youngmin109.png" width="100px;" alt="ペ・ヨンミン"/><br />
      <sub><b>ペ・ヨンミン</b></sub><br />
      チームメンバー<br />
      <a href="https://github.com/youngmin109" target="_blank">@youngmin109</a>
    </td>
  </tr>
</table>

<br>

## 技術スタック

### オーディオ処理ライブラリ
- **librosa** `0.9.2` - オーディオ信号分析および検証
- **torchaudio** `0.13.1` - MFCC、ZCR特徴抽出（メイン）
- **soundfile** `0.10.3` - オーディオファイルI/O
- **sounddevice** `0.4.6` - リアルタイムマイク入力

### ディープラーニングフレームワーク
- **PyTorch** `1.13.1+cu117` - CNN分類モデル実装（最終選択）
- **TensorFlow** - CNN-LSTMハイブリッド比較実験
- **torchvision** `0.14.1+cu117`

### 学習高速化 & 実験管理
- **CUDA** `11.7.1` / **cuDNN** `8` - GPU高速化（学習時間10倍短縮）
- **MLflow** `1.30.0` - 60+実験の体系的管理
- **Optuna** `3.0.3` - 自動ハイパーパラメータ最適化

### 推論最適化
- **ONNX** - プラットフォーム非依存モデル変換
- **TensorRT** - FP16最適化（50ms → 4ms、**12.5倍向上**）

### デプロイ & UI
- **Jetson Nano** - エッジデバイスリアルタイム推論
- **Docker** - 開発環境統合管理
  - Base: `pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel`
  - Alt: `nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04`
- **Gradio** - Webベースリアルタイム UI

### データ分析 & ビジュアライゼーション
- **NumPy** `1.22.4`
- **Pandas** `1.3.5`
- **scikit-learn** `1.0.2`
- **Matplotlib** `3.5.2`
- **Seaborn** `0.11.2`

<br>

##  プロジェクト構造

```
catch-noise-dev/
├── data/
│   ├── 2class_noisy_vs_nonnoisy/  # 最終データセット
│   │   ├── noisy/
│   │   └── non_noisy/
│   ├── 3_class/                    # 初期試行
│   └── 3_class_modify/             # 改善試行
├── src/
│   ├── models/
│   │   ├── cnn_model.py           # CNNモデル定義
│   │   ├── train.py               # 学習スクリプト
│   │   └── optuna_optimize.py     # 自動HPO
│   ├── preprocessing/
│   │   ├── feature_extraction.py  # MFCC、ZCR抽出
│   │   └── data_augmentation.py   # Time Shift拡張
│   └── inference/
│       ├── realtime_inference.py  # リアルタイム推論
│       └── tensorrt_engine.py     # TensorRTエンジン
├── dev/
│   └── [メンバー別実験スペース]    # ブランチベース開発
├── ui/
│   ├── record_ui_gradio.py        # Gradio UI
│   └── record_ui_gradio_jp.py     # Jetson用UI
├── models/                         # 学習済みモデル保存
├── outputs/                        # ビジュアライゼーション、ログ、評価結果
├── test/                           # サンプルテストオーディオ
├── docker/
│   └── Dockerfile
├── scripts/                        # ユーティリティスクリプト
├── docker-compose.yml
├── requirements.txt
└── README.md
```

<br>

## プロジェクト目標 & 特徴

| 目標 | 説明 |
|------|------|
| **騒音分類** | 教室内の音を`Non-Noisy` / `Noisy`にリアルタイム分類 |
| **モード切替** | `図書館`、`会議`、`休憩時間` 3つのモード別許容基準異なる |
| **Web可視化** | Gradioベースリアルタイム UI（色、dB値、分類結果） |
| **フィードバック学習** | ユーザーフィードバック収集によるモデル継続改善 |
| **自動制御** | 時間割ベース授業時間自動測定中断 |

### 主要特徴
- **遠距離集音**: 無指向性マイク使用
- **エッジコンピューティング**: Jetson Nanoベースオンデバイス推論
- **主観的ラベリング**: 収集者主観 + ユーザーフィードバック反映
- **体系的評価**: 混同行列、F1 Scoreベース性能測定
- **環境一貫性**: Dockerで開発/デプロイ環境統一

<br>

## モデルアーキテクチャ

### CNN構造
```
Input (14 features: MFCC 13 + ZCR 1)
  ↓
Conv1D(32, kernel=3) → ReLU → MaxPool(2)
  ↓
Conv1D(64, kernel=3) → ReLU → MaxPool(2)
  ↓
Flatten
  ↓
Dense(128) → Dropout(0.3) → ReLU
  ↓
Dense(2) → Softmax
  ↓
Output (Non-Noisy / Noisy)
```

### 最適ハイパーパラメータ (Optuna結果)
| パラメータ | 値 |
|---------|-----|
| conv1_filters | 32 |
| conv2_filters | 64 |
| dense_units | 128 |
| dropout | 0.3 |
| learning_rate | 0.003 |
| batch_size | 32 |

<br>
