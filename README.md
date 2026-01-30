# 🔊 CATCH-NOISE

**실시간 교실 소음 분류 AI 시스템**  
딥러닝 기반으로 교실 소음을 실시간 분석하여 학습 방해 소음을 감지하고,  
학생 스스로 소음 환경을 인지하고 조절할 수 있도록 돕는 **자율 학습 환경 구축 프로젝트**

<br>

## 📌 프로젝트 개요

- **기간**: 2025.09 ~ 2026.01 (5개월)
- **팀 구성**: 6인 팀
- **목표**: 실시간 오디오 분류로 학습 방해 소음 감지 및 시각적 피드백 제공

### 주요 기능
- 🎯 실시간 오디오 분류 (Non-Noisy / Noisy)
- ⚡ 10-15ms 저지연 추론으로 30fps+ 실시간 처리
- 🖥️ Gradio 기반 직관적 웹 UI
- 📱 Jetson Nano 엣지 디바이스 배포
- 🎛️ 3가지 모드 제공 (도서관, 회의, 쉬는 시간)

### 핵심 성과
| 지표 | 값 |
|------|-----|
| Test Accuracy | **92.3%** |
| 실시간 정확도 | **91.5%** |
| 추론 속도 | **10-15ms** |
| F1 Score | **0.916** |
| 모델 최적화 | **12.5배 향상** |

<br>

## 👥 팀원 소개

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/jammmin02.png" width="100px;" alt="박정민"/><br />
      <sub><b>박정민</b></sub><br />
      팀장<br />
      <a href="https://github.com/jammmin02" target="_blank">@jammmin02</a>
    </td>
    <td align="center">
      <img src="https://github.com/HyoChan1117.png" width="100px;" alt="김효찬"/><br />
      <sub><b>김효찬</b></sub><br />
      팀원 <br />
      <a href="https://github.com/HyoChan1117" target="_blank">@HyoChan1117</a>
    </td>
    <td align="center">
      <img src="https://github.com/youngmin109.png" width="100px;" alt="배영민"/><br />
      <sub><b>배영민</b></sub><br />
      팀원<br />
      <a href="https://github.com/youngmin109" target="_blank">@youngmin109</a>
    </td>
  </tr>
</table>


<br>

## 🛠️ 기술 스택

### 오디오 처리 라이브러리
- **librosa** `0.9.2` - 오디오 신호 분석 및 검증
- **torchaudio** `0.13.1` - MFCC, ZCR 특징 추출 (메인)
- **soundfile** `0.10.3` - 오디오 파일 I/O
- **sounddevice** `0.4.6` - 실시간 마이크 입력

### 딥러닝 프레임워크
- **PyTorch** `1.13.1+cu117` - CNN 분류 모델 구현 (최종 선택)
- **TensorFlow** - CNN-LSTM 하이브리드 비교 실험
- **torchvision** `0.14.1+cu117`

### 학습 가속 & 실험 관리
- **CUDA** `11.7.1` / **cuDNN** `8` - GPU 가속 (학습 시간 10배 단축)
- **MLflow** `1.30.0` - 60+ 실험 체계적 관리
- **Optuna** `3.0.3` - 자동 하이퍼파라미터 최적화

### 추론 최적화
- **ONNX** - 플랫폼 독립적 모델 변환
- **TensorRT** - FP16 최적화 (50ms → 4ms, **12.5배 향상**)

### 배포 & UI
- **Jetson Nano** - 엣지 디바이스 실시간 추론
- **Docker** - 개발 환경 통합 관리
  - Base: `pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel`
  - Alt: `nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04`
- **Gradio** - 웹 기반 실시간 UI

### 데이터 분석 & 시각화
- **NumPy** `1.22.4`
- **Pandas** `1.3.5`
- **scikit-learn** `1.0.2`
- **Matplotlib** `3.5.2`
- **Seaborn** `0.11.2`

<br>

## 📂 프로젝트 구조

```
catch-noise-dev/
├── data/
│   ├── 2class_noisy_vs_nonnoisy/  #  최종 데이터셋
│   │   ├── noisy/
│   │   └── non_noisy/
│   ├── 3_class/                    # 초기 시도
│   └── 3_class_modify/             # 개선 시도
├── src/
│   ├── models/
│   │   ├── cnn_model.py           # CNN 모델 정의
│   │   ├── train.py               # 학습 스크립트
│   │   └── optuna_optimize.py     # 자동 HPO
│   ├── preprocessing/
│   │   ├── feature_extraction.py  # MFCC, ZCR 추출
│   │   └── data_augmentation.py   # Time Shift 증강
│   └── inference/
│       ├── realtime_inference.py  # 실시간 추론
│       └── tensorrt_engine.py     # TensorRT 엔진
├── dev/
│   └── [팀원별 실험 공간]          # 브랜치 기반 개발
├── ui/
│   ├── record_ui_gradio.py        # Gradio UI
│   └── record_ui_gradio_jp.py     # Jetson용 UI
├── models/                         # 학습된 모델 저장
├── outputs/                        # 시각화, 로그, 평가 결과
├── test/                           # 샘플 테스트 오디오
├── docker/
│   └── Dockerfile
├── scripts/                        # 유틸 스크립트
├── docker-compose.yml
├── requirements.txt
└── README.md
```

<br>

## 프로젝트 목표 & 특징

| 목표 | 설명 |
|------|------|
| **소음 분류** | 교실 내 소리를 `Non-Noisy` / `Noisy`로 실시간 분류 |
|  **모드 전환** | `도서관`, `회의`, `쉬는 시간` 3가지 모드별 허용 기준 다름 |
|  **웹 시각화** | Gradio 기반 실시간 UI (색상, dB 값, 분류 결과) |
|  **피드백 학습** | 사용자 피드백 수집으로 모델 지속 개선 |
|  **자동 제어** | 시간표 기반 수업 시간 자동 측정 중단 |

### 주요 특징
-  **원거리 수음**: 무지향성 마이크 사용
-  **엣지 컴퓨팅**: Jetson Nano 기반 온디바이스 추론
- **주관적 라벨링**: 수집자 주관 + 사용자 피드백 반영
- **체계적 평가**: 혼동 행렬, F1 Score 기반 성능 측정
-  **환경 일관성**: Docker로 개발/배포 환경 통일

<br>

## 🔬 모델 아키텍처

### CNN 구조
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

### 최적 하이퍼파라미터 (Optuna 결과)
| 파라미터 | 값 |
|---------|-----|
| conv1_filters | 32 |
| conv2_filters | 64 |
| dense_units | 128 |
| dropout | 0.3 |
| learning_rate | 0.003 |
| batch_size | 32 |

<br>
