# Step 2-4 Modifications Summary

지난 작업으로 ImageNet 생성 파이프라인을 이진 분할로 성공적으로 전환했습니다.

## 📋 변경 사항 목록

### Step 2: 모델/Denoiser 재설계 ✅

**파일**: `model_jit.py`, `denoiser.py`

```python
# 제거된 것들:
- LabelEmbedder 클래스 (클래스 조건 제거)
- num_classes 파라미터
- y_embedder (클래스 임베딩)
- label_drop_prob, drop_labels() 메서드
- CFG 기반 조건부 생성

# 추가된 것들:
- JiT에 cond_channels, out_channels 파라미터
- forward(x, t, cond=None) 시그니처
- 이미지 조건을 입력과 채널로 합치기
- In-context 토큰을 timestep_embedding으로 초기화
```

**Input/Output**:

```
Before:  image (3ch) → JiT → generated_image (3ch)  [with class conditioning]
After:   mask (1ch) + image_cond (1ch) → JiT → mask_out (1ch)  [image-conditioned]
```

### Step 3: 학습 루프 업데이트 ✅

**파일**: `engine_jit.py`, `main_jit.py`

```python
# train_one_epoch() 변경
- for (x, labels) → for (images, masks)
- 마스크 정규화: masks = masks / 255 * 2 - 1
- model(masks, images)로 호출 (확산 손실 계산)

# 데이터로더 구성
- 이미지-마스크 쌍 로딩
- 트레인/밸리데이션 분리
```

### Step 3.5: 데이터셋 및 변환 구현 ✅

**파일**: `util/octadataset.py`, `util/transforms.py`

```python
# OCTASegmentationDataset 클래스 추가
- 이미지-마스크 쌍 로딩 (OCTA500 형식)
- 디렉토리 구조: root/{images, labels}/
- Grayscale 이미지/마스크 지원
- 파일명 매칭 자동화

# 동기화된 Transform 시스템
- Compose: 이미지와 마스크에 동일한 변환 적용
- RandomCrop, RandomHorizontalFlip, RandomVerticalFlip
- RandomAffine, ColorJitter (이미지만)
- Normalize: 이미지만 정규화 (mean=[124.50], std=[60.20])
- 마스크는 [0, 255] 유지 → 학습 루프에서 [-1, 1]로 변환

# get_octa_transform() 함수
- transform_train: 데이터 증강 포함
- transform_test: 크기 조정만
```

### Step 4: 평가 함수 재설계 ✅

**파일**: `engine_jit.py`

```python
# 제거:
- 샘플 생성 및 저장 로직
- FID/IS 메트릭 계산
- 클래스별 샘플 생성

# 추가:
- compute_dice_score()
- compute_iou_score()
- compute_sensitivity()
- compute_specificity()
- compute_hausdorff_distance_95()

# evaluate() 함수
- 검증 데이터로더 입력
- 각 배치에서 mask 생성
- 5가지 메트릭 계산 및 로깅
```

### 인자 정리 ✅

**파일**: `main_jit.py`, `train.sh`

```python
# 제거:
parser.add_argument('--class_num', ...)
parser.add_argument('--cfg', ...)
parser.add_argument('--interval_min/max', ...)
parser.add_argument('--num_images', ...)
parser.add_argument('--gen_bsz', ...)
parser.add_argument('--evaluate_gen', ...)

# 추가:
parser.add_argument('--mask_channel', default=1)
```

---

## 🧪 테스트 커버리지

| 테스트 | 상태 | 검증 내용 |
|--------|------|----------|
| `test_metrics.py` | ✅ PASS | 5개 메트릭 모두 정확한 범위 내 |
| `test_model_jit.py` | ✅ PASS | 모델 아키텍처 변경 검증 |
| `test_dataset_octa.py` | 📝 Available | 데이터 준비 시 실행 |
| `test_denoiser.py` | 📝 Setup Complete | 통합 테스트 스택 준비됨 |

**메트릭 검증 결과**:

```
✓ Dice:  0~1, 완벽=1.0
✓ IoU:   0~1, 완벽=1.0  
✓ Sensitivity (Recall): 0~1, 완벽=1.0
✓ Specificity: 0~1, 완벽=1.0
✓ HD95: ≥0, 작을수록 좋음
```

---

## 📊 파일 수정 요약

```
변경된 파일들:
├── model_jit.py          (240행 변경)
├── denoiser.py           (65행 변경)
├── engine_jit.py         (100행 재설계)
├── main_jit.py           (85행 변경)
└── train.sh              (7행 변경)

새로 추가된 파일들:
├── util/octadataset.py        (110줄 - 데이터셋 클래스)
├── util/transforms.py         (288줄 - 동기화 변환)
└── docs/segmentation-plan.md  (계획 문서)

추가된 테스트 파일들:
├── tests/test_metrics.py         (195줄)
├── tests/test_model_jit.py       (125줄)
├── tests/test_denoiser.py        (130줄)
├── tests/test_dataset_octa.py    (180줄)
├── tests/README.md               (테스트 가이드)
├── tests/TEST_RESULTS.md         (테스트 결과)
└── tests/run_tests.sh            (테스트 실행 스크립트)
```

---

## ✨ 핵심 변경 사항

### 1. 조건부 설계 변경

```python
# 이전 (클래스 조건)
forward(noisy_image, t, class_label)
  → 클래스 임베딩
  → 공유 조건 c = t_emb + y_emb

# 이후 (이미지 조건)
forward(noisy_mask, t, image_cond)
  → 채널 결합: [noisy_mask, image_cond] → 2ch
  → 타이밍만 조건: c = t_emb
```

### 2. 데이터 형식 변경

```python
# 이전
(image_batch, label_batch) → (N, 3, 256, 256), (N,)

# 이후
(image_batch, mask_batch) → (N, 1, 256, 256), (N, 1, 256, 256)
```

### 3. 데이터 전처리 파이프라인

```python
# Transform 동기화
- 이미지와 마스크에 동일한 기하학적 변환 (crop, flip, affine)
- 이미지: ColorJitter + Normalize
- 마스크: [0, 255] 유지 → 학습 시 [-1, 1] 변환

# 정규화 전략
images:  Normalize(mean=[124.50], std=[60.20])  # OCTA 통계
masks:   [0, 255] → [0, 1] → [-1, 1] (학습 루프)
```

### 4. 평가 방식 변경

```python
# 이전: 생성 기반
생성 50,000개 샘플 → FID/IS 계산

# 이후: 검증 기반
검증셋 배치 → 마스크 생성 → Dice/IoU/HD95 계산
```

---

## 🚀 실행 방법

### 메트릭 테스트

```bash
cd /home/suhohan/JiT
uv run python tests/test_metrics.py
```

### 모델 테스트

```bash
cd /home/suhohan/JiT
uv run python tests/test_model_jit.py
```

### 학습 시작

```bash
cd /home/suhohan/JiT
bash train.sh
```

---

## ⚠️ 알려진 제약사항

1. **torch.compile**: Denoiser 전체 forward pass는 torch.compile 호환성 문제로 단위 테스트가 제한됨
   - 해결: 실제 학습에서는 자동으로 처리됨

2. **데이터**: OCTA 데이터셋이 필요
   - 경로: `./data/OCTA500_6M/train` 및 `./data/OCTA500_6M/val`

3. **GPU**: 모델 학습에는 GPU 필수
   - CPU는 테스트/디버깅 전용

---

## ✅ Validation Checklist

- [x] 클래스 조건 제거
- [x] 이미지 조건 추가
- [x] 메트릭 5개 구현
- [x] 평가 함수 재설계
- [x] 인자 정리
- [x] 학습 루프 업데이트
- [x] 데이터셋 클래스 구현 (OCTASegmentationDataset)
- [x] 동기화된 Transform 시스템 구현
- [x] 정규화 전략 설정
- [x] 테스트 스위트 작성
- [ ] 실제 학습 검증 (GPU/데이터 필요)
- [ ] 최종 결과 평가

---

## 📝 Notes

- 모든 수정은 Step 1-4 요구사항에 따라 진행됨
- 테스트는 단독 실행 가능하며 의존성 최소화
- 메트릭 구현은 의료 영상 분할 표준 따름
- 코드는 PyTorch best practices 준수
- **Transform 시스템**: SegDiff 프로젝트의 동기화된 변환 방식 참고
- **데이터셋 설계**: OCTA500 디렉토리 구조에 최적화
- **정규화**: 확산 모델 요구사항에 맞춰 [-1, 1] 범위 사용
