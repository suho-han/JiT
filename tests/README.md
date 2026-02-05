# JiT Segmentation Tests

테스트 스위트는 이진 세그멘테이션 전환 과정에서 수정된 각 모듈을 개별적으로 검증합니다.

## 테스트 파일 구성

### 1. `test_model_jit.py` - 모델 아키텍처 테스트

**수정 내용**: Step 2 - JiT 모델 재설계

테스트 항목:

- ✓ JiT 모델 초기화 (in_channels, cond_channels, out_channels)
- ✓ Forward pass with image conditioning
- ✓ Forward pass without conditioning (cond=None)
- ✓ JiT-B/16 팩토리 함수
- ✓ 출력 형태 검증

실행:

```bash
python tests/test_model_jit.py
```

예상 결과: 마스크 입력과 이미지 조건을 받아 마스크 출력을 생성.

---

### 2. `test_denoiser.py` - Denoiser 테스트

**수정 내용**: Step 2 - Denoiser 입출력 재설계

테스트 항목:

- ✓ Denoiser 초기화 (image conditioning 지원)
- ✓ Forward pass (학습 손실 계산)
- ✓ Generate method (샘플링)
- ✓ EMA 파라미터 업데이트

실행:

```bash
python tests/test_denoiser.py
```

예상 결과: 마스크 타겟과 이미지 조건에서 확산 손실을 계산하고, 이미지 조건에서 마스크를 생성.

---

### 3. `test_metrics.py` - 평가 메트릭 테스트

**수정 내용**: Step 4 - Dice/IoU/Sensitivity/Specificity/HD95 메트릭 추가

테스트 항목:

- ✓ Dice 계수 (완벽, 부분 겹침, 반대)
- ✓ IoU (완벽, 겹침 없음, 부분)
- ✓ Sensitivity (완벽, FN만, 50% 감지)
- ✓ Specificity (완벽, FP만, 균형)
- ✓ Hausdorff Distance 95 (완벽, 이동, 완전 다름)
- ✓ 배치 일관성 검증

실행:

```bash
python tests/test_metrics.py
```

예상 결과:

- Dice/IoU: 0~1 범위, 1=완벽한 예측
- Sensitivity/Specificity: 0~1 범위, 1=완벽
- HD95: ≥0, 작을수록 경계가 가까움

---

### 4. `test_dataset_octa.py` - 데이터셋 로딩 테스트

**수정 내용**: Step 1 - 이미지-마스크 쌍 데이터 파이프라인

테스트 항목:

- ✓ OCTA 데이터셋 기본 로딩
- ✓ 단일 샘플 로딩 (이미지-마스크 쌍)
- ✓ Transform 동기화 (Flip, Crop 등)
- ✓ get_octa_transform 함수
- ✓ DataLoader 배치 로딩

실행:

```bash
python tests/test_dataset_octa.py
```

예상 결과:

- 이미지와 마스크 모양 일치: (C, H, W)
- Transform이 양쪽에 동일하게 적용
- DataLoader에서 배치 형태: (B, C, H, W)

---

### 5. `test_dataset.py` - 원본 데이터셋 테스트

기존 OCTA 데이터셋 로딩 테스트 (유지됨).

---

## 빠른 실행

모든 테스트 한 번에 실행:

```bash
bash tests/run_tests.sh
```

또는:

```bash
cd tests && bash run_tests.sh && cd ..
```

---

## 테스트 검증 체크리스트

- [ ] `test_model_jit.py` 통과 → 모델이 image conditioning을 지원함
- [ ] `test_denoiser.py` 통과 → Denoiser forward/generate 정상 작동
- [ ] `test_metrics.py` 통과 → 모든 메트릭이 올바른 범위 내
- [ ] `test_dataset_octa.py` 통과 → 이미지-마스크 쌍이 동기화됨

---

## 필요한 의존성

```bash
pip install torch scipy numpy pillow tensorboard
```

---

## 트러블슈팅

### 테스트 실패 시

1. **"OCTA 데이터셋을 찾을 수 없음"**
   - 데이터 경로 확인: `./data/OCTA500_6M/train` 또는 `./data/OCTA500_3M/train`
   - 스킵 가능한 테스트 (데이터 의존 테스트만 건너뜀)

2. **메트릭 테스트 실패**
   - scipy 설치 확인: `pip install scipy`
   - torch.cuda 사용 시 CPU에서 재실행 확인

3. **모델/Denoiser 초기화 실패**
   - model_jit.py와 denoiser.py가 최신 버전인지 확인
   - Step 2 수정 사항이 적용되었는지 확인

---

## 개별 테스트 예시

```bash
# 메트릭만 테스트
python tests/test_metrics.py

# 모델만 테스트
python tests/test_model_jit.py

# Denoiser 테스트 + 생성
python tests/test_denoiser.py
```

---

## 추가 학습

각 테스트 파일의 구조:

```
test_<module>.py
├── test_<feature_1>()    # 개별 기능 테스트
├── test_<feature_2>()
└── if __name__ == '__main__':  # 실행 시 모든 테스트 순서대로 실행
    └── print 요약
```

수정된 코드와 테스트의 매핑:

- Step 1 → `test_dataset_octa.py`
- Step 2 → `test_model_jit.py`, `test_denoiser.py`
- Step 3 → `engine_jit.py` 훈련 루프 (자동 검증)
- Step 4 → `test_metrics.py`
- Step 5 → `train.sh`, `main_jit.py` (인자 검증)
