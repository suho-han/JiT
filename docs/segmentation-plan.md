# Binary Segmentation Transition Plan

ImageNet 기반 생성 파이프라인을 이진 분할 작업으로 전환하기 위한 변경 계획입니다.

## Steps

1. 데이터 파이프라인을 이미지-마스크 쌍으로 통일하고 변환 동기화하기 (util/dataset.py, main_jit.py)
2. `Denoiser`/`JiT` 입력·출력 채널과 조건부 설계를 분할 목표에 맞게 재설계 (denoiser.py, model_jit.py)
3. `train_one_epoch()` 손실을 분할 손실로 교체하거나 마스크 확산 설정으로 조정 (engine_jit.py)
4. `evaluate()`를 Dice/IoU 기반으로 교체하고 생성 샘플링 로직 제거 (engine_jit.py)
5. 학습 옵션(`class_num`, CFG,  샘플링) 정리 및 문서 업데이트 (main_jit.py, README.mdte)

## Further Considerations

- 마스크를 확산으로 예측할지, 지도학습 분할로 전환할지 선택: 확산 유지 / 분할 손실 전환.
- 입력 조건: 이미지-조건(이미지→마스크) / 클래스-조건 제거 / 다중채널 결합.
