# JiT_ParaCondWaveFix 설계 시 참고 논문

`JiT_ParaCondWaveFix` 설계에서 확인 가능한 참고 출처를 정리합니다.

## 1) 코드에 명시된 직접 레퍼런스

`src/models/JiT_paracondwavefix.py` 상단 주석의 `References`에 다음 두 구현/연구 계열이 명시되어 있습니다.

- SiT (Scalable Interpolant Transformer 계열)
  - 코드 레퍼런스: [https://github.com/willisma/SiT](https://github.com/willisma/SiT)
- Lightning-DiT (DiT 가속/경량화 계열)
  - 코드 레퍼런스: [https://github.com/hustvl/LightningDiT](https://github.com/hustvl/LightningDiT)

## 2) WaveFix 고유 설계(웨이블릿 분해 + 스트림 가중 안정화)에 대응되는 논문

`JiT_ParaCondWaveFix`는 `JiT_ParaCondWave`의 wavelet conditioning을 계승하고,
세 조건 스트림(`cond`, `low_cond`, `high_cond`) 가중치를 softmax로 정규화해 scale collapse를 줄이는 구조입니다.

### 2-1. Wavelet decomposition 관련

코드의 `HaarSplitter`는 Haar wavelet 분해/복원을 사용합니다.
이론적 기반으로는 고전적인 다해상도 wavelet 분석 문헌이 대응됩니다.

- **A Theory for Multiresolution Signal Decomposition: The Wavelet Representation** (Mallat, 1989)
  - IEEE TPAMI
  - 핵심 아이디어: 저주파/고주파 서브밴드 분해(멀티스케일 분석)
  - 코드 대응: `src/models/JiT_paracondwave.py`의 `HaarSplitter`

### 2-2. Diffusion + frequency/wavelet conditioning 최근 흐름

저장소 내 서베이(`papers/**conditioning_survey**.md`)에서 Wave 계열 설계를 뒷받침하는 레퍼런스로 아래가 정리되어 있습니다.

- **DiMSUM: Diffusion Mamba — Scalable Unified Spatial-Frequency Method** (2024)
  - arXiv: [https://arxiv.org/abs/2411.04168](https://arxiv.org/abs/2411.04168)
- **HDW-SR: High-Frequency Guided Diffusion via Wavelet Decomposition** (2025)
  - arXiv: [https://arxiv.org/abs/2511.13175](https://arxiv.org/abs/2511.13175)

### 2-3. `softmax stream weighting` 직접 출처 확인 결과

`src/models/JiT_paracondwavefix.py`의 구현은 세 스트림(`cond`, `low_cond`, `high_cond`)의
가중치를 `torch.softmax(...)`로 정규화해 합산합니다.

- 코드 위치: `src/models/JiT_paracondwavefix.py`의 `weights = torch.softmax(torch.stack([w_c, w_lc, w_hc], dim=0), dim=0)`
- 확인 결과: 저장소 내 코드/문서 기준으로 이 연산을 특정 단일 논문에서 직접 가져왔다는 명시적 인용은 확인되지 않았음

가장 근접한 아이디어 계열 후보는 다음과 같습니다.

- **Structure-Accurate Medical Image Translation via Dynamic Frequency Balance and Knowledge Guidance** (2025)
  - arXiv: [https://arxiv.org/abs/2504.09441](https://arxiv.org/abs/2504.09441)
  - 관련성: 주파수 분해 후 attention 내 softmax 기반 가중 처리 및 주파수 균형을 다룸
  - 차이점: WaveFix처럼 3개 조건 스트림 가중치를 직접 softmax 정규화해 합산하는 동일 식을 제시하지는 않음

## 3) JiT/DiT 기반 백본 문맥에서의 참조

WaveFix 역시 JiT(=DiT 계열 pixel-space diffusion) 기반 변형이므로,
백본 관점에서는 아래 논문 계열을 함께 참고한 것으로 볼 수 있습니다.

- **Back to Basics: Let Denoising Generative Models Denoise** (Li & He, 2025)
  - arXiv: [https://arxiv.org/abs/2511.13720](https://arxiv.org/abs/2511.13720)
- **Scalable Diffusion Models with Transformers (DiT)** (Peebles & Xie, 2022/ICCV 2023)
  - arXiv: [https://arxiv.org/abs/2212.09748](https://arxiv.org/abs/2212.09748)

## 메모

- WaveFix의 `softmax-normalized stream weights` 자체는 코드 설명에서 안정화 목적이 명시되어 있으며,
  저장소 내에서 특정 단일 논문으로 직접 고정해 인용한 흔적은 확인되지 않았습니다.
- 따라서 이 문서는 `코드에 명시된 직접 출처`와 `구조적으로 대응되는 핵심/관련 논문`을 분리해 기록했습니다.
