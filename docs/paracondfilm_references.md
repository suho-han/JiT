# JiT_ParaCondFiLM 설계 시 참고 논문

`JiT_ParaCondFiLM` 설계에서 확인 가능한 참고 출처를 정리합니다.

## 1) 코드에 명시된 직접 레퍼런스

`src/models/JiT_paracondfilm.py` 상단 주석의 `References`에 다음 두 구현/연구 계열이 명시되어 있습니다.

- SiT (Scalable Interpolant Transformer 계열)
  - 코드 레퍼런스: [https://github.com/willisma/SiT](https://github.com/willisma/SiT)
- Lightning-DiT (DiT 가속/경량화 계열)
  - 코드 레퍼런스: [https://github.com/hustvl/LightningDiT](https://github.com/hustvl/LightningDiT)

## 2) ParaCondFiLM 고유 설계에 대응되는 핵심 논문

`JiT_ParaCondFiLM`은 ParaCond 블록에 FiLM modulation(`film_shift`, `film_scale`)을 추가한 구조입니다.
이 아이디어의 직접적인 원전은 FiLM 논문입니다.

- **FiLM: Visual Reasoning with a General Conditioning Layer** (Perez et al., 2017)
  - arXiv: [https://arxiv.org/abs/1709.07871](https://arxiv.org/abs/1709.07871)
  - 핵심 아이디어: 조건 벡터로부터 channel-wise scale/shift를 생성해 feature를 선형 변조
  - 코드 대응: `src/models/JiT_paracondfilm.py`의 `film_modulation` 및 `modulate(cond, film_shift, film_scale)`

## 3) JiT/DiT 기반 백본 문맥에서의 참조

ParaCondFiLM은 JiT(=DiT 계열 pixel-space diffusion) 문맥 위에 추가된 변형입니다.
따라서 베이스 모델 측면에서는 아래 논문 계열을 함께 참고한 것으로 볼 수 있습니다.

- **Back to Basics: Let Denoising Generative Models Denoise** (Li & He, 2025)
  - arXiv: [https://arxiv.org/abs/2511.13720](https://arxiv.org/abs/2511.13720)
- **Scalable Diffusion Models with Transformers (DiT)** (Peebles & Xie, 2022/ICCV 2023)
  - arXiv: [https://arxiv.org/abs/2212.09748](https://arxiv.org/abs/2212.09748)

## 메모

- 저장소에서 FiLM 관련으로 가장 직접적으로 확인되는 근거는 `FiLM (2017)` 및 모델 코드의 구현 패턴입니다.
- SiT/Lightning-DiT는 코드 상단에 명시된 구현 레퍼런스입니다.
