# Cascade vs Direct
### Korean Spoken Intent Recognition

Speech Recognition Challenge 2026 (Ajou University) — Team 14 · 김민승 (202020760)

음성 명령의 **의도(intent)**를 알아내기 위해 꼭 음성을 텍스트로 받아 적어야 할까?
이 챌린지는 한국어 스마트 스피커 명령 60종에 대해

- **Cascade** — Whisper ASR → 텍스트 의도 분류기
- **Direct** — 전사 없이 오디오 인코더 → 의도 분류

두 방식을 **같은 평가셋·같은 GPU·같은 지표**(정확도, Macro-F1, ECE, 지연시간)로 비교합니다.

## Repository layout

```
common.py                      공통 경로, 정규화, ECE
data/
  prepare_data.py              데이터 분할 생성 (speaker-disjoint)
  labels.json                  60개 intent 라벨
  eval/eval.jsonl, eval/audio/ ★ 평가용 데이터셋 (레포에 포함)
baseline/
  train_text_clf.py            Cascade 2단계 텍스트 분류기 (KLUE-RoBERTa-base)
  cascade.py                   Cascade 추론 (Whisper → 분류기), --oracle 지원
  direct_model.py              Direct 모델 (frozen Whisper encoder + attentive pooling)
  train_direct.py              Direct head 학습
  direct_infer.py              Direct 추론
evaluation/
  evaluate.py                  지표 계산 (Acc, Macro-F1, ECE, Latency, CER)
  make_leaderboard.py          리더보드 생성 + 제출 HTML 자동 갱신
notebooks/run_all_colab.ipynb  Colab(T4) 원클릭 재현
run_all.sh                     전체 파이프라인
```

## Dataset

| Split | Source | Size | Use |
|---|---|---|---|
| Text train | MASSIVE 1.1 ko-KR `train` | 11,514 utt. (text) | Cascade 분류기 학습 |
| Text dev | MASSIVE 1.1 ko-KR `dev` − eval IDs | ≈1.4k utt. (text) | 분류기 모델 선택 |
| Audio train | Speech-MASSIVE ko-KR `validation`, ~70% of speakers | ≈1.4k utt. | Direct head 학습 |
| **Eval** | Speech-MASSIVE ko-KR `validation`, ~30% of speakers | ≈0.6k utt. | **모든 시스템 평가** |

- 평가셋은 학습 오디오와 **화자가 겹치지 않도록** 분할 (seed 42)
- 평가 발화의 정답 텍스트는 어떤 학습·모델 선택 단계에도 쓰이지 않음
- 정확한 발화 수/화자 수는 `python data/prepare_data.py` 실행 로그에 출력됨
- License: Speech-MASSIVE / MASSIVE — CC BY-NC-SA 4.0 (비상업적 교육 목적 사용)

## Systems (baselines)

| ID | Track | Pipeline |
|---|---|---|
| `oracle_text` | Reference | 정답 텍스트 → KLUE-RoBERTa (ASR 오류가 없을 때의 상한) |
| `cascade_whisper-small` | Cascade | Whisper-small (ko, greedy) → KLUE-RoBERTa |
| `cascade_whisper-large-v3-turbo` | Cascade | Whisper-large-v3-turbo → KLUE-RoBERTa |
| `direct_whisper-small` | Direct | Whisper-small encoder (frozen) + attentive pooling head |

## Metrics

| Metric | 설명 |
|---|---|
| **Accuracy ↑** (primary) | top-1 intent 정확도 |
| Macro-F1 ↑ | 평가셋에 등장한 intent별 F1 평균 |
| ECE ↓ | 15-bin expected calibration error (top-1 confidence) |
| Latency ↓ | 발화당 end-to-end 지연 median/p90 (batch 1, fp16, 동일 GPU) |
| CER ↓ | Cascade의 ASR 문자 오류율 (공백·문장부호 제거) |

순위는 Accuracy, 동점이면 median latency 기준. Oracle은 순위에서 제외.

## Reproduce

Colab: `notebooks/run_all_colab.ipynb` 를 열고 T4 GPU로 순서대로 실행.

로컬 GPU:

```bash
pip install -r requirements.txt
bash run_all.sh                       # data → train → infer → evaluate → leaderboard
# 제출 페이지의 리더보드 자동 갱신
python evaluation/make_leaderboard.py --html path/to/Team_KimMinseung_202020760.html
```

## References

1. B. Lee et al., *Speech-MASSIVE: A Multilingual Speech Dataset for SLU and Beyond*, Interspeech 2024.
2. J. FitzGerald et al., *MASSIVE: A 1M-Example Multilingual NLU Dataset with 51 Typologically-Diverse Languages*, ACL 2023.
3. A. Radford et al., *Robust Speech Recognition via Large-Scale Weak Supervision* (Whisper), ICML 2023.
4. S. Park et al., *KLUE: Korean Language Understanding Evaluation*, NeurIPS Datasets & Benchmarks 2021.
5. C. Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017.
