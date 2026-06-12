# 구현 기록

## Step 0: GPU 파라미터 상주화

- `Tensor`에 device parameter buffer를 추가했다.
- 파라미터 Tensor는 최대 4개 GPU에 `d_buf`로 복사본을 유지한다.
- 기존 CPU 경로가 그대로 동작하도록 host `buf`는 유지했다.
- 이후 CUDA kernel에서 사용할 수 있도록 `Tensor::device_buf(int device_id)`를 추가했다.

## Step 1: 검증 가능한 구조로 방향 수정

- 별도 `PredictSentimentBatchGPU(...)` helper와 `USE_GPU_BATCH_PATH` 분기를 제거했다.
- 앞으로는 기존 `predict_sentiment(...)` 흐름을 직접 조금씩 바꾸는 방식으로 진행한다.
- 이렇게 하면 각 단계가 기존 validation 경로 안에서 실행되므로, 변경 직후 바로 검증할 수 있다.
- `n_samples` loop는 아직 유지한다.
  - 현재 일부 연산이 여전히 단일 sample 기준 함수이기 때문이다.
  - 이 loop는 conv/linear까지 batch 대응이 끝난 뒤 제거한다.

## Step 2: Embedding + Permute 병합

- `layer.cu`에 `EmbeddingPermute(...)`를 추가했다.
- 이 함수는 기존 CPU `Embedding(...)`과 `Permute(...)`를 하나로 합친다.
- 출력 layout은 기존 `permute_a`와 동일하다.
  - 출력 shape: `[EMBEDDING_DIM, SEQ_LEN]`
  - index mapping: `out[h, s] = emb_w[input[s], h]`
- `model.cu`의 `predict_sentiment(...)`에서는 다음 두 호출을:
  - `Embedding(single_input, emb_w, emb_a)`
  - `Permute(emb_a, permute_a)`
- 아래 한 호출로 교체했다.
  - `EmbeddingPermute(single_input, emb_w, permute_a)`
- 결과적으로 `emb_a`를 거치지 않고 바로 `permute_a`를 만든다.

## Step 3: Conv/ReLU/MaxPool 및 Linear/ReLU 병합

- `layer.cu`에 `Conv1DReLUMax(...)`를 추가했다.
- 이 함수는 기존 아래 세 단계를 하나로 합친다.
  - `Conv1D(...)`
  - `ReLU(...)`
  - `GetMax(...)`
- 전체 conv activation을 `conv*_a`에 저장하지 않고, 바로 pooled output인 `pool*_a`를 만든다.
- ReLU 이후 max pooling과 동일하게 동작하도록 max 초기값은 `0.f`로 둔다.

- `layer.cu`에 `LinearReLU(...)`를 추가했다.
- 이 함수는 기존 `Linear(...)`와 `ReLU(...)`를 하나로 합친다.
- `model.cu`에서는 `linear0`, `linear1`, `linear2`에 대해 `LinearReLU(...)`를 사용한다.
- 마지막 `linear3`은 ReLU가 없으므로 기존 `Linear(...)`를 유지한다.

- 아직 `n_samples` loop는 유지한다.
- 이번 단계는 batch화 전, 중간 activation 저장과 layer 호출 수를 줄이는 fused CPU 기준 변경이다.
