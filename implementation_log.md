# 구현 기록

## Step 0: GPU 파라미터 상주화

- `Tensor`에 device parameter buffer를 추가했다.
- 파라미터 Tensor는 최대 4개 GPU에 `d_buf`로 복사본을 유지한다.
- 기존 CPU 경로가 그대로 동작하도록 host `buf`는 유지했다.
- 이후 CUDA kernel에서 사용할 수 있도록 `Tensor::device_buf(int device_id)`를 추가했다.

## Step 1: 검증 가능한 구조로 방향 수정

- 별도 `PredictSentimentBatchGPU(...)` helper와 `USE_GPU_BATCH_PATH` 분기를 제거했다.
- 기존 `predict_sentiment(...)` 흐름을 직접 조금씩 바꾸는 방식으로 진행하기로 했다.
- 각 단계가 validation 경로 안에서 실행되므로 변경 직후 검증할 수 있다.

## Step 2: Embedding + Permute 병합

- `layer.cu`에 `EmbeddingPermute(...)`를 추가했다.
- 기존 CPU `Embedding(...)`과 `Permute(...)`를 하나로 합쳤다.
- 출력 layout은 기존 `permute_a`와 동일하다.
  - 출력 shape: `[EMBEDDING_DIM, SEQ_LEN]`
  - index mapping: `out[h, s] = emb_w[input[s], h]`
- `model.cu`에서는 `Embedding(...)`과 `Permute(...)` 두 호출을 `EmbeddingPermute(...)` 한 호출로 교체했다.

## Step 3: Conv/ReLU/MaxPool 및 Linear/ReLU 병합

- `layer.cu`에 `Conv1DReLUMax(...)`를 추가했다.
- 기존 `Conv1D(...)`, `ReLU(...)`, `GetMax(...)`를 하나로 합쳤다.
- 전체 conv activation을 `conv*_a`에 저장하지 않고 바로 pooled output인 `pool*_a`를 만든다.
- ReLU 이후 max pooling과 동일하게 동작하도록 max 초기값은 `0.f`로 둔다.
- `layer.cu`에 `LinearReLU(...)`를 추가했다.
- `linear0`, `linear1`, `linear2`는 `LinearReLU(...)`를 사용하고, ReLU가 없는 `linear3`은 기존 `Linear(...)`를 유지했다.

## Step 4: Embedding/Conv Branch Batch GPU화

- `layer.cu`에 `EmbeddingPermuteBatchCUDA(...)`를 추가했다.
- 전체 batch 입력 `[B, SEQ_LEN]`을 GPU에서 `[B, EMBEDDING_DIM, SEQ_LEN]` layout으로 만든다.
- index mapping은 `d_permute[b, h, s] = emb_w[input[b, s], h]`이다.
- `layer.cu`에 `Conv1DReLUMaxBatchCUDA(...)`를 추가했다.
- batch 전체에 대해 Conv1D, ReLU, MaxPool을 하나의 CUDA kernel에서 수행한다.
- 각 conv branch 결과는 별도 pool buffer를 거치지 않고 `d_concat[b, offset + oc]`에 바로 저장한다.
- `model.cu`는 GPU 0에서 `d_inputs`, `d_permute`, `d_concat` workspace를 할당하고, Embedding/Permute와 conv branch 4개를 batch로 처리한다.
- 이 단계에서는 Linear layers를 CPU에 남겨 검증 지점을 유지했다.

## Step 5: Linear Layers Batch GPU화

- `layer.cu`에 `LinearBatchCUDA(...)`를 추가했다.
- batch 전체에 대해 Linear를 수행하고, 필요하면 ReLU까지 함께 적용한다.
- output element 하나 `[batch, output_channel]`를 CUDA thread 하나가 계산한다.
- `model.cu`에서는 `d_concat` 이후의 linear layers를 모두 GPU에서 처리하도록 변경했다.
  - `d_concat -> d_linear0` with ReLU
  - `d_linear0 -> d_linear1` with ReLU
  - `d_linear1 -> d_linear2` with ReLU
  - `d_linear2 -> d_outputs` without ReLU
- `predict_sentiment(...)` 안의 `n_samples` loop가 제거되었다.
- 최종 `[n_samples, N_CLASSES]` output만 host `outputs`로 복사한다.

## Step 6: Conv Batch Kernel Block Reduction 최적화

- 기존 `Conv1DReLUMaxBatchKernel`은 thread 하나가 `[sample, output_channel]` 하나를 맡았다.
- 그 thread 하나가 `EMBEDDING_DIM * K * os`에 해당하는 곱셈/덧셈을 전부 직렬로 수행했다.
- 이 구조는 GPU thread 수는 많아 보여도, 각 thread 내부 일이 너무 커서 실제 병렬성이 부족했다.

- `Conv1DReLUMaxBatchOptimizedKernel`을 추가했다.
- 새 커널은 block 하나가 `[sample, output_channel]` 하나를 맡도록 바꿨다.
- block 안에는 256 threads를 두었다.
- 각 thread는 `EMBEDDING_DIM * K` dot product의 일부를 strided 방식으로 나눠 계산한다.
  - 예: thread `t`는 `t`, `t + blockDim.x`, `t + 2 * blockDim.x` 위치의 항을 담당한다.
  - 이 방식은 일반적인 matrix tiling이라기보다 dot product work partitioning에 가깝다.

- 각 thread가 계산한 partial sum은 shared memory 배열 `partial[256]`에 저장한다.
- 이후 block 내부에서 shared memory reduction을 수행한다.
  - stride를 `128 -> 64 -> 32 -> ... -> 1`로 줄이며 partial sum을 합친다.
  - 매 reduction 단계마다 `__syncthreads()`로 block 내 thread 동기화를 보장한다.
- reduction이 끝나면 `partial[0]`에 해당 `pos`의 conv 결과가 모인다.

- `threadIdx.x == 0`인 thread가 bias를 더하고 ReLU+MaxPool을 적용한다.
  - ReLU 효과를 위해 max 초기값은 `0.f`로 유지한다.
  - 각 `pos`의 conv 결과 중 최대값을 `max_val`에 저장한다.
- 모든 `pos` 처리가 끝나면 thread 0이 `d_concat[b, offset + oc]`에 최종 pooled 값을 쓴다.

- `Conv1DReLUMaxBatchCUDA(...)` wrapper는 이제 기존 naive kernel 대신 optimized kernel을 호출한다.
- 아직 적용하지 않은 최적화:
  - weight/input tile을 shared memory에 캐싱하는 진짜 tiled convolution
  - warp-level reduction
  - branch 4개를 하나의 kernel로 합치는 fusion
  - Tensor Core 또는 vectorized load 활용
