# 구현 기록

## Step 0: GPU 파라미터 상주화
- `Tensor`에 device parameter buffer를 추가했다.
- 파라미터 Tensor는 최대 4개 GPU에 `d_buf`로 복사본을 유지한다.
- 기존 CPU 경로가 그대로 동작하도록 host `buf`도 유지했다.
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
- ReLU 이후 max pooling과 동일하게 동작하도록 max 초기값은 `0.f`로 두었다.
- `layer.cu`에 `LinearReLU(...)`를 추가했다.
- `linear0`, `linear1`, `linear2`는 `LinearReLU(...)`를 사용하고, ReLU가 없는 `linear3`는 기존 `Linear(...)`를 유지했다.

## Step 4: Embedding/Conv Branch Batch GPU화
- `layer.cu`에 `EmbeddingPermuteBatchCUDA(...)`를 추가했다.
- 전체 batch 입력 `[B, SEQ_LEN]`을 GPU에서 `[B, EMBEDDING_DIM, SEQ_LEN]` layout으로 만든다.
- index mapping은 `d_permute[b, h, s] = emb_w[input[b, s], h]`이다.
- `layer.cu`에 `Conv1DReLUMaxBatchCUDA(...)`를 추가했다.
- batch 전체에 대한 Conv1D, ReLU, MaxPool을 하나의 CUDA kernel에서 수행한다.
- 각 conv branch 결과는 별도 pool buffer를 거치지 않고 `d_concat[b, offset + oc]`에 바로 저장한다.
- `model.cu`는 GPU 0에서 `d_inputs`, `d_permute`, `d_concat` workspace를 할당하고, Embedding/Permute와 conv branch 4개를 batch로 처리한다.
- 이 단계에서는 Linear layers를 CPU에 남겨 검증 지점을 유지했다.

## Step 5: Linear Layers Batch GPU화
- `layer.cu`에 `LinearBatchCUDA(...)`를 추가했다.
- batch 전체에 대한 Linear를 수행하고, 필요하면 ReLU까지 함께 적용한다.
- 초기 구현은 output element 하나 `[batch, output_channel]`를 CUDA thread 하나가 계산하는 구조였다.
- `model.cu`에서는 `d_concat` 이후의 linear layers를 모두 GPU에서 처리하도록 변경했다.
  - `d_concat -> d_linear0` with ReLU
  - `d_linear0 -> d_linear1` with ReLU
  - `d_linear1 -> d_linear2` with ReLU
  - `d_linear2 -> d_outputs` without ReLU
- `predict_sentiment(...)` 안의 `n_samples` loop가 제거되었다.
- 최종 `[n_samples, N_CLASSES]` output만 host `outputs`로 복사한다.

## Step 6: Conv Batch Kernel Block Reduction 최적화
- 기존 `Conv1DReLUMaxBatchKernel`은 thread 하나가 `[sample, output_channel]` 하나를 맡았다.
- 그 thread 하나가 `EMBEDDING_DIM * K * os`에 해당하는 곱셈/덧셈을 모두 직렬로 수행했다.
- 이 구조는 GPU thread 수는 많아 보여도 각 thread 내부 일이 너무 커서 실제 병렬성이 부족했다.

- `Conv1DReLUMaxBatchOptimizedKernel`을 추가했다.
- 새 kernel은 block 하나가 `[sample, output_channel]` 하나를 맡도록 바꿨다.
- block 안에서는 256 threads를 사용한다.
- 각 thread는 `EMBEDDING_DIM * K` dot product의 일부를 strided 방식으로 나누어 계산한다.
  - 예를 들어 thread `t`는 `t`, `t + blockDim.x`, `t + 2 * blockDim.x` 위치의 일을 맡는다.
  - 이 방식은 일반적인 matrix tiling이라기보다 dot product work partitioning에 가깝다.

- 각 thread가 계산한 partial sum은 shared memory 배열 `partial[256]`에 저장한다.
- 이후 block 내부에서 shared memory reduction을 수행한다.
  - stride를 `128 -> 64 -> 32 -> ... -> 1`로 줄이며 partial sum을 합친다.
  - 매 reduction 단계마다 `__syncthreads()`로 block 내 thread 동기화를 보장한다.
- reduction이 끝나면 `partial[0]`에 해당 `pos`의 conv 결과가 모인다.

- `threadIdx.x == 0`인 thread가 bias를 더하고 ReLU+MaxPool을 적용한다.
  - ReLU 효과를 위해 max 초기값은 `0.f`로 유지한다.
  - 각 `pos`의 conv 결과 중 최댓값을 `max_val`에 저장한다.
- 모든 `pos` 처리가 끝나면 thread 0이 `d_concat[b, offset + oc]`에 최종 pooled 값을 쓴다.

- `Conv1DReLUMaxBatchCUDA(...)` wrapper는 이제 기존 naive kernel 대신 optimized kernel을 호출한다.
- 아직 적용하지 않은 최적화 후보는 다음과 같다.
  - weight/input tile을 shared memory에 캐싱하는 더 정교한 tiled convolution
  - warp-level reduction
  - branch 4개를 하나의 kernel로 합치는 fusion
  - vectorized load 적용

## Step 7: Linear Batch Kernel Tiling 최적화
- 기존 `LinearBatchKernel`은 thread 하나가 `[sample, output_channel]` 하나를 맡고, `N` 길이 dot product 전체를 혼자 계산했다.
- 이 방식은 conv 초기 구현과 비슷하게 thread 내부 직렬 작업이 커서 GPU 병렬성이 부족했다.

- `LinearBatchKernel`을 2차원 tiled GEMM 형태로 바꿨다.
- CUDA block은 `(16, 16)` thread를 사용한다.
  - `threadIdx.y` 방향은 batch sample 16개를 담당한다.
  - `threadIdx.x` 방향은 output channel 16개를 담당한다.
- grid는 `(ceil(M / 16), ceil(batch / 16))`로 구성한다.
  - `M`은 linear layer의 output feature 수다.
  - `batch`는 한 번에 처리하는 sample 수다.

- shared memory tile 두 개를 사용한다.
  - `in_tile[16][16]`: input matrix의 `[sample, input_feature]` tile
  - `w_tile[16][16]`: weight matrix의 `[output_channel, input_feature]` tile
- 각 tile 단계에서 input feature 축 `N`을 16개씩 나눠 가져온다.
- block 내부 thread들이 input과 weight 일부를 shared memory에 적재한 뒤 `__syncthreads()`로 동기화한다.
- 이후 각 thread는 shared memory에 올라온 16개 값만 사용해 자기 output element의 partial dot product를 누적한다.

- boundary 처리를 위해 tile이 `batch`, `M`, `N` 범위를 넘어가는 경우에는 `0.f`를 넣는다.
- 모든 tile 누적이 끝나면 bias를 더하고, `use_relu`가 켜져 있으면 ReLU를 적용한 뒤 `out[sample, output_channel]`에 저장한다.
- 이 변경으로 Linear 계층도 단순 element-wise thread 구조에서 shared memory와 tiling을 사용하는 구조로 바뀌었다.

- 아직 남은 Linear 최적화 후보는 다음과 같다.
  - tile 크기 조정
  - weight load coalescing 개선
  - layer별 shape에 맞춘 specialized kernel 분리
  - vectorized load 적용
  - 여러 GPU로 batch를 나누는 multi-GPU 분산

## Step 8: 단일 노드 Multi-GPU Batch Split
- 기존 `predict_sentiment(...)`는 rank 0에서 GPU 0 하나만 사용해 전체 `n_samples`를 처리했다.
- 이번 단계에서는 사용 가능한 GPU 수를 확인한 뒤 최대 4개 GPU를 사용하도록 변경했다.
  - `cudaGetDeviceCount(...)`로 실제 GPU 개수를 확인한다.
  - `Tensor::MAX_GPU_BUFS`가 4이므로 active GPU 수도 최대 4로 제한한다.
  - sample 수보다 GPU 수가 많으면 sample 수만큼만 GPU를 사용한다.

- 전체 batch를 GPU 개수만큼 나누어 chunk를 만든다.
  - `base_chunk = n_samples / active_gpus`
  - `remainder = n_samples % active_gpus`
  - 앞쪽 GPU들이 remainder를 하나씩 더 가져간다.
- 각 GPU는 자기 chunk의 시작 위치 `chunk_start[gpu]`와 크기 `chunk_size[gpu]`를 가진다.

- GPU마다 독립 workspace를 할당한다.
  - `d_inputs[gpu]`
  - `d_permute[gpu]`
  - `d_concat[gpu]`
  - `d_linear0[gpu]`
  - `d_linear1[gpu]`
  - `d_linear2[gpu]`
  - `d_outputs[gpu]`
- host input에서는 `inputs + chunk_start[gpu] * SEQ_LEN` 위치부터 해당 GPU chunk만 복사한다.

- 각 GPU에서 기존 batch CUDA 함수들을 그대로 호출한다.
  - `EmbeddingPermuteBatchCUDA(...)`
  - `Conv1DReLUMaxBatchCUDA(...)` 4개 branch
  - `LinearBatchCUDA(...)` 4개 linear layer
- 함수 호출 시 `device_id`를 GPU 번호로 넘겨 각 GPU에 상주한 parameter buffer를 사용한다.
- kernel launch 전에는 `cudaSetDevice(gpu)`로 현재 device를 맞춘다.

- 모든 GPU에 kernel launch를 끝낸 뒤, GPU별 output chunk를 host `outputs`의 원래 위치로 복사한다.
  - `outputs + chunk_start[gpu] * N_CLASSES`
- 마지막에는 GPU별 workspace를 해제하고 기존 device로 복구한다.

- 이 단계는 아직 CUDA stream을 명시적으로 사용하지 않는다.
- 다음 최적화 후보는 GPU별 stream을 만들어 H2D copy, kernel execution, D2H copy를 더 명확하게 overlap하는 것이다.

## Step 9: GPU별 CUDA Stream 적용
- Step 8에서는 GPU별 chunk를 나눴지만, 복사와 kernel 호출이 host loop 안에서 동기적으로 진행되는 부분이 남아 있었다.
- 이번 단계에서는 GPU마다 `cudaStream_t`를 하나씩 생성해 각 GPU의 작업을 독립 stream에 enqueue하도록 변경했다.

- `layer.h`와 `layer.cu`의 batch CUDA wrapper에 `cudaStream_t stream` 인자를 추가했다.
  - `EmbeddingPermuteBatchCUDA(...)`
  - `Conv1DReLUMaxBatchCUDA(...)`
  - `LinearBatchCUDA(...)`
- 각 wrapper 내부 kernel launch는 `<<<grid, block, 0, stream>>>` 형태로 바꿨다.
- 이렇게 하면 `predict_sentiment(...)`에서 지정한 stream에 kernel이 들어가므로 GPU별 작업 순서를 stream 단위로 제어할 수 있다.

- `predict_sentiment(...)`에서는 GPU마다 stream을 생성한다.
  - `cudaStreamCreate(&streams[gpu])`
- input copy는 `cudaMemcpyAsync(..., cudaMemcpyHostToDevice, streams[gpu])`로 변경했다.
- output copy도 `cudaMemcpyAsync(..., cudaMemcpyDeviceToHost, streams[gpu])`로 변경했다.

- 각 GPU stream에는 다음 순서로 작업이 들어간다.
  - host input chunk를 `d_inputs[gpu]`로 복사
  - Embedding/Permute batch kernel
  - Conv/ReLU/MaxPool batch kernel 4개
  - Linear batch kernel 4개
  - `d_outputs[gpu]`를 host output chunk로 복사
- 같은 stream 안에서는 순서가 보장되므로 별도 device-wide synchronize 없이도 데이터 의존성이 유지된다.

- 모든 GPU의 작업을 enqueue한 뒤 `cudaStreamSynchronize(streams[gpu])`로 각 stream 완료를 기다린다.
- 완료 후 GPU별 workspace를 해제하고 stream을 destroy한다.

- host 메모리가 pinned memory가 아니면 `cudaMemcpyAsync`의 overlap 효과가 제한될 수 있다.
- 다음 최적화 후보는 input/output host buffer를 pinned memory staging buffer로 옮겨 H2D/D2H copy와 kernel 실행의 overlap 효과를 더 키우는 것이다.

## Step 10: Pinned Host Staging Buffer 적용
- Step 9에서 `cudaMemcpyAsync`를 사용했지만, 원본 `inputs`와 `outputs`는 외부에서 전달되는 host pointer라 pinned memory라고 보장할 수 없다.
- pageable host memory를 대상으로 한 async copy는 내부적으로 동기적인 staging이 발생할 수 있어 copy와 kernel overlap 효과가 제한될 수 있다.

- 이번 단계에서는 GPU별 pinned host staging buffer를 추가했다.
  - `h_inputs_pinned[gpu]`
  - `h_outputs_pinned[gpu]`
- 각 GPU chunk에 대해 `cudaMallocHost(...)`로 pinned input/output staging buffer를 할당한다.
- 원본 `inputs` chunk는 CPU `memcpy`로 `h_inputs_pinned[gpu]`에 복사한다.
- H2D copy는 `h_inputs_pinned[gpu] -> d_inputs[gpu]`로 수행한다.
- D2H copy는 `d_outputs[gpu] -> h_outputs_pinned[gpu]`로 수행한다.

- H2D/D2H copy는 계속 `cudaMemcpyAsync(..., streams[gpu])`를 사용한다.
- 같은 stream 안에서 input copy, kernel들, output copy가 순서대로 실행되므로 데이터 의존성은 유지된다.
- `cudaStreamSynchronize(streams[gpu])` 이후 pinned output staging buffer의 내용을 원래 `outputs` 위치로 CPU `memcpy`한다.

- GPU별 workspace 해제 시 pinned staging buffer도 `cudaFreeHost(...)`로 해제한다.
- 이 변경은 특히 GPU별 stream overlap과 PCIe transfer overlap을 더 잘 활용하기 위한 준비 단계다.
- 다음 최적화 후보는 현재 매 호출마다 반복되는 `cudaMalloc/cudaFree/cudaMallocHost/cudaFreeHost` 비용을 줄이기 위해 workspace를 재사용하는 것이다.
