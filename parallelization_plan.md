# 2노드 x 4 GPU 배치 추론 최적화 계획

## Summary

- `model.cu` 수정 가능으로 확정하고, `predict_sentiment` 안에서 전체 입력을 MPI rank별로 분할한다.
- 각 rank는 자기 노드의 GPU 4개에 local batch를 다시 분할해 병렬 추론한다.
- 파라미터는 초기화 시 각 GPU 메모리에 복사해 상주시킨다.
- 추론은 기존 모델 구조와 연산 순서를 유지하되, Embedding/Permute/Conv/ReLU/MaxPool/Concat/Linear를 CUDA 배치 커널로 구현하고 가능한 구간은 fusion한다.

## Key Changes

- `Tensor` 확장:
  - 기존 host `buf`는 유지해 파일 로딩/검증 호환성을 보장한다.
  - 파라미터 Tensor는 생성 시 4개 GPU 각각에 device buffer를 할당하고 복사한다.
  - activation Tensor는 기존 단일 샘플 CPU buffer를 유지하되, fast path에서는 별도 batch device workspace를 사용한다.

- MPI 분산:
  - `predict_sentiment(inputs, outputs, n_samples)`에서 rank 0이 sample 단위로 입력을 `MPI_Scatterv` 한다.
  - 각 rank는 `local_n`개 문장을 처리하고 결과 `[local_n, 2]`를 `MPI_Gatherv`로 rank 0에 되돌린다.
  - 출력 순서는 원래 sample index 순서를 그대로 보존한다.

- GPU 배치 추론:
  - rank별 `local_n`을 GPU 0~3에 균등 분할한다.
  - 각 GPU에서 input IDs, intermediate activations, output buffer를 batch 단위로 할당한다.
  - warmup `n=1`도 같은 경로를 타되 빈 GPU chunk는 skip한다.

- CUDA 커널:
  - Embedding과 Permute는 분리하지 않고 Conv 입력 접근에서 embedding 결과의 logical layout을 직접 사용하거나, 필요 시 `[B, 4096, 16]` layout으로 한 번만 만든다.
  - Conv1D + ReLU + MaxPool을 kernel size 3/5/7/9별 fused kernel로 만든다. 출력 전체 conv activation을 저장하지 않고 `[B, 1024]` pooled 결과만 저장한다.
  - 네 개 conv branch 결과를 `[B, 4096]` concat buffer에 바로 배치한다.
  - Linear0/1/2는 `Linear + ReLU` fused CUDA kernel, Linear3은 bias 포함 Linear kernel로 구현한다.
  - cuBLAS/cuDNN 등 외부 CUDA 라이브러리는 사용하지 않는다.

## Step-By-Step Implementation

1. Baseline 확인:
   - 현재 CPU 구현으로 작은 `-n` validation을 통과하는지 확인한다.
   - 가능하면 `-n 1`, `-n 64` 기준 elapsed time을 기록한다.

2. GPU 파라미터 상주화:
   - `Tensor`에 `float *d_buf[4]`와 device allocation 여부를 추가한다.
   - 파라미터 생성 시 GPU 0~3에 복사하고, 소멸자에서 해제한다.
   - host `buf`는 기존 코드 호환을 위해 그대로 둔다.

3. 단일 GPU fast path:
   - `layer.h/cu`에 batch inference helper를 추가한다.
   - GPU 0 하나에서 fused conv/pool, linear kernels로 `local_n` 전체를 처리한다.
   - rank 0, GPU 0만 사용해 validation을 먼저 맞춘다.

4. 4 GPU 확장:
   - 한 rank 안에서 batch를 GPU 4개로 나눈다.
   - 각 GPU chunk를 독립 stream에서 처리하고, 결과만 host local output으로 복사한다.
   - 모든 GPU 동기화 후 rank local output을 완성한다.

5. MPI 확장:
   - `predict_sentiment`에서 rank별 sample counts/displacements를 계산한다.
   - rank 0만 가진 input/output을 scatter/gather로 연결한다.
   - 2노드 실행에서 validation과 출력 순서를 확인한다.

## Test Plan

- Correctness:
  - `-n 1 -v`
  - `-n 8 -v`
  - `-n 64 -v`
  - 가능하면 실제 평가 크기에 가까운 대량 `-n`으로 validation

- Parallel cases:
  - `NODES=1`, 1 rank x 4 GPU
  - `NODES=2`, 2 ranks x 4 GPU
  - `n_samples < mpi_size * 4`인 경우 빈 chunk 처리 확인

- Performance:
  - CPU baseline 대비 throughput 비교
  - 단일 GPU fast path, 4 GPU, 2노드 순서로 단계별 throughput 기록

## Assumptions

- `model.cu` 수정이 제출/채점에서 허용된다.
- 평가 입력은 대량 배치이며 throughput이 핵심 지표다.
- validation 기준은 현재 `main.cpp`의 `1e-3` 오차 기준을 따른다.
- 모델 구조 변경 없이 동일한 Embedding, Conv1D, ReLU, MaxPool, Concat, Linear 연산을 수행한다.
