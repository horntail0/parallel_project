#include "layer.h"
#include "model.h"


/* Embedding
 * @param [in1]  in: [s]
 * @param [in2]   w: [NUM_VOCAB, H]
 * @param [out] out: [s, H]
 * 's' is the sequence length
 * 'H' is the embedding dimension
 */
void Embedding(int *in, Tensor* w, Tensor *out) {
  size_t s = out->shape[0];
  size_t H = out->shape[1];

  for (size_t i = 0; i < s; i++) {
    for (size_t j = 0; j < H; j++) {
      out->buf[i * H + j] = w->buf[in[i] * H + j];
    }
  }
}

/* Permute
 * @param [in]   in: [M, N]
 * @param [out] out: [N, M]
 */
void Permute(Tensor *in, Tensor *out) {
  size_t s = in->shape[0];
  size_t H = in->shape[1];

  for (size_t i = 0; i < s; i++) {
    for (size_t j = 0; j < H; j++) {
      out->buf[j * s + i] = in->buf[i * H + j];
    }
  }
}

/* Fused Embedding + Permute
 * @param [in1]  in: [s]
 * @param [in2]   w: [NUM_VOCAB, H]
 * @param [out] out: [H, s]
 */
void EmbeddingPermute(int *in, Tensor *w, Tensor *out) {
  size_t H = out->shape[0];
  size_t s = out->shape[1];

  for (size_t i = 0; i < s; i++) {
    for (size_t j = 0; j < H; j++) {
      out->buf[j * s + i] = w->buf[in[i] * H + j];
    }
  }
}

__global__ void EmbeddingPermuteBatchKernel(const int *inputs,
                                            const float *emb,
                                            float *out,
                                            size_t batch) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x; // 결과 out 의 위치를 1차원적으로 생각했을 때 인덱스.
  // index 는 그냥 일차원적으로 생각되지만 이 위치를 3차원적으로 생각하면 sample 번째 문장, hidden 번째 임베딩 float, seq번째 토큰 위치로 생각할 수 있다.
  size_t total = batch * EMBEDDING_DIM * SEQ_LEN;
  if (idx >= total) return;

  size_t seq = idx % SEQ_LEN; // 문장안에서의 토큰 위치
  size_t hidden = (idx / SEQ_LEN) % EMBEDDING_DIM; // 임베딩 차원에서의 위치. 4096임베딩 float 중 몇번째인지.
  size_t sample = idx / (EMBEDDING_DIM * SEQ_LEN); // 배치에서의 샘플 위치. n번째 문장.
  int token = inputs[sample * SEQ_LEN + seq]; // sample 번째 입력문장의 seq 번째 토큰.
  // out[sample][hidden][seq] = emb[token][hidden] // 이 token의 hidden 번째 임베딩 float이 out의 sample 번째 문장에서 행과 열이 바뀌어서 들어감.
  // 즉, 원래 embedding 에서의 위치가 열이었는데 그게 행이 되고, 토큰 위치가 원래 행이였는데 열이 됨.
  // 이것은 sample * (SEQ_LEN * EMBEDDING_DIM) + hidden * SEQ_LEN + seq 로 계산됨.
  out[idx] = emb[token * EMBEDDING_DIM + hidden];
}

void EmbeddingPermuteBatchCUDA(int *d_inputs, Tensor *w, float *d_permute,
                               size_t batch, int device_id) {
  size_t total = batch * EMBEDDING_DIM * SEQ_LEN;
  EmbeddingPermuteBatchKernel<<<(total + 255) / 256, 256>>>(
      d_inputs, w->device_buf(device_id), d_permute, batch);
  CHECK_CUDA(cudaGetLastError());
}

/* Conv1D
 * @param [in1]  in: [C, s]
 * @param [in2]   w: [OC, C, K]
 * @param [in3]   b: [OC]
 * @param [out] out: [OC, os]
 *
 *    In this model, K is 3, 5, 7, or 9,
 *    with stride = 1, pad = 0, dilation = 1.
 *    The formula for the output sequence length:
 *      os = (in - K + 2 * pad) / stride + 1
 *          = (s - K + 2 * 0) / 1 + 1
 *          = s - K + 1
 *
 * 'C' is the input channel size
 * 's' is the input sequence length
 * 'OC' is the output channel size
 * 'os' is the output sequence length
 * 'K' is the kernel (or filter) size
 */
void Conv1D(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
  size_t s = in->shape[1];
  size_t C = in->shape[0];
  size_t OC = w->shape[0];
  size_t K = w->shape[2];

  size_t os = s - K + 1;

  for (size_t i = 0; i < OC; i++) {
    for (size_t j = 0; j < os; j++) {
      float val = 0.f;
      for (size_t k = 0; k < C; k++) {
        for (size_t l = 0; l < K; l++) {
          val += in->buf[k * s + j + l] *
                  w->buf[i * C * K + k * K + l];
        }
      }
      out->buf[i * os + j] = val + b->buf[i];
    }
  }
}

/* Fused Conv1D + ReLU + GetMax
 * @param [in1]  in: [C, s]
 * @param [in2]   w: [OC, C, K]
 * @param [in3]   b: [OC]
 * @param [out] out: [OC]
 */
void Conv1DReLUMax(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
  size_t s = in->shape[1];
  size_t C = in->shape[0];
  size_t OC = w->shape[0];
  size_t K = w->shape[2];
  size_t os = s - K + 1;

  for (size_t i = 0; i < OC; i++) {
    float max_val = 0.f;
    for (size_t j = 0; j < os; j++) {
      float val = 0.f;
      for (size_t k = 0; k < C; k++) {
        for (size_t l = 0; l < K; l++) {
          val += in->buf[k * s + j + l] *
                 w->buf[i * C * K + k * K + l];
        }
      }
      val += b->buf[i];
      if (val > max_val) max_val = val;
    }
    out->buf[i] = max_val;
  }
}

__global__ void Conv1DReLUMaxBatchKernel(const float *in, const float *w,
                                         const float *bias, float *concat,
                                         size_t batch, size_t K,
                                         size_t out_offset) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = batch * N_FILTERS;
  if (idx >= total) return;

  size_t sample = idx / N_FILTERS; // 몇번째 문장인가
  size_t oc = idx % N_FILTERS; // 몇번째 필터인가. oc는 out의 채널 위치이기도 함.
  size_t os = SEQ_LEN - K + 1; // conv 거치고 나오는 sequence 길이. os는 out의 시퀀스 위치이기도 함.
  float max_val = 0.f;

  for (size_t pos = 0; pos < os; pos++) {
    float val = 0.f;
    for (size_t c = 0; c < EMBEDDING_DIM; c++) {
      for (size_t k = 0; k < K; k++) {
        // val += in[sample][c][pos+k] * w[oc][c][k]
        // in sample번째 문장의 c번 임베딩차원의 pos+k 번째 토큰 위치 값과 w의 oc번째 필터의 c번 채널의 k번째 커널 위치 값 곱해서 더하기.
        val += in[sample * EMBEDDING_DIM * SEQ_LEN + c * SEQ_LEN + pos + k] *
               w[oc * EMBEDDING_DIM * K + c * K + k];
      }
    }
    val += bias[oc];
    if (val > max_val) max_val = val;
  }

  concat[sample * (N_FILTERS * 4) + out_offset + oc] = max_val;
}

void Conv1DReLUMaxBatchCUDA(float *d_in, Tensor *w, Tensor *b,
                            float *d_concat, size_t batch,
                            size_t out_offset, int device_id) {
  size_t total = batch * N_FILTERS;
  size_t K = w->shape[2]; // w의 shape는 [OC, C, K]이므로 w->shape[2]는 K가 됨.
  // OC = N_FILTERES, C = EMBEDDING_DIM, K = 3, 5, 7, or 9.
  Conv1DReLUMaxBatchKernel<<<(total + 255) / 256, 256>>>(
      d_in, w->device_buf(device_id), b->device_buf(device_id), d_concat,
      batch, K, out_offset);
  CHECK_CUDA(cudaGetLastError());
}

/* ReLU
 * @param [in & out] inout: [N]
 * 'N' is the number of elements in the tensor.
 */
void ReLU(Tensor *inout) {
  size_t N = inout->num_elem();

  for (size_t i = 0; i < N; i++) {
    inout->buf[i] = inout->buf[i] > 0 ? inout->buf[i] : 0;
  }
}
/* ReLU CUDA kernel */
__global__ void ReLU_Kernel(float *inout, size_t N) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    inout[i] = inout[i] > 0 ? inout[i] : 0;
  }
}
/* ReLU using CUDA */
void ReLU_CUDA(Tensor *inout) {
  size_t N = inout->num_elem();

  float *d_inout;
  CHECK_CUDA(cudaMalloc(&d_inout, N * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_inout, inout->buf, N * sizeof(float),
                        cudaMemcpyHostToDevice));

  ReLU_Kernel<<<(N + 255) / 256, 256>>>(d_inout, N);
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(inout->buf, d_inout, N * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(d_inout));
}

/* GetMax
 * @param [in]   in: [C, s]
 * @param [out] out: [C]
 *
 *    This layer is to get the max value along the sequence dim.
 *    The formula for this layer: out = max(in, dim=-1)
 *
 * 'C' is the channel size
 * 's' is the sequence length
 */
void GetMax(Tensor *in, Tensor *out) {
  size_t C = in->shape[0];
  size_t s = in->shape[1];

  for (size_t i = 0; i < C; i++) {
    out->buf[i] = in->buf[i * s];
    for (size_t j = 1; j < s; j++) {
      out->buf[i] = in->buf[i * s + j] > out->buf[i] ?
        in->buf[i * s + j] : out->buf[i];
    }
  }
}

/* Concat
 * @param [in1] in1: [N1]
 * @param [in2] in2: [N2]
 * @param [in3] in3: [N3]
 * @param [in4] in4: [N4]
 * @param [out] out: [N1 + N2 + N3 + N4]
 * 'N1', 'N2', 'N3', and 'N4' are the num of elems in the tensors.
 */
void Concat(Tensor *in1, Tensor *in2, Tensor *in3, Tensor *in4,
            Tensor *out) {
  size_t N1 = in1->shape[0];
  size_t N2 = in2->shape[0];
  size_t N3 = in3->shape[0];
  size_t N4 = in4->shape[0];

  for (size_t i = 0; i < N1; i++) {
    out->buf[i] = in1->buf[i];
  }
  for (size_t i = 0; i < N2; i++) {
    out->buf[N1 + i] = in2->buf[i];
  }
  for (size_t i = 0; i < N3; i++) {
    out->buf[N1 + N2 + i] = in3->buf[i];
  }
  for (size_t i = 0; i < N4; i++) {
    out->buf[N1 + N2 + N3 + i] = in4->buf[i];
  }
}

/* Linear
 * @param [in1]  in: [N]
 * @param [in2]   w: [M, N]
 * @param [in3]   b: [M]
 * @param [out] out: [M]
 * 'N' is the input feature size
 * 'M' is the output feature size
 */
void Linear(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
  size_t N = in->shape[0];
  size_t M = w->shape[0];

  for (size_t i = 0; i < M; i++) {
    float val = 0.f;
    for (size_t j = 0; j < N; j++) {
      val += in->buf[j] * w->buf[i * N + j];
    }
    out->buf[i] = val + b->buf[i];
  }
}

/* Fused Linear + ReLU
 * @param [in1]  in: [N]
 * @param [in2]   w: [M, N]
 * @param [in3]   b: [M]
 * @param [out] out: [M]
 */
void LinearReLU(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
  size_t N = in->shape[0];
  size_t M = w->shape[0];

  for (size_t i = 0; i < M; i++) {
    float val = 0.f;
    for (size_t j = 0; j < N; j++) {
      val += in->buf[j] * w->buf[i * N + j];
    }
    val += b->buf[i];
    out->buf[i] = val > 0.f ? val : 0.f;
  }
}
