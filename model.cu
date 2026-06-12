#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "layer.h"
#include "model.h"


/* [Model Parameters]
 * _w: Weight parameter
 * _b: Bias parameter
 */
Parameter *emb_w;
Parameter *conv0_w, *conv0_b;
Parameter *conv1_w, *conv1_b;
Parameter *conv2_w, *conv2_b;
Parameter *conv3_w, *conv3_b;
Parameter *linear0_w, *linear0_b;
Parameter *linear1_w, *linear1_b;
Parameter *linear2_w, *linear2_b;
Parameter *linear3_w, *linear3_b;

void alloc_and_set_parameters(float *param, size_t param_size) {
  size_t pos = 0;

  emb_w = new Parameter({21635, 4096}, param + pos);
  pos += 21635 * 4096; 

  conv0_w = new Parameter({1024, 4096, 3}, param + pos);
  pos += 1024 * 4096 * 3; 
  conv0_b = new Parameter({1024}, param + pos);
  pos += 1024;

  conv1_w = new Parameter({1024, 4096, 5}, param + pos);
  pos += 1024 * 4096 * 5; 
  conv1_b = new Parameter({1024}, param + pos);
  pos += 1024;

  conv2_w = new Parameter({1024, 4096, 7}, param + pos);
  pos += 1024 * 4096 * 7;
  conv2_b = new Parameter({1024}, param + pos);
  pos += 1024;

  conv3_w = new Parameter({1024, 4096, 9}, param + pos);
  pos += 1024 * 4096 * 9;
  conv3_b = new Parameter({1024}, param + pos);
  pos += 1024;

  linear0_w = new Parameter({2048, 4096}, param + pos);
  pos += 2048 * 4096;
  linear0_b = new Parameter({2048}, param + pos);
  pos += 2048;

  linear1_w = new Parameter({1024, 2048}, param + pos);
  pos += 1024 * 2048;
  linear1_b = new Parameter({1024}, param + pos);
  pos += 1024;

  linear2_w = new Parameter({512, 1024}, param + pos);
  pos += 512 * 1024;
  linear2_b = new Parameter({512}, param + pos);
  pos += 512;

  linear3_w = new Parameter({2, 512}, param + pos);
  pos += 2 * 512;
  linear3_b = new Parameter({2}, param + pos);
  pos += 2;

  if (pos != param_size) {
    fprintf(stderr, "Parameter size mismatched: %zu != %zu\n", 
            pos, param_size);
    exit(EXIT_FAILURE);
  }
}

void free_parameters() {
  delete emb_w;
  delete conv0_w;
  delete conv0_b;
  delete conv1_w;
  delete conv1_b;
  delete conv2_w;
  delete conv2_b;
  delete conv3_w;
  delete conv3_b;
  delete linear0_w;
  delete linear0_b;
  delete linear1_w;
  delete linear1_b;
  delete linear2_w;
  delete linear2_b;
  delete linear3_w;
  delete linear3_b;
}

/* [Model Activations] 
 * _a: Activation buffer
 */
Activation *emb_a;
Activation *permute_a;
Activation *conv0_a, *relu0_a, *pool0_a;
Activation *conv1_a, *relu1_a, *pool1_a;
Activation *conv2_a, *relu2_a, *pool2_a;
Activation *conv3_a, *relu3_a, *pool3_a;
Activation *concat_a;
Activation *linear0_a, *linear1_a, *linear2_a, *linear3_a;

void alloc_activations() {
  emb_a = new Activation({SEQ_LEN, 4096});
  permute_a = new Activation({4096, SEQ_LEN});
  conv0_a = new Activation({1024, SEQ_LEN - 2});
  pool0_a = new Activation({1024});
  conv1_a = new Activation({1024, SEQ_LEN - 4});
  pool1_a = new Activation({1024});
  conv2_a = new Activation({1024, SEQ_LEN - 6});
  pool2_a = new Activation({1024});
  conv3_a = new Activation({1024, SEQ_LEN - 8});
  pool3_a = new Activation({1024});
  concat_a = new Activation({4096});
  linear0_a = new Activation({2048});
  linear1_a = new Activation({1024});
  linear2_a = new Activation({512});
  linear3_a = new Activation({2});
}

void free_activations() {
  delete emb_a;
  delete permute_a;
  delete conv0_a;
  delete pool0_a;
  delete conv1_a;
  delete pool1_a;
  delete conv2_a;
  delete pool2_a;
  delete conv3_a;
  delete pool3_a;
  delete concat_a;
  delete linear0_a;
  delete linear1_a;
  delete linear2_a;
  delete linear3_a;
}

/* [Model Computation: Sentiment Analysis Task] */
void predict_sentiment(int *inputs, float *outputs, size_t n_samples) {
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if (mpi_rank == 0) {
    if (n_samples == 0) return;

    int prev_device = 0;
    CHECK_CUDA(cudaGetDevice(&prev_device));
    CHECK_CUDA(cudaSetDevice(0));

    int *d_inputs = nullptr;
    float *d_permute = nullptr;
    float *d_concat = nullptr;
    float *d_linear0 = nullptr;
    float *d_linear1 = nullptr;
    float *d_linear2 = nullptr;
    float *d_outputs = nullptr;

    size_t input_bytes = n_samples * SEQ_LEN * sizeof(int);
    size_t permute_bytes = n_samples * EMBEDDING_DIM * SEQ_LEN * sizeof(float);
    size_t concat_bytes = n_samples * N_FILTERS * 4 * sizeof(float);
    size_t linear0_bytes = n_samples * 2048 * sizeof(float);
    size_t linear1_bytes = n_samples * 1024 * sizeof(float);
    size_t linear2_bytes = n_samples * 512 * sizeof(float);
    size_t output_bytes = n_samples * N_CLASSES * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_inputs, input_bytes));
    CHECK_CUDA(cudaMalloc(&d_permute, permute_bytes));
    CHECK_CUDA(cudaMalloc(&d_concat, concat_bytes));
    CHECK_CUDA(cudaMalloc(&d_linear0, linear0_bytes));
    CHECK_CUDA(cudaMalloc(&d_linear1, linear1_bytes));
    CHECK_CUDA(cudaMalloc(&d_linear2, linear2_bytes));
    CHECK_CUDA(cudaMalloc(&d_outputs, output_bytes));

    CHECK_CUDA(cudaMemcpy(d_inputs, inputs, input_bytes,
                          cudaMemcpyHostToDevice));

    /* in [SEQ_LEN] -> out [SEQ_LEN, 4096] */
    /* in [SEQ_LEN, 4096] -> out [4096, SEQ_LEN] */
    EmbeddingPermuteBatchCUDA(d_inputs, emb_w, d_permute, n_samples, 0);

    /* in [4096, SEQ_LEN] -> out [1024, SEQ_LEN - n]  n == 2, 4, 6..*/
    /* in [1024, SEQ_LEN - 2] -> out [1024] */
    Conv1DReLUMaxBatchCUDA(d_permute, conv0_w, conv0_b, d_concat,
                           n_samples, 0, 0);
    Conv1DReLUMaxBatchCUDA(d_permute, conv1_w, conv1_b, d_concat,
                           n_samples, N_FILTERS, 0);
    Conv1DReLUMaxBatchCUDA(d_permute, conv2_w, conv2_b, d_concat,
                           n_samples, N_FILTERS * 2, 0);
    Conv1DReLUMaxBatchCUDA(d_permute, conv3_w, conv3_b, d_concat,
                           n_samples, N_FILTERS * 3, 0);

    /* in [1024 * 4] -> out [2048] */
    LinearBatchCUDA(d_concat, linear0_w, linear0_b, d_linear0,
                    n_samples, true, 0);

    /* in [2048] -> out [1024] */
    LinearBatchCUDA(d_linear0, linear1_w, linear1_b, d_linear1,
                    n_samples, true, 0);

    /* in [1024] -> out [512] */
    LinearBatchCUDA(d_linear1, linear2_w, linear2_b, d_linear2,
                    n_samples, true, 0);

    /* in [512] -> out [2] */
    LinearBatchCUDA(d_linear2, linear3_w, linear3_b, d_outputs,
                    n_samples, false, 0);

    CHECK_CUDA(cudaMemcpy(outputs, d_outputs, output_bytes,
                          cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_outputs));
    CHECK_CUDA(cudaFree(d_linear2));
    CHECK_CUDA(cudaFree(d_linear1));
    CHECK_CUDA(cudaFree(d_linear0));
    CHECK_CUDA(cudaFree(d_concat));
    CHECK_CUDA(cudaFree(d_permute));
    CHECK_CUDA(cudaFree(d_inputs));
    CHECK_CUDA(cudaSetDevice(prev_device));
  }
}
