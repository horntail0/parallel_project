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
    int device_count = 0;
    CHECK_CUDA(cudaGetDevice(&prev_device));
    CHECK_CUDA(cudaGetDeviceCount(&device_count));

    int active_gpus = device_count < Tensor::MAX_GPU_BUFS ?
                      device_count : Tensor::MAX_GPU_BUFS;
    if (active_gpus > (int) n_samples) active_gpus = (int) n_samples;
    if (active_gpus <= 0) {
      fprintf(stderr, "No CUDA device available\n");
      exit(EXIT_FAILURE);
    }

    int *d_inputs[Tensor::MAX_GPU_BUFS] = {nullptr, nullptr, nullptr, nullptr};
    float *d_permute[Tensor::MAX_GPU_BUFS] = {nullptr, nullptr, nullptr, nullptr};
    float *d_concat[Tensor::MAX_GPU_BUFS] = {nullptr, nullptr, nullptr, nullptr};
    float *d_linear0[Tensor::MAX_GPU_BUFS] = {nullptr, nullptr, nullptr, nullptr};
    float *d_linear1[Tensor::MAX_GPU_BUFS] = {nullptr, nullptr, nullptr, nullptr};
    float *d_linear2[Tensor::MAX_GPU_BUFS] = {nullptr, nullptr, nullptr, nullptr};
    float *d_outputs[Tensor::MAX_GPU_BUFS] = {nullptr, nullptr, nullptr, nullptr};
    size_t chunk_start[Tensor::MAX_GPU_BUFS] = {0, 0, 0, 0};
    size_t chunk_size[Tensor::MAX_GPU_BUFS] = {0, 0, 0, 0};

    size_t base_chunk = n_samples / active_gpus;
    size_t remainder = n_samples % active_gpus;
    size_t start = 0;

    for (int gpu = 0; gpu < active_gpus; gpu++) {
      size_t batch = base_chunk + (gpu < (int) remainder ? 1 : 0);
      chunk_start[gpu] = start;
      chunk_size[gpu] = batch;
      start += batch;

      size_t input_bytes = batch * SEQ_LEN * sizeof(int);
      size_t permute_bytes = batch * EMBEDDING_DIM * SEQ_LEN * sizeof(float);
      size_t concat_bytes = batch * N_FILTERS * 4 * sizeof(float);
      size_t linear0_bytes = batch * 2048 * sizeof(float);
      size_t linear1_bytes = batch * 1024 * sizeof(float);
      size_t linear2_bytes = batch * 512 * sizeof(float);
      size_t output_bytes = batch * N_CLASSES * sizeof(float);

      CHECK_CUDA(cudaSetDevice(gpu));
      CHECK_CUDA(cudaMalloc(&d_inputs[gpu], input_bytes));
      CHECK_CUDA(cudaMalloc(&d_permute[gpu], permute_bytes));
      CHECK_CUDA(cudaMalloc(&d_concat[gpu], concat_bytes));
      CHECK_CUDA(cudaMalloc(&d_linear0[gpu], linear0_bytes));
      CHECK_CUDA(cudaMalloc(&d_linear1[gpu], linear1_bytes));
      CHECK_CUDA(cudaMalloc(&d_linear2[gpu], linear2_bytes));
      CHECK_CUDA(cudaMalloc(&d_outputs[gpu], output_bytes));

      CHECK_CUDA(cudaMemcpy(d_inputs[gpu], inputs + chunk_start[gpu] * SEQ_LEN,
                            input_bytes, cudaMemcpyHostToDevice));

      /* in [SEQ_LEN] -> out [SEQ_LEN, 4096] */
      /* in [SEQ_LEN, 4096] -> out [4096, SEQ_LEN] */
      EmbeddingPermuteBatchCUDA(d_inputs[gpu], emb_w, d_permute[gpu],
                                batch, gpu);

      /* in [4096, SEQ_LEN] -> out [1024, SEQ_LEN - n]  n == 2, 4, 6..*/
      /* in [1024, SEQ_LEN - 2] -> out [1024] */
      Conv1DReLUMaxBatchCUDA(d_permute[gpu], conv0_w, conv0_b, d_concat[gpu],
                             batch, 0, gpu);
      Conv1DReLUMaxBatchCUDA(d_permute[gpu], conv1_w, conv1_b, d_concat[gpu],
                             batch, N_FILTERS, gpu);
      Conv1DReLUMaxBatchCUDA(d_permute[gpu], conv2_w, conv2_b, d_concat[gpu],
                             batch, N_FILTERS * 2, gpu);
      Conv1DReLUMaxBatchCUDA(d_permute[gpu], conv3_w, conv3_b, d_concat[gpu],
                             batch, N_FILTERS * 3, gpu);

      /* in [1024 * 4] -> out [2048] */
      LinearBatchCUDA(d_concat[gpu], linear0_w, linear0_b, d_linear0[gpu],
                      batch, true, gpu);

      /* in [2048] -> out [1024] */
      LinearBatchCUDA(d_linear0[gpu], linear1_w, linear1_b, d_linear1[gpu],
                      batch, true, gpu);

      /* in [1024] -> out [512] */
      LinearBatchCUDA(d_linear1[gpu], linear2_w, linear2_b, d_linear2[gpu],
                      batch, true, gpu);

      /* in [512] -> out [2] */
      LinearBatchCUDA(d_linear2[gpu], linear3_w, linear3_b, d_outputs[gpu],
                      batch, false, gpu);
    }

    for (int gpu = 0; gpu < active_gpus; gpu++) {
      size_t batch = chunk_size[gpu];
      size_t output_bytes = batch * N_CLASSES * sizeof(float);

      CHECK_CUDA(cudaSetDevice(gpu));
      CHECK_CUDA(cudaMemcpy(outputs + chunk_start[gpu] * N_CLASSES,
                            d_outputs[gpu], output_bytes,
                            cudaMemcpyDeviceToHost));
    }

    for (int gpu = 0; gpu < active_gpus; gpu++) {
      CHECK_CUDA(cudaSetDevice(gpu));
      CHECK_CUDA(cudaFree(d_outputs[gpu]));
      CHECK_CUDA(cudaFree(d_linear2[gpu]));
      CHECK_CUDA(cudaFree(d_linear1[gpu]));
      CHECK_CUDA(cudaFree(d_linear0[gpu]));
      CHECK_CUDA(cudaFree(d_concat[gpu]));
      CHECK_CUDA(cudaFree(d_permute[gpu]));
      CHECK_CUDA(cudaFree(d_inputs[gpu]));
    }

    CHECK_CUDA(cudaSetDevice(prev_device));
  }
}
