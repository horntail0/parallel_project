#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

using std::vector;

/* Macro for checking CUDA errors */
#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)


/* [Tensor Structure] */
struct Tensor {
  static const int MAX_GPU_BUFS = 4;

  size_t ndim = 0;
  size_t shape[4];
  float *buf = nullptr;
  float *d_buf[MAX_GPU_BUFS] = {nullptr, nullptr, nullptr, nullptr};
  int num_device_bufs = 0;

  Tensor(const vector<size_t> &shape_);
  Tensor(const vector<size_t> &shape_, float *buf_);
  ~Tensor();

  size_t num_elem();
  float *device_buf(int device_id);
};

typedef Tensor Parameter;
typedef Tensor Activation;
