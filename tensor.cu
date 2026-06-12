#include "model.h"

#include <cstring>


/* [Tensor Structure] */
/* Tensor
 * @brief - A multi-dimensional matrix containing elements of a single data
 type.
 * @member - buf  : Data buffer containing elements
 * @member - shape: Shape of tensor from outermost dimension to innermost
 dimension e.g., {{1.0, -0.5, 2.3}, {4.3, 5.6, -7.8}} => shape = {2, 3}
 */
Tensor::Tensor(const vector<size_t> &shape_) {
  ndim = shape_.size();
  for (size_t i = 0; i < ndim; i++) { shape[i] = shape_[i]; }
  size_t N_ = num_elem();
  buf = (float *) calloc(N_, sizeof(float));
}

Tensor::Tensor(const vector<size_t> &shape_, float *buf_) {
  ndim = shape_.size();
  for (size_t i = 0; i < ndim; i++) { shape[i] = shape_[i]; }
  size_t N_ = num_elem();
  buf = (float *) malloc(N_ * sizeof(float));
  memcpy(buf, buf_, N_ * sizeof(float));

  int prev_device = 0;
  int device_count = 0;
  CHECK_CUDA(cudaGetDevice(&prev_device));
  CHECK_CUDA(cudaGetDeviceCount(&device_count));

  num_device_bufs = device_count < MAX_GPU_BUFS ? device_count : MAX_GPU_BUFS;
  for (int dev = 0; dev < num_device_bufs; dev++) {
    CHECK_CUDA(cudaSetDevice(dev));
    CHECK_CUDA(cudaMalloc(&d_buf[dev], N_ * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_buf[dev], buf_, N_ * sizeof(float),
                          cudaMemcpyHostToDevice));
  }

  CHECK_CUDA(cudaSetDevice(prev_device));
}

Tensor::~Tensor() {
  int prev_device = 0;
  cudaError_t get_device_status = cudaGetDevice(&prev_device);

  for (int dev = 0; dev < num_device_bufs; dev++) {
    if (d_buf[dev] != nullptr) {
      CHECK_CUDA(cudaSetDevice(dev));
      CHECK_CUDA(cudaFree(d_buf[dev]));
      d_buf[dev] = nullptr;
    }
  }

  if (get_device_status == cudaSuccess) {
    CHECK_CUDA(cudaSetDevice(prev_device));
  }
  if (buf != nullptr) free(buf);
}

size_t Tensor::num_elem() {
  size_t size = 1;
  for (size_t i = 0; i < ndim; i++) { size *= shape[i]; }
  return size;
}

float *Tensor::device_buf(int device_id) {
  if (device_id < 0 || device_id >= num_device_bufs) {
    fprintf(stderr, "Invalid Tensor device buffer request: %d (available: %d)\n",
            device_id, num_device_bufs);
    exit(EXIT_FAILURE);
  }
  return d_buf[device_id];
}
