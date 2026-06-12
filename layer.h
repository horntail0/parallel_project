#pragma once

#include "tensor.h"


/* Operations (layers) */
void Embedding(int *in, Tensor *w, Tensor *out);
void Permute(Tensor *in, Tensor *out);
void EmbeddingPermute(int *in, Tensor *w, Tensor *out);
void EmbeddingPermuteBatchCUDA(int *d_inputs, Tensor *w, float *d_permute,
                               size_t batch, int device_id);
void Conv1D(Tensor *in, Tensor *w, Tensor *b, Tensor *out);
void Conv1DReLUMax(Tensor *in, Tensor *w, Tensor *b, Tensor *out);
void Conv1DReLUMaxBatchCUDA(float *d_in, Tensor *w, Tensor *b,
                            float *d_concat, size_t batch,
                            size_t out_offset, int device_id);
void ReLU(Tensor *inout);
void GetMax(Tensor *in, Tensor *out);
void Concat(Tensor *in1, Tensor *in2, Tensor *in3, Tensor *in4, 
            Tensor *out);
void Linear(Tensor *in, Tensor *w, Tensor *b, Tensor *out);
void LinearReLU(Tensor *in, Tensor *w, Tensor *b, Tensor *out);
void LinearBatchCUDA(float *d_in, Tensor *w, Tensor *b, float *d_out,
                     size_t batch, bool use_relu, int device_id);

/* Example of using CUDA kernel */
void ReLU_CUDA(Tensor *inout);
