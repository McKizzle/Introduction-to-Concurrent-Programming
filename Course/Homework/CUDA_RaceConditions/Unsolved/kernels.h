#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>  
#include <iostream>
#include <cstdio>

#ifndef KERNELS_H
#define KERNELS_H
__global__ void bin_kernel(int * random_sequence, int * binned, int bin_width, int rseq_len);
void execute_kernel(int * random_sequence_device, int * binned_device, int bin_width, int bin_count, int rseq_len, float & cuda_ms);

#endif


