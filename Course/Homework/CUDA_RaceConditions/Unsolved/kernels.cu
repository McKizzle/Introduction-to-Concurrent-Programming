#include "kernels.h"

#define DEF_SHRD_VAL 0
#define DEF_DIM_X    1024

//!
//! This binning kernel does not prevent race conditions. 
//!
__global__ void bin_kernel_simple(int * random_sequence, int * binned, int bin_width, int rseq_len)
{
    const int index = threadIdx.x + blockIdx.x * blockDim.x;
    if ( index < rseq_len) binned[random_sequence[index] / bin_width]++;
}

//!
//! Kernel that makes use of the atomic operator to eliminate race conditions. 
//!
__global__ void bin_kernel_atomic(int * random_sequence, int * binned, int bin_width, int rseq_len)
{
    ///< TODO: Implement code that uses atomic operations to bin values. 
}

///!
///! Write the necessary code to execute binning kernels. 
///!
void execute_kernel(int * random_sequence_device, int * binned_device, int bin_width, int bins, int rseq_len, float & cuda_ms)
{
    ///< TODO: Implement the necessary code to start/stop the timers and execute the kernel. 
}

