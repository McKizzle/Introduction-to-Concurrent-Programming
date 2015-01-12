#ifndef SCAN_H
#define SCAN_H

#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<cuda_runtime.h>
#include<assert.h>
#include<vector>

/* Kernel Function Prototypes */
__device__ int log2(int i);
__device__ void bl_sweep_up(int * to_sweep, int size);
__device__ void bl_sweep_down(int * to_sweep, int size);
__global__ void kernel_blelloch_scan(int * d_to_scan, int * d_scanned);
__global__ void kernel_block_scan(int * d_to_scan, int * d_scanned, int * d_block_reductions);
__global__ void kernel_block_offset(int * d_to_offset, int * d_offsets);

/* CUDA Function Prototypes */
float cuda_parallel_scan(int block_size, std::vector<int> & to_scan, 
                        std::vector<int> & scanned);
float cuda_parallel_scan(int block_size, std::vector<int> & to_scan, 
                         std::vector<int> & scanned, const bool debug);
void cuda_blelloch_scan(int block_size, std::vector<int> & to_scan, 
                        std::vector<int> & scanned, const bool debug);
void cuda_block_offset(int block_size, std::vector<int> & to_offset, 
                       std::vector<int> offsets, const bool debug);

/* Serial Functions */
void sequential_sum_scan(std::vector<int> & in, std::vector<int> & out, bool inclusive);

#endif
