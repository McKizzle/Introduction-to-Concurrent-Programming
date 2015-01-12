/*!
 * @file main.cu 
 * @date 14 Dec 2014
 * @brief Main entry point for the scanning algorithm solutions.  
 * 
 * Unlike the Unsovled portion of this homework. The solution is able run the scan
 * algorithm on sequences that are larger than the maximum supported block size.
 * Given the difficulty of performing a multiblock scan the portion of the homework 
 * should only require students to scan a sequence that is no larger than a single block. 
 *
 */

#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<cuda_runtime.h>
#include<assert.h>
#include<vector>

#include"utils.h"

#define SEQ_SIZE    134217757 //4194319  //1083109 //(1 << 20) //8192 //((1 << 27) + ( 1 << 26) + ( 1 << 25) + ( 1 << 24))
#define BLOCK_SIZE  256     //1024

// Function Declarations
float cuda_parallel_scan(int block_size, std::vector<int> & to_scan, 
                        std::vector<int> & scanned);
float cuda_parallel_scan(int block_size, std::vector<int> & to_scan, 
                         std::vector<int> & scanned, const bool debug);
void cuda_blelloch_scan(int block_size, std::vector<int> & to_scan, 
                        std::vector<int> & scanned, const bool debug);
void cuda_block_offset(int block_size, std::vector<int> & to_offset, 
                       std::vector<int> offsets, const bool debug);
void sequential_sum_scan(std::vector<int> & in, std::vector<int> & out, 
                         bool inclusive);

/*! 
 *  \brief Program Entry Point
 *
 * First run and benchmark the serial scan. Then run and benchmark 
 * the parallel scan using different block sizes. 
 * 
 */
int main(void) {
    reset_cuda_devs();
    srand(0);

    std::vector<int> seq_to_scan(SEQ_SIZE, 0);
    std::vector<int> scanned(SEQ_SIZE, 0);
    std::vector<int> gpu_scanned(SEQ_SIZE, 0);
    random_fill(seq_to_scan, 100);
    
    clock_t t = clock();
    sequential_sum_scan(seq_to_scan, scanned, false);
    t = clock(); 
    float t_sec =  (float)t / (float) CLOCKS_PER_SEC;
    if(t_sec < 1) {
        printf("Done. Took %0.2f ms\n", t_sec * 1e3);
    } else {
        printf("Done. Took %0.2f s\n", t_sec );
    }
    
    float cuda_runtime_ms = 0.0;
            
    cuda_runtime_ms = cuda_parallel_scan(BLOCK_SIZE, seq_to_scan, gpu_scanned);

    if(cuda_runtime_ms < 1000.0) {
        printf("Done. Took %0.2f ms.\n", cuda_runtime_ms);
    } else {
        printf("Done. Took %0.2f s.\n", cuda_runtime_ms / 1000.0);
    }
    
    int miss_index = equal(scanned, gpu_scanned);
    if( miss_index != -1 ) {
        printf("Missed at index %i\n", miss_index);
        printf("Expected %i got %i\n", scanned[miss_index], gpu_scanned[miss_index]);
    } else {
        printf("Scans match.\n");
    }

    return 0;
}

/*!
 *  \brief Calculate the base 2 log on the GPU.
 */
///< TODO: Implement the log2 function on the device. 

/*! \brief Hillis and Steel sum scan algorithm 
 * 
 * Work inefficient requires O(n log(n)) operations compared to n.
 * Udacity: https://www.youtube.com/watch?v=_5sM-4ODXaA
 * Nvidia: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
 * NOTE: Slightly modified so that the code makes more sense. 
 *
 * This kernel will only work on a single block. 
 *
 * @param[in] the sequence of elements to scan. 
 * @param[in, out] results stored in the out sequence
 * @param[in] inclusive or exclusive scan. 
 */
__global__ void hs_scan_kernel(int * d_to_scan, int * d_scanned, bool inclusive) {
    // Allocate the dynamically shared memory

    // Copy to the shared memory

    // Execute the Hillis & Steel Algorithm

    // Copy the scanned results back to global memory. 
}

/*! \brief Blelloch scan sweep-up operation
 *
 *  Performs the sweep-up substep in the blelloch scan.
 *  
 *  @param[in, out] the sequence to sweep. 
 *  @param[in] the size of the sequence to sweep.
 */
__device__ void bl_sweep_up(int * to_sweep, int size) { 
    // Implement the bl_sweep_up algorithm. 
    //1: for d = 0 to log2 n – 1 do
    //2:      for all t = 0 to n – 1 by 2^{d + 1} in parallel do
    //3:           x[t] = x[t] + x[t - t / 2]
}

/*! \brief Blelloch scan sweep-down operation
 *
 *  Performs the sweep-down substep in the blelloch scan.
 * 
 *  @param[in, out] the sequence to sweep. 
 *  @param[in] the size of the sequence to sweep.
 */
__device__ void bl_sweep_down(int * to_sweep, int size) {
    // Implement the bl_sweep_down algorithm
    //1: x[n – 1] <- 0
    //2: for d = log2 n – 1 down to 0 do
    //3:       for all t = 0 to n – 1 by 2^{d + 1} in parallel do
    //4:            carry = x[t]
    //5:            x[t] += x[t - t / 2]
    //6:            x[t - t / 2] = carry 
}

/*! \brief Blelloch Scan Kernel
 *  
 *  The blelloch scan algorithm kernel. Implementation based on Udacity's and 
 *  NVidia's explanations. This will only do a block level scan.
 *
 *  Udacity: https://www.youtube.com/watch?v=_5sM-4ODXaA
 *  Nvidia: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
 *
 *  @param[in] the sequence to scan. Assumes that the sequence is no larger than the block size. 
 *  @param[in, out] the sequence to store the scanned results. 
 */
__global__ void kernel_blelloch_scan(const int * d_to_scan, int * d_scanned) {
    // Allocate the dynamic shared memory. 

    // Copy the data to the shared memory. 

    // Blelloch sweep up.

    // Blelloch sweep down.
    
    // Copy the shared memory back to global memory. 
}

/*! \brief CUDA Parallel Scan
 *  
 * See cuda_parallel_scan(int, std::vector<int>, std::vector<int>, const bool debug)
 *
 */
float cuda_parallel_scan(int block_size, std::vector<int> & to_scan, std::vector<int> & scanned) {
    return cuda_parallel_scan(block_size, to_scan, scanned, false);
}

/*! \brief CUDA Parallel Scan
 *
 *  The cuda_parallel_scan is a recursive function that takes in a large set of values and applies
 *  the scan operator across the entire set. 
 *
 *  @param[in] The size of the block to use. 
 *  @param[in] the vector of elements to scan. The length must ALWAYS be a multiple of block_size and
 *     to_scan.size() / block_size must never surpass 2147483648 on CUDA 3.0 devices. 
 *  @param[in, out] a vector to store the scan results. 
 * 
 *  @returns the runtime of all the function calls involving the device. 
 */
float cuda_parallel_scan(int block_size, std::vector<int> & to_scan, std::vector<int> & scanned, const bool debug) { 
    // For timing 
    cudaEvent_t start, stop;
    float cuda_elapsed_time_ms = 0.0;
    
    // Calculate the block and grid dimensions.    

    // Initialize and begin the timers
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start));

    // Allocate the memory on the device

    // Copy from the host to the device. 

    // Calculate the shared memory size. 
     
    // Run a block level scan. 
    ///< Kernel call goes here. 
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaThreadSynchronize());

    // Copy from the device to the host. 
     
    // Free GPU resources
     
    // Now perform the offsets to get a global scan.

    // Stop the timers
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&cuda_elapsed_time_ms, start, stop));

    return cuda_elapsed_time_ms;
}

/*! \brief Sequential Sum Scan
 * 
 *  @param[in] the vector to scan.
 *  @param[in, out] the vector to store the scanned results. 
 *  @param[in] perform an inclusive or exclusive scan. 
 */
void sequential_sum_scan(std::vector<int> & in, std::vector<int> & out, bool inclusive) { 
    assert(in.size() == out.size());

    int sum = 0;
    for( std::vector<int>::iterator in_it = in.begin(), out_it = out.begin();  
            (in_it != in.end()) && (out_it != in.end());  
            ++in_it, ++out_it) {

        if( inclusive ) {
            sum += *in_it; //include the first element of in.
            *out_it = sum;    
        } else {
            *out_it = sum; //do not include the first element of in.
            sum += *in_it;
        }
    }
}

