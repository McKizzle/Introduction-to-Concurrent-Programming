#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<cuda_runtime.h>
#include<assert.h>
#include<vector>
#include<algorithm>

#include"utils.h"
#include"scan.h"

//! Function Prototypes
void binary_radix_sort(std::vector<int> & in, std::vector<int> & out);
void seq_bit_remap(const std::vector<int> &to_remap, std::vector<int> &remapped, 
                   const std::vector<int> &block_totals, const std::vector<int> &block_offsets,
                   const dim3 gridDims, const dim3 blockDims);
__global__ void kernel_radix_sort(const int * d_to_sort, int * d_sorted);
float cuda_parallel_radix_sort(std::vector<int> & to_sort, std::vector<int> & sorted);

#define BLOCK_SIZE  (1 << 10)

int main(void) {
    reset_cuda_devs();
    srand( 0 /*time(NULL)*/);

    printf("Vector Size %i\n", BLOCK_SIZE);
    printf("Estimated Memory Usage is %f MB\n", (float) (BLOCK_SIZE * sizeof(int)) / 1e6 * 4.0);
    printf("Allocating four vectors\n");
    std::vector<int> seq_to_sort(BLOCK_SIZE, 0);
    std::vector<int> sorted(BLOCK_SIZE, 0);
    std::vector<int> gpu_sorted(BLOCK_SIZE, 0);
    incremental_fill(seq_to_sort);
    shuffle(seq_to_sort);
     
    // First sort using the sequential version of radix sort.  
    printf("Performing Sequential Sort\n");
    clock_t t = clock();
    binary_radix_sort(seq_to_sort, sorted);
    t = clock();

    float t_sec =  (float)t / (float) CLOCKS_PER_SEC;
    if(t_sec < 1) {
        printf("Done. Took %0.2f ms\n", t_sec * 1e3);
    } else {
        printf("Done. Took %0.2f s\n", t_sec );
    }

    // Implement gpu radix sort algorithm. 
    printf("Performing Parallel Sort\n");
    printf("\tTo analyze the performance run 'nvprof ./radix_sort'. \n");
    float cuda_runtime_ms = cuda_parallel_radix_sort(seq_to_sort, gpu_sorted); 
    if(cuda_runtime_ms < 1000.0) {
        printf("Done. Took %0.2f ms.\n", cuda_runtime_ms);
    } else {
        printf("Done. Took %0.2f s.\n", cuda_runtime_ms / 1000.0);
    }

    int miss_index = equal(sorted, gpu_sorted);
    if( miss_index != -1 ) {
        printf("Expected %i got %i at index %i\n", sorted[miss_index], gpu_sorted[miss_index], miss_index);
    } else {
        printf("Success!\n");
    }


    return 0;
}

/// Parallel version of the radix sort kernel. 
/// This modified version of the parallel sort algorithm will only perform a single pass based on the 
/// exponent passed in. 
/// 
/// \param[in, out] d_in: The unsorted set of elements. 
/// \param[in, out] d_out:  The sorted set of elements. 
__global__ void kernel_radix_sort(const int * d_to_sort, int * d_sorted) {
    // Allocate the necessary static shared memory. 
    // Hint: Doesn't need to be larger than the block size.
    
    // Copy to the shared memroy.  
    
    // Loop through all of the bits sequence of numbers to sort. i.e. 32 bits for an integer. 
        // Calculate the predicate array for the target bits in the sequence of numbers. 

        // Perform a scan on the predicate shared array.
        
        // Construct the scatter indexes from the scanned array. 

        // Copy the elements from the unsorted shared array to the sorted shared array.  

        // Swap the shared unsorted and sorted array. 
    //End Loop

    // Copy from the sorted shared array to the global shared array. 
}

/*! \brief Parallel Radix Sort using CUDA. 
 *  
 *  Calls the necessary functions to perform a GPGPU based radix sort using the CUDA API.
 *  >>> Requires the definition of BLOCK_SIZE in the source. 
 *  
 *  @param[in, out] The sequence to sort. 
 *  @param[in] The sorted sequence. 
 *  
 *  @returns The execution time in milliseconds. 
 */
float cuda_parallel_radix_sort(std::vector<int> & to_sort, std::vector<int> & sorted) { 
    // For timing 
    cudaEvent_t start, stop;
    float cuda_elapsed_time_ms = 0.0;

    // Initialize and begin the timers
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start));
    
    // Allocate the memory on the device

    // Copy from the host to the device. 

    // Calculate the block and grid dimensions.

    // Sort the sequence in parallel. 
    // call the kernel. 
    checkCudaErrors(cudaGetLastError()); // call this after executing a kernel. 

    // Copy from the device to the host. 

    // Free up the device. 

    // Stop the timers
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&cuda_elapsed_time_ms, start, stop));

    return cuda_elapsed_time_ms;
}

/*! \brief Sequential Bit Remap
 *  
 *  Serial implementation of the bit remap kernel. 
 * 
 *  @param[in] to remap.
 *  @param[in, out] remapped.
 *  @param[in] block totals. 
 *  @param[in] block offsets. 
 */
void seq_bit_remap(const std::vector<int> &to_remap, std::vector<int> &remapped, 
                   const std::vector<int> &block_totals, const std::vector<int> &block_offsets,
                   const dim3 gridDims, const dim3 blockDims) {

    for(unsigned int bIdX = 0; bIdX < (unsigned int) gridDims.x; ++bIdX) {
        for(unsigned int tIdX = 0; tIdX < (unsigned int) blockDims.x; ++tIdX) {
            unsigned int gIdX = tIdX + bIdX * (unsigned int) blockDims.x; 
             
            if(tIdX < (unsigned int) block_totals[bIdX]) {
                unsigned int mapping = tIdX + (unsigned int) block_offsets[bIdX];
                remapped[mapping] = to_remap[gIdX];
            } else {
                unsigned int mapping = (tIdX - block_totals[bIdX]) + (unsigned int) block_offsets[gridDims.x + bIdX];
                remapped[mapping] = to_remap[gIdX];
            }
        }
    }
}

/*! \brief Binary version of radix sort. 
 * 
 *  Binary implemenation of radix sort. Function setup such that it is 
 *  easier to compare to the CUDA implementation. 
 * 
 *  @param[in] the sequence to sort. 
 *  @param[out] the sorted sequence.
 */
void binary_radix_sort(std::vector<int> & in, std::vector<int> & out) {
    out = in;
    
    std::vector<int> tmp(in.size(), 0);
    for(unsigned int exponent = 0; exponent < sizeof(int) * 8; ++exponent) {
        int i_n = 0;
        for(unsigned int i = 0; i < tmp.size(); ++i) { 
            if(!(out[i] & (1 << exponent))) {
                
                tmp[i_n] = out[i]; 
                ++i_n;
            }  
        }
        
        for(unsigned int i = 0; i < tmp.size(); ++i) {
            if(out[i] & (1 << exponent)) {
                tmp[i_n] = out[i];
                ++i_n;
            }
        }

        out = tmp;
    }
}

