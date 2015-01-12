#include"cuda_scan.h"
#include"utils.h"

// find the log_2 of a value
inline __device__ int log2(int i) {
    int p = 0; 
    while(i >>= 1) p++;
    return p;
}

/*! \brief Blelloch scan sweep-up operation
 *
 *  Performs the sweep-up substep in the blelloch scan.
 *  
 *  @param[in, out] the sequence to sweep. 
 *  @param[in] the size of the sequence to sweep.
 */
__device__ void bl_sweep_up(int * to_sweep, int size) { 
    //1: for d = 0 to log2 n – 1 do
    //2:      for all t = 0 to n – 1 by 2^{d + 1} in parallel do
    //3:           x[t] = x[t] + x[t - t / 2]
    int t = threadIdx.x; 
    for( int d =  0; d < log2(size); ++d) {
        __syncthreads(); // wrapping the condition with synthreads prevents a rare race condition. 
        if( t < size && !((t + 1) % (1 << (d + 1)))) {
            int tp = t - (1 << d);
            to_sweep[t] = to_sweep[t] + to_sweep[tp];
        }
    }
}

/*! \brief Blelloch scan sweep-down operation
 *
 *  Performs the sweep-down substep in the blelloch scan.
 * 
 *  @param[in, out] the sequence to sweep. 
 *  @param[in] the size of the sequence to sweep.
 */
__device__ void bl_sweep_down(int * to_sweep, int size) {
    //1: x[n – 1] <- 0
    //2: for d = log2 n – 1 down to 0 do
    //3:       for all t = 0 to n – 1 by 2^{d + 1} in parallel do
    //4:            carry = x[t]
    //5:            x[t] += x[t - t / 2]
    //6:            x[t - t / 2] = carry 
    to_sweep[size - 1] = 0;
    int t = threadIdx.x;
    for( int d = log2(size) - 1; d >= 0; --d) {
        __syncthreads(); // wrapping the condition with synthreads prevents a rare race condition. 
        if( (t < size) && !((t + 1) % (1 << (d + 1)))) {
            int tp = t - (1 << d);
            int tmp = to_sweep[t];
            to_sweep[t] += to_sweep[tp];
            to_sweep[tp] = tmp;
        }
    }
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
__global__ void kernel_blelloch_scan(const int d_to_scan_size, const int * d_to_scan, int * d_scanned) {
    int tIdx = threadIdx.x;

    extern __shared__ int s_to_scan[];
    s_to_scan[tIdx] = 0;
    
    if(tIdx < d_to_scan_size) {
        //copy d_to_scanto s_to_scan memory to speed up the performance. 
        s_to_scan[tIdx] = d_to_scan[tIdx];
    }
    __syncthreads();

    // perform the sweep up phase of the algorithm 
    bl_sweep_up(s_to_scan, blockDim.x);
    // perform the sweep down phase of the algorithm 
    bl_sweep_down(s_to_scan, blockDim.x);


    if(tIdx < d_to_scan_size) {
        //copy back to global memory. 
        d_scanned[tIdx] = s_to_scan[tIdx];
    }
}

/*! \brief Block scan kernel function.
 * 
 *  The kernel_block_scan kernel takes in a sequence of elemetns and performs a block level scans
 *  accross the sequence. The block dimensions need to be of size 2^x <= 1024 (or cuda specific maximum block size). 
 *  
 *  @param[in] the sequence of integers to scan.
 *  @param[in, out] the sequence to store the block level scans. 
 *  @param[in, out] the sequence to store the reductions of each block. AKA it's length is the 
 *      length of the grid. 
 */
__global__ void kernel_block_scan(const int d_to_scan_size, const int * d_to_scan, int * d_scanned, int * d_block_reductions) {
    int tIdx = threadIdx.x;
    int gIdx = threadIdx.x + blockIdx.x * blockDim.x;
     
    // Init the dynamic shared memory. 
    extern __shared__ int s_to_scan[];
    s_to_scan[tIdx] = d_to_scan[d_to_scan_size - 1];
    
    // Copy from global to local.
    if(gIdx < d_to_scan_size) {
        s_to_scan[tIdx] = d_to_scan[gIdx];
    }
    __syncthreads();
        
    // Scan the shared block of data. 
    bl_sweep_up(s_to_scan, blockDim.x);
    bl_sweep_down(s_to_scan, blockDim.x);

    // Now store the block reduction (sum of all elets)
    if(tIdx == (blockDim.x - 1)) {
        d_block_reductions[blockIdx.x] = s_to_scan[tIdx] + d_to_scan[gIdx];
    }
    __syncthreads();

    
    //Push the results back out to global memory. 
    if(gIdx < d_to_scan_size) {
        d_scanned[gIdx] = s_to_scan[tIdx];
    }
}

/*! \brief Offset blocks
 * 
 *  The kernel_block_offset takes in a sequence of values and performs an the corresponding block
 *  offset from the offset sequence. 
 *
 *  @param[in, out] the sequence to offset. 
 *  @param[in] the corresponding block offset values.
 */
__global__ void kernel_block_offset(int d_to_offset_size, int * d_to_offset, int * d_offsets) {
    int tIdx = threadIdx.x;
    int gIdx = threadIdx.x + blockIdx.x * blockDim.x;

    extern __shared__ int s_to_offset[];
    s_to_offset[tIdx] = 0;
    
    // Copy from global to shared. 
    if(gIdx < d_to_offset_size) {
        s_to_offset[tIdx] = d_to_offset[gIdx];
    }
    __shared__ int offset;
    offset = d_offsets[blockIdx.x];
    __syncthreads();
     
    s_to_offset[tIdx] += offset;
    __syncthreads();
    
    // Copy back to global memory. 
    if(gIdx < d_to_offset_size) {
        d_to_offset[gIdx] = s_to_offset[tIdx];
    }
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
    int gDmX = (int) (to_scan.size() + block_size - 1) / block_size;
    dim3 gridDims(gDmX, 1);
    dim3 blockDims(block_size, 1);
    
    if(debug) {
        printf("---> Entering: cuda_parallel_scan\n");
        printf("\tGrid dims: %i\n\tSeq Size: %lu\n\tBlock Dims: %i\n", gDmX, to_scan.size(), block_size);
    }

    // Vector initializations
    std::vector<int> block_reductions(gDmX, 0);
    std::vector<int> scanned_block_reductions(block_reductions.size(), 0);

    // create our pointers to the device memory (d prefix)
    int * d_to_scan = 0,
        * d_scanned = 0,
        * d_block_reductions = 0;
   
    // memory allocation sizes
    unsigned int to_scan_mem_size = sizeof(int) * to_scan.size(),
                 scanned_mem_size = sizeof(int) * scanned.size(),
                 block_reductions_mem_size = sizeof(int) * block_reductions.size();

    // Initialize and begin the timers
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start));

    // Allocate the memory on the device
    checkCudaErrors(cudaMalloc((void **)&d_to_scan, to_scan_mem_size));
    checkCudaErrors(cudaMalloc((void **)&d_scanned, scanned_mem_size));
    checkCudaErrors(cudaMalloc((void **)&d_block_reductions, block_reductions_mem_size));

    // Copy from the host to the device. 
    checkCudaErrors(cudaMemcpy(d_to_scan, &to_scan[0], to_scan_mem_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_block_reductions, &block_reductions[0], block_reductions_mem_size, 
                               cudaMemcpyHostToDevice));

    // Calculate the shared memory size. 
    unsigned int shared_mem_size = sizeof(int) * blockDims.x;
     
    // Run a block level scan. 
    kernel_block_scan<<<gridDims, blockDims, shared_mem_size>>>(to_scan.size(), d_to_scan, d_scanned, d_block_reductions);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaThreadSynchronize());

    // Copy from the device to the host. 
    checkCudaErrors(cudaMemcpy(&scanned[0], d_scanned, scanned_mem_size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&block_reductions[0], d_block_reductions, block_reductions_mem_size, 
                               cudaMemcpyDeviceToHost));
    
    // TESTING: kernel_block_scan
    //int miss_index = -1;
    //miss_index = equal(cmpr_b_scanned, scanned);
    //if(miss_index > -1) {
    //    printf("\tkernel_block_scan at index %i\n", miss_index);
    //    printf("\t\tBlock level scanning didn't work as expected.\n");
    //    printf("\t\tExpected %i got %i\n", cmpr_b_scanned[miss_index], scanned[miss_index]);
    //    print_vector_comparison(cmpr_b_scanned, scanned, miss_index - 128 - 8, miss_index + 128 + 8, miss_index, blockDims.x, gridDims.x);
    //    exit(0);
    //}
    //********//
 
    // Free GPU resources before recursing.
    checkCudaErrors(cudaFree(d_to_scan));
    checkCudaErrors(cudaFree(d_scanned));
    checkCudaErrors(cudaFree(d_block_reductions));
    //checkCudaErrors(cudaFree(d_cmpr_b_scanned)); // TESTING
    
    // TESTING
    std::vector<int> cmpr_scnnd_blk_red(block_reductions.size(), 0);
    sequential_sum_scan(block_reductions, cmpr_scnnd_blk_red, false); 
    //********//

    // If the size of block_reductions is larger than a single block then recurse and call
    //  cuda_parallel_scan on the block_reductions vector. 
    // Else, pass the block_reductions vector to kernel_blelloch_scan.
    if(block_reductions.size() > (unsigned int) block_size) {
        cuda_parallel_scan(block_size, block_reductions, scanned_block_reductions, debug); 
        
        // TESTING
        // Compare and check for errors. 
        //miss_index = equal(scanned_block_reductions, cmpr_scnnd_blk_red);
        //if(miss_index > -1) {
        //    printf("\tcuda_parallel_scan at index %i\n", miss_index);
        //    printf("\t\tScanning the block reductions didn't return the expected output\n");
        //    exit(0);
        //}
        //********//
    } else {
        cuda_blelloch_scan(block_size, block_reductions, scanned_block_reductions, debug);

        // TESTING
        //miss_index = equal(scanned_block_reductions, cmpr_scnnd_blk_red);
        //if(miss_index > -1) {
        //    printf("\tcuda_blelloch_scan at index %i\n", miss_index);
        //    printf("\t\tScanning the block reductions didn't return the expected output\n");
        //    exit(0);
        //}
        //********//
    }
    
    // TESTING
    //std::vector<int> cmpr_scanned(to_scan.size(), 0);
    //sequential_block_offset(block_size, scanned, cmpr_scanned, scanned_block_reductions);
    //********//
    
    // Now perform the offsets to get a global scan.
    cuda_block_offset(block_size, scanned, scanned_block_reductions, debug);

    // TESTING
    // Compare and check for errors.  
    //miss_index = equal(scanned, cmpr_scanned);
    //if(miss_index > -1) {
    //    printf("\tcuda_block_offset at index %i\n", miss_index);
    //    printf("\t\tExpected: %i\n", cmpr_scanned[miss_index]);
    //    printf("\t\tActual  : %i\n", scanned[miss_index]);
    //    printf("\t\tOffsets: ");
    //    print_vector(scanned_block_reductions);
    //    printf("\t\tExpected %i got %i\n", cmpr_scanned[miss_index], scanned[miss_index]);
    //    print_vector_comparison(cmpr_scanned, scanned, miss_index - 64, miss_index + 64, miss_index, blockDims.x, gridDims.x);
    //    exit(0);
    //}
    //********//

    // Stop the timers
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&cuda_elapsed_time_ms, start, stop));

    return cuda_elapsed_time_ms;
}

/*! \brief CUDA Blelloch scan. 
 *
 * The cuda-blelloch-scan method will only perfrom a single block scan of a sequence using cuda. 
 * For devices with a compute capability of 3.0 then the maximum block size is 1024.
 *
 *  @param[in] the vector of elements to scan. The size of the sequence must not surpass the 
 *      maximum supported block size on the device. 
 *  @param[in, out] a vector to store the scan results. 
 */
void cuda_blelloch_scan(int block_size, std::vector<int> & to_scan, std::vector<int> & scanned, const bool debug) {
    int * d_to_scan = 0,
        * d_scanned = 0;
    
    unsigned int to_scan_mem_size = sizeof(int) * to_scan.size(),
                 scanned_mem_size = to_scan_mem_size;
    
    if(debug) {
        printf("---> Entering: cuda_blelloch_scan\n");
        printf("\tGrid dims: %i\n\tSeq Size: %lu\n\tBlock Dims: %i\n", 1, to_scan.size(), block_size);
    }

    // Allocate the memory on the devicee
    checkCudaErrors(cudaMalloc((void **)&d_to_scan, to_scan_mem_size));
    checkCudaErrors(cudaMalloc((void **)&d_scanned, scanned_mem_size));

    // Copy from the host to the device
    checkCudaErrors(cudaMemcpy(d_to_scan, &to_scan[0], to_scan_mem_size, cudaMemcpyHostToDevice));

    // Calculate the shared memory size
    unsigned int shared_mem_size = sizeof(int) * block_size;
    
    // Calculate the block and grid dimensions.
    dim3 gridDims(1, 1);
    dim3 blockDims(block_size, 1);

    // Run the blelloch scan on the block_reductions array.
    kernel_blelloch_scan<<<gridDims, blockDims, shared_mem_size>>>(to_scan.size(), d_to_scan, d_scanned);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaThreadSynchronize());

    // Copy from the device to the host
    checkCudaErrors(cudaMemcpy(&scanned[0], d_scanned, scanned_mem_size, cudaMemcpyDeviceToHost));
    
    // Clean up
    checkCudaErrors(cudaFree(d_to_scan));
    checkCudaErrors(cudaFree(d_scanned));
}

/*! \brief CUDA Parallel Offset
 *
 * Takes in a sequence and offsets the values per block. The offsets sequence contains the 
 * per block offsets. 
 *
 *  @param[in] the size of the block. 
 *  @param[in, out] the sequence of elements to perform block level offsets.
 *  @param[in] the offset values. 
 */
void cuda_block_offset(int block_size, std::vector<int> & to_offset, std::vector<int> offsets, const bool debug) {
    // Calculate the block and grid dimensions.
    int gDmX = (int) (to_offset.size() + block_size - 1) / block_size;
    dim3 gridDims(gDmX, 1);
    dim3 blockDims(block_size, 1);
    
    if(debug) {
        printf("---> Entering: cuda_block_offset\n");
        printf("\tGrid dims: %i\n\tSeq Size: %lu\n\tBlock Dims: %i\n", gDmX, to_offset.size(), blockDims.x);
    }
     
    int * d_to_offset = 0,
        * d_offsets = 0;
    
    unsigned int to_offset_mem_size = sizeof(int) * to_offset.size(),
                 offsets_mem_size = sizeof(int) * offsets.size();
    
    // Allocate the memory
    checkCudaErrors(cudaMalloc((void **)&d_to_offset, to_offset_mem_size));
    checkCudaErrors(cudaMalloc((void **)&d_offsets, offsets_mem_size));

    // Copy from the host to the device
    checkCudaErrors(cudaMemcpy(d_to_offset, &to_offset[0], to_offset_mem_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_offsets, &offsets[0], offsets_mem_size, cudaMemcpyHostToDevice));

    // Calculate the shared memory size. 
    unsigned int shared_mem_size = sizeof(int) * blockDims.x;
 
    // Execute the kernel
    kernel_block_offset<<<gridDims, blockDims, shared_mem_size>>>(to_offset.size(), d_to_offset, d_offsets);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaThreadSynchronize());
    
    // Copy from the device to the host
    checkCudaErrors(cudaMemcpy(&to_offset[0], d_to_offset, to_offset_mem_size, cudaMemcpyDeviceToHost));
    
    // Free unused memory
    checkCudaErrors(cudaFree(d_to_offset));
    checkCudaErrors(cudaFree(d_offsets));
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


