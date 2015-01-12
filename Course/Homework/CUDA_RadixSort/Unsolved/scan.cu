
#include"scan.h"
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
        __syncthreads();
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
        __syncthreads();
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


