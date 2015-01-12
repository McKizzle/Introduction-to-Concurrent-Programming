//////////////////////////////////////////////////////////////////////////////////////////////
//
// This reduction example was influenced by Programming Massively Parallel Processors page 102
// and Udacity's Introduction to Parallel Programming at https://github.com/udacity/cs344 
// refer to lecture 3 materials in the repository.
//
//////////////////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <vector>

#include "utils.h"

#define SEQ_SIZE 1024
#define BLOCK_SIZE 1024

/// Function prototypes
__global__ void cuda_sum_reduce(int * sequence_to_reduce, int * reduced);
int cuda_sum_reduce_launcher(std::vector<int> & sequence_to_reduce);
int sequential_sum_reduce(std::vector<int> & sequence_to_reduce);


/// Main
int main(void) {
    srand(0 /*time(NULL)*/); // call srand only once. 
    reset_cuda_devs();

    std::vector<int> sequence_to_reduce(SEQ_SIZE, 0);
    random_fill(sequence_to_reduce, 100);

    printf("Sequential Reduce: %d\n", sequential_sum_reduce(sequence_to_reduce));
    printf("GPGPU Reduce:      %d\n", cuda_sum_reduce_launcher(sequence_to_reduce));
    
    return 0;
}

/// \brief Reduction kernel.
///
/// Implementation of the reduction algorithm (Figure 6.4) on page 102 in 
/// Programming Massively Parallel Processors. Takes in a sequence of values
/// and reduces them using a binary operator. The operator for this function 
/// is the addition operator. This function will only perform a reduction 
/// across a block. 
///
/// \param Sequence of values to reduce. 
/// \param The calculated reduction. 
__global__ void cuda_sum_reduce(int * sequence_to_reduce, int * reduced) { 
    /// Allocate and initialized the shared memory. 

    /// Implement reduction loop.  
    
    /// Update the global variable reduced. 
}

/// \brief Reduces Sequence using CUDA
///
/// Allocates memory, sets up the block and grid dimensions, and executes the cuda_sum_reduce kernel.
///
/// \param The sequence to reduce. 
/// 
/// \param The reduction. 
int cuda_sum_reduce_launcher(std::vector<int> & sequence_to_reduce) {
    /// Allocate memory. 

    /// Copy to the GPU.
    
    /// Setup grid and block dimensions. 
    
    /// Execute the kernel. 

    ////Copy answer and dealloc

    return 0;
}

/// \brief Serial reduction algorithm. 
int sequential_sum_reduce(std::vector<int> & sequence_to_reduce) {
    int reduction = 0;
    for( std::vector<int>::iterator it = sequence_to_reduce.begin(); it != sequence_to_reduce.end(); ++it) {
        reduction += *it;
    }

    return reduction;
}


