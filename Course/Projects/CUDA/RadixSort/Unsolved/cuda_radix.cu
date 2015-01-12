#include <vector>
#include<cuda.h>
#include<cuda_runtime.h>

#include"utils.h"
#include"cuda_radix.h" 
#include"cuda_scan.h"
#include"RadixSort.hpp"

namespace project {

//! Remember the prototype to this function is in RadixSort.hpp
//! NOTE: For simplicity make sure that to_sort.size() is always
//!     a multiple of BLOCK_SIZE in cuda_radix.h.
std::vector<int> & RadixSort::parallel_radix_sort() {
    int gDmX = (int) this->to_sort.size() / BLOCK_SIZE;
    int bDmX = BLOCK_SIZE;
    
    assert(this->to_sort.size() == this->sorted.size());
    assert((int)this->to_sort.size() >= BLOCK_SIZE);
    assert((int)this->to_sort.size() % BLOCK_SIZE == 0);
    
    /*****************************************************************
     * NEED TO IMPLEMENT THE NECESSARY CODE FOR THE METHOD.
     ****************************************************************/

    return this->sorted;
}

//---------- Kernel Implementations ------------//
//! \brief Radix Sort Kernel
__global__ void kernel_radix_sort_1bit( const int exponent, const int * d_to_sort, 
                                        int * d_sorted, int * d_block_totals) {
    /*****************************************************************
     * NEED TO IMPLEMENT THE NECESSARY CODE.
     ****************************************************************/
}

//! \brief Remap Kernel
__global__ void kernel_remap( const int * d_to_remap, const int * d_block_totals, 
                              const int * d_offsets, int * d_remapped) {
    /*****************************************************************
     * NEED TO IMPLEMENT THE NECESSARY CODE.
     ****************************************************************/
}

}
