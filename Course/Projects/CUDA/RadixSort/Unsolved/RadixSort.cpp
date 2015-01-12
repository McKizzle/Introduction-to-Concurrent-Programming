#include<vector>

#include"RadixSort.hpp"

namespace project
{

/*!
 * Implemented such that it is easy to compare the differences between the
 * CUDA radix sort and serial radix sort. 
 !*/
std::vector<int> & RadixSort::sequential_radix_sort() {
    
    /********************************************************************
     * NEED TO IMPLEMENT THE METHOD
     *******************************************************************/

    return this->sorted;
} // END sequential_radix_sort



///< RadixSort::parallel_radix_sort needs to be implemented in cuda_radix.cu

}


