#ifndef CUDA_RADIX_H
#define CUDA_RADIX_H

#define BLOCK_SIZE  (1 << 7) //Change the default block size. 

namespace project
{
    /*! \brief Radix Sort Kernel Prototype
     *
     * Students will need to implement this function in cuda_radix.cu. Takes in a pointer to the global 
     * unsorted array of integers sorts them at the desired exponent and then stores the results into the
     * global array d_sorted. 
     *
     * @param[in] the bit position to sort at. 
     * @param[in] the vector to sort. 
     * @param[in, out] the sorted results. 
     * @param[in, out] store the count of bits that contained 0. This will be needed inside of the bitremap
     *                 function.
     *
     !*/
    __global__ void kernel_radix_sort_1bit( const int exponent, const int * d_to_sort, 
                                            int * d_sorted, int * d_block_totals);


    /*! \brief Remaps integers in a array.
     *
     * Remaps the contents of an array to a new array in global memory. The mappings are dependent on the
     * block totals and the offsets (which is calculated using the scan operation). 
     *
     * @param[in] the vector to remap.
     * @param[in] the block totals. Refer to 'kernel_radix_sort' for a brief description of the block totals. 
     * @param[in] the offsets.
     * @param[in, out] the vector to store the mapped values. Determined by using the block totals and offset
     *      together. 
     *
     !*/
    __global__ void kernel_remap( const int * d_to_remap, const int * d_block_totals, 
                                  const int * d_offsets, int * d_global_store);
}

#endif
