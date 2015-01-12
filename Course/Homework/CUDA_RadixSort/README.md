# Radix Sort Homework

## Description
Implement a block-level radix sort on a set of integers. 

# Directions
  - Complete `radix.cu` such that the following functions have been implemented. 
    1. `cuda_parallel_radix_sort` 
    2. `kernel_radix_sort`
  - `cuda_parallel_radix_sort` will serve as the staging grounds to copy data from and to the GPU. 
  - `kernel_radix_sort` will contain the parallel sorting instructions. 
  - To build the program for debugging run `make clean && make debug`. 
  - To build a release then run `make clean && make debug`. 

# Learning Goals

After completing this assignment you will have a basic understanding how to sort sequences in parallel on the GPU. In conjunction to sorting, you will have additional exposure to initializing variables in static memory.

# Grading

This homework is worth 100 points. 

  - Implement all of the necessary code in `cuda_parallel_radix_sort`. +10
  - Implement `kernel_radix_sort`.
    0. Allocate the shared memory in the kernel. +15
    1. Calculate the predicate given the target bits in the sequence of numbers. +15
    2. Using the either the supplied parallel scanning function or you own scan the predicate. +15
    3. Calculate the scatter indexes. +15
    4. Remap the sequence of numbers to their new locations. +15
    5. Repeat steps 1-5 for all of the bit locations. +15
  - EXTRA CREDIT: If possible use the scan function you created in the previous homework. +5


