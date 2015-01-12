# CUDA Reducing

## Description

Implement the parallel reduction algorithm explained in page 102 from Programming Massively Parallel Processors. 

## Directions

  - Complete the components within `main.cu`.
    - `cuda_sum_reduce_launcher` will need allocate and copy memory, setup the grid and block dimensions, and execute the kernel. 
    - Implement  the missing components within `cuda_sum_reduce` such that it can sum the elements of a sequence in parallel and store the result back into global memory. 
  - To compile the program run `make clean && make`

## Learning Goals

After completing this assignment you will have learned how to perform a block-level reduction on a sequence of 
values. 

## Grading

The homework is worth a maximum of 100 points.
  - `cuda_sum_reduce_launcher` needs to be able to:
    - Allocate memory on the device. +12
    - Copy data to the device. +12
    - Setup the block and grid dimensions and execute the kernel. +12
    - Copy data from the device. +12
    - Free up the memory allocated on the device. +12
  - `cudo_sum_reduce` needs to be able to:
    - Allocate shared memory. +12
    - Execute the parallel reduction loop as described on page 102 in Programming Massively Parallel Processors. +12
    - Store the result back into global memory. +12
  - EXTRA CREDIT: Use dynamic shared memory. +5. 


