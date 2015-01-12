# CUDA Scanning

## Description

This homework will give you the option to either implement the Hillis & Steele or Blelloch scan algorithms. 

## Instructions

  - Add the necessary code to the `cuda_parallel_scan` function and call the desired scanning kernel.
  - Implement either the `kernel_blelloch_scan` and `hs_scan_kernel`.
  - To build the program run:
    - `make clean && make debug` to build the binary with debugging symbols. 
    - `make clean && make` to build the binary.

## Learning Goals

You will learn how to perform a parallel scan across a sequence of values. In conjunction you will be able to implement a kernel allocates dynamic shared memory, and device only functions. 

## Grading

This homework is worth 100 points. 

  - Implement `cuda_parallel_scan`.
    - Allocate and Free memory on the device. +16
    - Copy memory to and from the device. +16
    - Implement a kernel call that specifies the amount of dynamic shared memory to allocate. +16
  - Implement either the `kernel_blelloch_scan` or `hs_scan_kernel` global  functions.
    - Allocate the dynamic shared memory. +16
    - Implement the desired scanning algorithm. +16
    - Implement the necessary device functions. +16
  - EXTRA CREDIT: Implement a multi-block scan. +25

