# CUDA Race Conditions 

## Description

Implement a binning kernel that does not suffer from any race conditions. 

## Instructions
  - Implement:
    - `execute_kernel()`
    - `bin_kernel_atomic()`
  - To build the homework run `make clean && make` to build the program with debugging symbols. 

# Learning Goals

Understand the side effects of race conditions. 

# Scoring

This homework is worth a maximum of 100 points. 

  - Program runs. +50
  - `kernels.cu:execute_kernel()` implemented properly. +25
  - `kernels.cu:bin_kernel_atomic()` implemented properly. +25

