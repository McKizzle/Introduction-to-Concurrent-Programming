# CUDA Radix Sort Project

## Description
This project requires that you implement the radix sort algorithm. The first 
implementation will focus on the sequential algorithm while the second will 
focus on the parallel algorithm. 

## Directions
  0. Documentation
    1. Crate a blank document that contains your full name and the assignment 
    name (CUDA Radix Sort)
  1. Getting Started
    1. First run the command `make debug`. That will compile the program 
    (there should be no errors). Take a screenshot of the results. 
    2. Next run the executable `./radix_sort` that was created. It should 
    display some usage information. 
    4. Take a screenshot of the previous steps and past it into your document. 
  2. There are three files you will need to modify in order to complete the 
  project. 
    1. Write the implementation details for your RadixSort class in 
    `RadixSort.cpp`. 
    2. Write the CUDA kernel implementations in `cuda_radix.cu`. If you decide 
    to change what arguments the kernel function takes make sure to update 
    `cuda_radix.h` to reflect your changes. 
    3. Unlike the other RadixSort class methods, 
    `std::vector<int> & parallel_radix_sort()` needs to be located inside the 
    `cuda_radix.cu` file. This is due to the fact that calling a kernel 
    function requires using a special syntax that only the `nvcc` compiler 
    understands. 
  3. Once you have implemented the necessary code and all of the tests pass. 
  Take a screenshot of the output and paste it into your document. 

## Compilation Directions
  1. To build a debugable version of the program run `make clean && make debug`
  2. To build a release version of the program run `make clean && make`

## Running the Program
  1. To run the program pass in a newline delimited vector from a file. 
  For example if the vector file is called `256.vec` then run 
  `cat 256.vec | ./radix_sort -`. 
  2. To test the program run `tests/run_tests.sh`.

## Extra Notes
  C++ Allows class methods to be defined in separate files. For example, 
  in this project the `RadixSort` class method `parallel_radix_sort` is in 
  `cuda_radix.cu` instead of `RadixSort.cpp`. The reason is that it makes use 
  of the CUDA extensions which requires that it gets compiled by `nvcc`. Even 
  though `nvcc` can compile all of the source files in this directory. It is a 
  lot more clean to separate kernel functions into their own file. This then 
  allows you to use other compilers for the rest of your code and then link the 
  objects together without any ill effects. 

## Learning Goals 
After completing this project. You will have learned how to sort a large 
dataset on the GPU and how to deal with the difficulties of writing an 
algorithm that is not limited by the block size.

## Grading
The project is worth a maximum of 100 points. 

  - Submit the document that contains the screenshots. +20
  - `sequential_radix_sort` has been properly implemented. +5
  - `parallel_radix_sort` has been properly implemented. +30
  - `kernel_radix_sort_1bit` has been properly implemented. +30
  - `kernel_remap` has been properly implemented. +15
  - _Extra Credit_  Modify your scan function from the homeworks so that it 
  can be used in this project. +10 



