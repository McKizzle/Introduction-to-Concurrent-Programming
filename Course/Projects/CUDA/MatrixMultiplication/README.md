# CUDA Matrix Multiplication

## Description
This assignment will require that you implement a classic matrix multiplication
function. Afterwards you will be required to implement the same function using 
the CUDA API and run it on the GPU. 

## Directions
  0. Documentation
    1. Crate a blank document that contains your full name and the assignment 
    name (CUDA Matrix Multiplication)
  1. Getting Started
    1. First run the command `make`. That will compile the program (there 
    should be no errors). Take a screenshot of the results. 
    2. Next run the executable `./matrixMultiply -h` that was created. It will print out the usage information. 
    3. Next run the command `make testgpu` to test the binary using `tests/run_test.pl`. While the test is running it will output results back to the screen. 
    4. Take a screenshot of the previous two steps and past it into your document. 
  2. There are three files you will need to modify in order to complete the 
  project. 
    1. Write the implementation details for your Matrix class in `Matrix.cpp`. 
    2. Write the CUDA kernel implementation in `kernels.cu`. If you decide to 
change what arguments the kernel function takes make sure to update 
`kernels.h` to reflect your changes. 
    3. Unlike the other Matrix class methods, 
`Matrix cuda_multiply_by(Matrix& B, float &cuda_ms);` needs to be located 
by the kernels.cu file. This is due to the fact that calling a kernel 
function requires using a special syntax that only the `nvcc` compiler 
understands. 
  3. Once you have implemented the necessary code and all of the tests pass. Run `make clean && make` and then run `make testgpu`. Take a screenshot of the output and paste it into your document.
  4. You will also need to make sure that the CPU matrix multiplication algorithm works. Update the code and run `make testcpu` until there are no more errors. 
  5. If your are experiencing issues with the program and would like to debug it with `cuda-gdb` then run `make clean && make debug`.

## Extra Notes
C++ Allows class methods to be defined in separate files. For example, in 
this project the `Matrix` class method `cuda_multiply_by` is in `kernels.cu` 
instead of `Matrix.cpp`. The reason is that it makes use of the CUDA 
extensions which requires that it gets compiled by `nvcc`. Even though 
`nvcc` can compile all of the source files in this directory. It is a lot 
more clean to separate kernel functions into their own file. This then allows
you to use other compilers for the rest of your code and then link the 
objects together without any ill effects. 

## Learning Goals 
After completing this project will have learned how to properly perform matrix 
multiplication on the CPU and on the GPU. In conjunction, you will have learned
how to allocate and deallocate data onto the GPU and implement a kernel 
function that takes advantage of the GPU and the allocated data. 

## Grading
The project is worth a maximum of 100 points. 

  - Submit the document that contains the screenshots. +20
  - The Matrix constructor has been properly implemented. +10
  - `get_rows` has been properly implemented. +5
  - `get_cols` has been properly implemented. +5
  - `get_value_at` has been properly implemented. +5
  - `multiply_by` has been properly implemented. +20
  - `cuda_multiply_by` has been properly implemented. +17
  - `KernelMatrixMultiply` has been properly implemented. +17
  - _Extra Credit_ Use the CUDA API to time how long your kernel function 
takes. Log this to a file. (`run_tests.pl` will get confused and report errors). +5 



