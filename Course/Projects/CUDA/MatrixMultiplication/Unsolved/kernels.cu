#include <sstream> // stringstream
#include <stdexcept> // invalid_argument
#include <stdio.h> // printf
#include <cmath> // ceil

#include "kernels.h"
#include "Matrix.hpp"

namespace project
{

Matrix Matrix::cuda_multiply_by(Matrix& B, float &cuda_ms)
{  
    //Step 1: Allocate the resources onto the GPU.

    //Step 2: Setup and run the kernel. 
    
    //Step 3: Wait for all of the threads to complete their work before continueing. 
     
    //Step 4: Pull the data from the GPU. 

    //Step 5: Free the memory allocated on the GPU

    return Matrix(2, 2, 2);
}

__global__ void KernelMatrixMultiply(const double * AD, const double * BD, double * CD)
{
    printf("Devices with a Compute Capability of 3.0 allow\nthe usage of printf statements in the kernel."); 

    ///< TODO: Add the code that would calculate the value of each cell in the new matrix.
    ///< HINT: First get the index of the thread instance. 
}

}
