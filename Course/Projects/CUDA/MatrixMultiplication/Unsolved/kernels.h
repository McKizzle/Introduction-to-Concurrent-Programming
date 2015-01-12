#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>  

#ifndef KERNELS_H
#define KERNELS_H

namespace project
{
    //!
    //! GPU kernel function that performs matrix multiplication. NOTE: You will have to add
    //! additional arguments to this kernel in order to propery multiply the matrices A and B. 
    //! 
    //! \param[in] a one dimensional array of double values that represents the data in A.
    //! \param[in] a single dimensional array that stores doubles. Represents the data in B.
    //! \param[in, out] The values of the new matrix need to be stored in CD. 
    //! \param[in, out] TODO: Additional arguments may need to be added. 
    //! 
    __global__ void KernelMatrixMultiply(const double * AD, const double * BD, double * CD);
}

#endif

