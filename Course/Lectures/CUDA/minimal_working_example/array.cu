#include<ostream>
#include<vector>
#include<stdio.h>
#include<cuda_runtime.h>

// reset GPGPUs to be safe. 
void reset_cuda_devs() {
    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);

    while( dev_count --> 0 )
    {
        printf("Resetting device %i", dev_count);
        cudaSetDevice(dev_count);
        cudaDeviceReset();
    }
    printf("\n");
}

void sizes() {
    printf("sizeof(char) = %u\n", sizeof(char));
    printf("sizeof(int) = %u\n", sizeof(int));
    printf("sizeof(unsigned int) = %u\n", sizeof(unsigned int));
    printf("sizeof(float) = %u\n", sizeof(float));
    printf("sizeof(double) = %u\n", sizeof(double));
    printf("sizeof(long double) = %u\n", sizeof(long double));
    printf("sizeof(long) = %u\n", sizeof(long));
    printf("sizeof(long long int) = %u\n", sizeof(long long int));
    printf("sizeof(unsigned long) = %u\n", sizeof(unsigned long));
}

#define LENGTH 1024
__device__ int cube(int a) {
    return a * a * a;
}

__global__ void kernel_cube_array(int * dev_array, int length) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    
    __shared__ int shared_mem[LENGTH];
    shared_mem[tidx] = dev_array[tidx];
    __syncthreads();

    shared_mem[tidx] = cube(shared_mem[tidx]);
    __syncthreads();
    
    dev_array[tidx] = shared_mem[tidx];
}

__host__ int main() {
    std::vector<int> host_array(LENGTH, 2); 

    int * dev_array = NULL; 
    cudaMalloc((void **)&dev_array, LENGTH * sizeof(int)); 
    cudaMemcpy(dev_array, &host_array[0], LENGTH * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockDims(LENGTH, 1, 1);
    dim3 gridDims(1, 1, 1);
    kernel_cube_array<<<gridDims, blockDims>>>(dev_array, LENGTH);
    cudaDeviceSynchronize();

    cudaMemcpy(&host_array[0], dev_array, LENGTH * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_array);
}


