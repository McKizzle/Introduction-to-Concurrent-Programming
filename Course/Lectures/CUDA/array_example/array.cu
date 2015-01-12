#include<ostream>
#include<stdio.h>
#include<cuda_runtime.h>

#define HOST_LENGTH (unsigned int) 600000000 // Approx 4.8G when using an int array. 
#define DEV_LENGTH (unsigned int)   60000000 // Approx 480M on the GPU. 

/// cuda_cube_array
__global__ void cuda_cube_array(int * dev_array, int length) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;

    if(tidx < length) {
        int array_i = dev_array[tidx];
        dev_array[tidx] = array_i * array_i * array_i;
    }
      
}

// Cube all elements in an array. 
void cube_array(int * array, unsigned int length) {
    for (unsigned int i = 0; i < length; ++i) {
        int array_i = array[i];
        array[i] = array_i * array_i * array_i;
    }
}

void set_array(int * array, unsigned int length, int value) {
    for(unsigned int i = 0; i < length; ++i) {
        array[i] = value;
    }
}

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

    exit(0);
}

int main() {
    reset_cuda_devs();
    //sizes();
    
    float calloc_size_mb = (float)(HOST_LENGTH * sizeof(int)) / 1e6;
    printf("Allocating %0.2f MB of Memory\n", calloc_size_mb);
    int * host_array = (int *)calloc(HOST_LENGTH, sizeof(int));
    int * dev_array = NULL;

    if(host_array == NULL) { 
        printf("Failed to allocate %0.2f MB of memory. :(\n", calloc_size_mb);
        return 0;
    }

    printf("----- Testing GPGPU Runtime -----\n");
    printf("Initializing data.\n"); 
    set_array(host_array, HOST_LENGTH, 2);
    
    printf("Running GPGPU Test...\n");
    dim3 blckDims(1024, 1);
    dim3 gridDims(DEV_LENGTH / blckDims.x, 1); 

    float kernel_ms = 0;
    float mcpy_to_dev_ms = 0;
    float mcpy_to_host_ms = 0;
    float ave_kernel_ms = 0;
    float ave_mcpy_to_dev_ms = 0;
    float ave_mcpy_to_host_ms = 0;
    cudaEvent_t start, stop;
    
    float cuda_malloc_size_mb = (float)(DEV_LENGTH * sizeof(int)) / 1e6;
    //cudaMallocHost((void **)&dev_array, DEV_LENGTH * sizeof(int));
    cudaMalloc((void **)&dev_array, DEV_LENGTH * sizeof(int));    
    for(unsigned int i = 0; i < HOST_LENGTH / DEV_LENGTH; i++) {
        printf("--- [%i] ---\n", i);

        unsigned int start_idx = DEV_LENGTH * i;
        
        // Measure the time it takes to copy the data over to the device. 
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        cudaMemcpy(dev_array, &host_array[start_idx], DEV_LENGTH * sizeof(int), cudaMemcpyHostToDevice); 
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&mcpy_to_dev_ms, start, stop); 
        ave_mcpy_to_dev_ms += mcpy_to_dev_ms;
        float rate = (float)(DEV_LENGTH * sizeof(int)) / 1e3 / mcpy_to_dev_ms;
        printf("Memcpy Host to Device | \tTime: %0.2f ms\t Data: %0.2f MB\t Rate: %0.2f MB/s\n", 
                mcpy_to_dev_ms, 
                cuda_malloc_size_mb, 
                rate);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
         
        // Setup timers and execute the kernel. 
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        cuda_cube_array<<<gridDims, blckDims>>>(dev_array, DEV_LENGTH); // execute the kernel. 
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&kernel_ms, start, stop);
        ave_kernel_ms += kernel_ms;
        printf("Kernel Execution      | \tTime: %0.2f ms\n", kernel_ms);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        // Measure the time it takes to copy the data back to the host. 
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        cudaMemcpy(&host_array[start_idx], dev_array, DEV_LENGTH * sizeof(int), cudaMemcpyDeviceToHost);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&mcpy_to_host_ms, start, stop);
        ave_mcpy_to_host_ms += mcpy_to_host_ms;
        rate = (float)(DEV_LENGTH * sizeof(int)) / 1e3 / mcpy_to_host_ms;
        printf("Memcpy Device to Host | \tTime: %0.2f ms\t Data: %0.2f MB\t Rate: %0.2f MB/s\n", 
                mcpy_to_host_ms, 
                cuda_malloc_size_mb, 
                rate);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    printf("-----------\n"); 
    cudaFree(dev_array); // clean up. 

    printf("Done.\n");
    printf("GPGPU Results.\n");
    ave_kernel_ms /= HOST_LENGTH / DEV_LENGTH;
    ave_mcpy_to_dev_ms /= HOST_LENGTH / DEV_LENGTH;
    ave_mcpy_to_host_ms /= HOST_LENGTH / DEV_LENGTH;
    printf("Average Times\n");
    printf("\tMemcpy host to device:\t %0.2f ms\n", ave_mcpy_to_dev_ms);
    printf("\tKernel:               \t %0.4f ms\n", ave_kernel_ms);
    printf("\tMemcpy device to host:\t %0.2f ms\n", ave_mcpy_to_host_ms);

    printf("----- Testing CPU Runtime -----\n");
    printf("Initializing the data.\n");
    set_array(host_array, HOST_LENGTH, 2);

    printf("Running the CPU test... ");
 
    clock_t t = clock();
    cube_array(host_array, HOST_LENGTH);
    t = clock() - t;

    printf("Done.\n");
    printf("CPU Results\n");
    printf("\tCPU Execution:\t %0.2f ms\n", (float) t / 1e3);
    
    printf("Cleaning up.\n");
    free(host_array);
}


