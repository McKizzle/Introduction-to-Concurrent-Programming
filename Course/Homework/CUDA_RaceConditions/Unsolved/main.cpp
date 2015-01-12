#include "kernels.h"

#include<string>
#include<iostream> // cout, endl
#include<cstdlib>
#include <ctime> // clock
#include <chrono> // time conversion to milliseconds.
#include<vector>
#include<array>
#include<ctime>
#include<cuda_runtime.h>
#include<cassert>

//!
//! Used to test the results of the CUDA kernel. 
//! 
void sequential_binner(std::vector<int> & rseq, std::vector<int> & bins, int bin_width)
{
    for(const auto& elmt: rseq) { // foreach is only available in C++11 or greater.  
        bins[elmt / bin_width] += 1;
    }
}

//!
//! Print the values in the bins
//!
void print_bin_values(std::vector<int> & bins) 
{ 
    for(const int& elmt: bins) { 
        std::cout << elmt << " "; 
    }
    std::cout << std::endl;
}

int sum(std::vector<int> & arry) {
    int elmt_sum = 0;
    for(const int& elmt: arry) { 
        elmt_sum += elmt; 
    }
    return elmt_sum;    
}

int main(int argc, char * argv[]) {
    // First generate sequence of random numbers. 
    const uint len = 1000000; // length of random numbers. 
    const int max_rand_size = 100; // set the maximum generated value. 
    const int bin_width = 10;
    const int bins = float(max_rand_size) / float(bin_width) + 0.5;
    std::vector<int> rand_sequence(len, 0);
    std::vector<int> golden_binned(bins, 0);
    std::vector<int> binned(bins, 0);
    int * binned_device;
    int * rand_sequence_device;

    std::cout << "Sequence Len: " << len << std::endl;
    std::cout << "Sequence Range: " << 0 << "-" << max_rand_size - 1 << std::endl;
    std::cout << "Bins: " << bins << std::endl;
    std::cout << "Bin Width: " << bin_width << std::endl;
   
    // Generate the random values. 
    std::srand(std::time(0));
    for(auto& elmt: rand_sequence) {
        elmt = int(float(std::rand()) / float(RAND_MAX) * max_rand_size);
    }
 
    std::clock_t c_start = std::clock(); // CPU start time. 
    std::clock_t c_stop = std::clock(); // CPU stop time. 
    auto t_start = std::chrono::high_resolution_clock::now(); // Wall start time.
    auto t_stop = std::chrono::high_resolution_clock::now(); // Wall stop time.

    // Perform the measurements.
    c_start = std::clock();
    t_start = std::chrono::high_resolution_clock::now(); 
    sequential_binner(rand_sequence, golden_binned, bin_width); 
    t_stop = std::chrono::high_resolution_clock::now();
    c_stop = std::clock();

    // Print the golden bin values. 
    std::cout << std::endl << "------------ CPU ------------" << std::endl;
    print_bin_values(golden_binned);
    std::cout << "Sum: " << sum(golden_binned) << std::endl;

    // Now lets allocate memory onto cuda and copy the data over. 
    std::cout << std::endl << "------------ GPU ------------" << std::endl;
    cudaMalloc((void**) &binned_device, sizeof(int) * bins);
    cudaMalloc((void**) &rand_sequence_device, sizeof(int) * len);
    cudaMemcpy(rand_sequence_device, (void*) &rand_sequence[0], sizeof(int) * len,  cudaMemcpyHostToDevice);
    cudaMemcpy(binned_device, (void*) &binned[0], sizeof(int) * bins, cudaMemcpyHostToDevice);
    
    float cuda_ms = 0.0;
    execute_kernel( rand_sequence_device, binned_device, bin_width, bins, len, cuda_ms); // execute the kernels in kernels.cu

    // copy back to the host
    cudaMemcpy(&binned[0], binned_device, sizeof(int) * bins, cudaMemcpyDeviceToHost);
    cudaFree(binned_device);
    cudaFree(rand_sequence_device);
    
    print_bin_values(binned);
    std::cout << "Sum: " << sum(binned) << std::endl;
    
    if(sum(golden_binned) != sum(binned)) std::cout << "Sums don't match " << std::endl;
     
    std::cout << std::endl;
    std::cout << "Kernel execution time: " << cuda_ms << std::endl;
    std::cout << "CPU execution time: " << 1000 * (c_stop - c_start) / CLOCKS_PER_SEC << std::endl;
    
    
    return 0;
}



