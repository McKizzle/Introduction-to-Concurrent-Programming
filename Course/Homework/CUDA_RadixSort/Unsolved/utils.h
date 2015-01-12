/// The original source for this file is available from Udacity's github repository for
/// their "Intro to Parallel Computing" course. 
/// url: https://raw.githubusercontent.com/udacity/cs344/master/Problem%20Sets/Problem%20Set%201/utils.h
/// Use this when programming!

#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include <cmath>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

// Pass all cuda api calls to this fuction. That will help you find mistakes quicker. 
template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

// Swap two pointers. 
inline void swap_pointers(int ** p0, int ** p1) {
    int * temp = *p0;
    *p0 = *p1;
    *p1 = temp;
}

// Fills a vector with random numbers. 
inline void random_fill(std::vector<int> & to_fill) {
    for( std::vector<int>::iterator it = to_fill.begin(); it != to_fill.end(); ++it) {
        *it = (int)((float)rand() / RAND_MAX);
    }
}

// Fills a vector of size t from 0 ... t - 1. 
inline void incremental_fill(std::vector<int> & to_fill) {
    int i = 0;
    for( std::vector<int>::iterator it = to_fill.begin(); it != to_fill.end(); ++it) {
        *it = i;
        i++;
    }
}

// Dump the contents of a vector. 
inline void print_vector(const std::vector<int> & to_print) {
    for(std::vector<int>::const_iterator it = to_print.begin(); it != to_print.end(); ++it) {
        printf("%i ", *it);
    }
    printf("\n");
}

// Performs a fisher yates shuffle on the array. 
inline void shuffle(std::vector<int> & to_shuffle) {
    for( unsigned int i = 0; i < to_shuffle.size(); ++i) {
        unsigned int j = (unsigned int)((float) rand() / RAND_MAX * (i + 1));

        int tmp = to_shuffle[j];
        to_shuffle[j] = to_shuffle[i];
        to_shuffle[i] = tmp;
    }
}

// Prints the partitioning of a vector if it were to be passed to a CUDA kernel. 
inline void print_vector_with_dims(const std::vector<int> & to_print, 
                                   const int blockDimX, const int gridDimX) {
    int bDx_counter = 0;
    int gDx_counter = 0;
    printf("\n---------------\n");
    for( std::vector<int>::const_iterator it = to_print.begin(); it != to_print.end(); ++it) {
        if(bDx_counter && !(bDx_counter % blockDimX)) {
            printf("\n");
            if(gDx_counter && !(gDx_counter % gridDimX)) {
                printf("****");
                gDx_counter++;
            } 
            printf("\n");
        }
        printf("%i ", *it);

        bDx_counter++;
    }
    printf("\n---------------\n");
}

// Print two vectors side by side. 
inline void print_vector_comparison(const std::vector<int> & v1, 
                                    const std::vector<int> & v2, 
                                    const int start_index, const int stop_index, 
                                    const int point_at, const int blockDimX, 
                                    const int gridDimX) {
    int index = 0;
    int bDx_counter = 0;
    int gDx_counter = 0;
    printf("\n---------------\n");
    for( std::vector<int>::const_iterator it1 = v1.begin(), it2 = v2.begin(); 
         (it1 != v1.end()) && (it2 != v2.end()) && (index < stop_index);
         ++it1, ++it2, ++index) {
        
        if(bDx_counter && !(bDx_counter % blockDimX)) {
            if(index >= start_index) printf("\n");
            if(gDx_counter && !(gDx_counter % gridDimX)) {
                if(index >= start_index) printf("****");
                gDx_counter++;
            } 
            if(index >= start_index) printf("\n");
        }

        if(index == point_at) {
            if(index >= start_index) printf("\n|->(%i, %i)<-|\n", *it1, *it2);
        } else {
            if(index >= start_index) printf("(%i, %i) ", *it1, *it2);
        }

        bDx_counter++; 
    }
    printf("\n---------------\n");
}

// Compare to vectors. If equal return -1 else return index. 
inline int equal(const std::vector<int> & v1, const std::vector<int> & v2) {
    int index = 0;
    for( std::vector<int>::const_iterator it1 = v1.begin(), it2 = v2.begin(); 
         (it1 != v1.end()) && (it2 != v2.end());
         ++it1, ++it2, ++index) {
        
        if(*it1 != *it2) {
            return index;
        }
    }

    return -1;
}


inline void reset_cuda_devs() {
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

#endif
