#include<stdlib.h>

// C++ 2011 header includes. 
#include<iostream>
#include<algorithm>
#include<thread>
#include<ctime>

// Custom includes
#include"utils.h"

void serial_sorting(std::vector< std::vector<int> > S) {
    for(std::vector< std::vector< int > >::iterator s_i = S.begin();
            s_i != S.end(); ++s_i) {
        std::sort(s_i->begin(), s_i->end());
    }
}

void sort(std::vector< int > & to_sort) {
    std::sort(to_sort.begin(), to_sort.end());
}

void parallel_sorting(std::vector< std::vector<int> > & T) {
    std::vector< std::thread > threads; 
    for(std::vector< std::vector< int > >::iterator s_i = T.begin();
            s_i != T.end(); ++s_i) {
        threads.push_back(std::thread(sort, std::ref((*s_i))));
    }

    for(std::vector< std::thread >::iterator t = threads.begin(); 
            t != threads.end(); ++t) {
        (*t).join();
    }
}

int main(int argc, char * argv[]) {
    std::vector< std::vector<int> > S;
    std::vector< std::vector<int> > T;
    fill_with_vectors(S, 10000, 1000);
    copy_vector_of_vectors(S, T);
    
    std::cout << "Commencing serial sorting" << std::endl;
    auto t_start = std::chrono::high_resolution_clock::now();
    serial_sorting(S);
    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "Wall clock time: " << std::chrono::duration<double, std::milli>(t_end-t_start).count() << " ms" << std::endl;
    std::cout << std::endl; 
    std::cout << "Commencing parallel sorting" << std::endl;
    t_start = std::chrono::high_resolution_clock::now();
    parallel_sorting(T);
    t_end = std::chrono::high_resolution_clock::now();
    std::cout << "Wall clock time: " << std::chrono::duration<double, std::milli>(t_end-t_start).count() << " ms" << std::endl;

    return EXIT_SUCCESS;
}


