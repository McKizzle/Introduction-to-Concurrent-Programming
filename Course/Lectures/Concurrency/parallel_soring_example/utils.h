#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <vector>
#include <cmath>

// Function declarations
void swap_pointers(int ** p0, int ** p1);
void fill_with_vectors(std::vector< std::vector<int> > & to_fill, unsigned int count, unsigned int sub_size);
void random_fill(std::vector<int> & to_fill);
void incremental_fill(std::vector<int> & to_fill);
inline void print_vector_of_vectors(const std::vector< std::vector<int> > & to_print);
void print_vector_of_ints(const std::vector<int> & to_print);
void shuffle(std::vector<int> & to_shuffle);

// Swap two pointers. 
inline void swap_pointers(int ** p0, int ** p1) {
    int * temp = *p0;
    *p0 = *p1;
    *p1 = temp;
}

// Fills a vector of vectors with vectors. 
inline void fill_with_vectors(std::vector< std::vector<int> > & to_fill, unsigned int count, unsigned int sub_size) {
    while(count --> 0) {
        std::vector<int> sub_vector(sub_size, 0);
        //std::vector<int> sub_vector((int)((float)rand() / RAND_MAX * 10000), 0);
        incremental_fill(sub_vector);
        shuffle(sub_vector);

        to_fill.push_back(sub_vector);
    }
}

// Fills a vector with random numbers. 
inline void random_fill(std::vector<int> & to_fill) {
    for( std::vector<int>::iterator it = to_fill.begin(); it != to_fill.end(); ++it) {
        *it = (int)((float)rand() / RAND_MAX * 100);
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

// Copy vector to another vector. 
inline void copy_vector_of_vectors(const std::vector< std::vector<int> > & to_copy,
        std::vector< std::vector<int> > & copied) {
    //copied.erase(to_copy.begin());
    for(std::vector< std::vector<int> >::const_iterator it = to_copy.begin(); it != to_copy.end(); ++it) {
        std::vector< int > cpy = *it;
        copied.push_back(cpy);
    }
}

// Dump the contents of a vector. 
inline void print_vector_of_vectors(const std::vector< std::vector<int> > & to_print) { 
    for(std::vector< std::vector<int> >::const_iterator it = to_print.begin(); it != to_print.end(); ++it) {
        print_vector_of_ints(*it);
    }
}

// Dump the contents of a vector. 
inline void print_vector_of_ints(const std::vector<int> & to_print) {
    for(std::vector<int>::const_iterator it = to_print.begin(); it != to_print.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;
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

#endif
