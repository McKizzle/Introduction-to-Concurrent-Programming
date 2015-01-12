#ifndef RADIXSORT_H
#define RADIXSORT_H

#include<vector>

namespace project 
{
    /*! \brief Radix Sort Interface / Abstract Class
     *
     * Allows the testing of their code for their 
     * convenience. 
     !*/
    class IRadixSort
    {
        public:
            /*! \brief Default constructor.
             * 
             * Copies the contents of the to_sort vector into
             * the class's internal vector.
             *
             * @param[in] the vector to sort. Copied into the class. 
             !*/
            IRadixSort(const std::vector<int> &to_sort) {
                this->to_sort = to_sort;
                this->sorted = std::vector<int>(to_sort.size(), 0);
            };
            virtual ~IRadixSort() { }; ///< Default Destructor
             
            /*! \brief Sequential Radix Sort
             * 
             * Perform the classic sequential radix sort on the vector. 
             *
             * \returns the sorted vector reference. 
             !*/
            virtual std::vector<int> & sequential_radix_sort() = 0;

            /*! \brief Parallel Radix Sort
             * 
             * Performs paralle radix sort on the vector. Calls all
             * of the necessary CUDA memory operations and kernels
             * that will sort the data. 
             *
             * \returns the sorted vector reference. 
             !*/
            virtual std::vector<int> & parallel_radix_sort() = 0;
            
            /*! \brief Get the sorted vector. 
             *
             * \returns the sorted vector. 
             !*/
            virtual std::vector<int> & get_sorted()
            {
                return this->sorted;
            };

        protected:
            std::vector<int> to_sort; ///< This doesn't change
            std::vector<int> sorted;  ///< Storage container for the sorted contents. 
    };
    
    /*! \brief Radix Sorting Class
     * 
     * Class that contains all of the logic to sort a sequence using
     * the classic sequential radix sort and a modern parallel radix 
     * sorting algorithm. 
     *
     * The functions for this class need to be implemented in RadixSort.cpp
     * unless another location is specified. 
     !*/
    class RadixSort: public IRadixSort
    {
        public:
            RadixSort(const std::vector<int> &to_sort) : IRadixSort(to_sort) { }
            ~RadixSort() { };

            std::vector<int> & sequential_radix_sort();
            std::vector<int> & parallel_radix_sort(); ///< Implement this function inside of 'cuda_radix.cu'
    };
}

#endif

