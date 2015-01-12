#include<iostream>
#include<sstream>
#include<string>
#include<vector>

#include <sysexits.h> // Program termination codes for linux! 

#include"utils.h"
#include"RadixSort.hpp"

/// Function Prototypes
void cin2vectors(std::vector< int > & v0, std::vector< int > & v1, 
                 const char del, const int vector_size);

/*! \brief Program entry point  
 *
 * Main entry point to the program. To run the application first compile 
 * and then run the following command:
 *      cat filewithseq.txt | ./radix_sort - 
 *
 * This passes in a sequence from a file to ./radix_sort and the em-dash at the end tells the program 
 * to accept from standard input. 
 *
 * \param[in] argc the number of arguments 
 * \param[in] argv the arguments. 
 *
 */
int main(int argc, char *argv[]) { 
    std::vector< int > to_sort; // the vector to sort. 
    std::vector< int > golden;  // The expected results.

    char del = '\n';
    int vector_size = -1;

    if(argc < 4) {
        std::cout << "Not enough arguments." << std::endl;
        std::cout << "Key: [optional] <required>" << std::endl;
        std::cout << "usage: \n\tcat shuffled.vec | ./radix_sort <delimiter> <vector size> - " << std::endl;
        std::cout << "usage: \n\tcat shuffled.vec [golden.vec] | ./radix_sort <delimiter> <vector size> - " << std::endl;
        
        return EX_IOERR; 
    } else {
    
        if(*argv[3] == '-') {
            del = *argv[1];
            vector_size = std::stoi(std::string(argv[2]));
            cin2vectors(to_sort, golden, del, vector_size);
            
            project::RadixSort sorter(to_sort);
            sorter.parallel_radix_sort();
            sorter.sequential_radix_sort();
            
            if(golden.size() == to_sort.size()) 
            { 
                // For personal use when building the program.
                // Just make sure not to output anything else 
                // except the sorted vector when submitting the 
                // assignment. 
            }

            print_vector(sorter.get_sorted(), del);
        } else { 
            return EX_IOERR;
        }
    }
}

/*! \brief cin to an int vector
 *
 * Reads from cin and constructs two vectors given the specified
 * length. 
 *
 * \param[in, out] The vector to fill. 
*/
void cin2vectors(std::vector< int > & v0, std::vector< int > & v1, 
                 const char del, const int vector_size) {
    int i = 0; 
    for(std::string line; std::getline(std::cin, line);) {
        std::stringstream ss(line);
        std::string number;
        while(std::getline(ss, number, del)) {
            if( i < vector_size)
            {
                v0.push_back(std::stoi(number));
            } else if(i < vector_size * 2) {
                v1.push_back(std::stoi(number));
            }
            i++;
        }
    }
}




