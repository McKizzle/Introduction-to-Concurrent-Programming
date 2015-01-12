#include <iostream> // cout, endl
#include <sstream> // stringstream
#include <stdexcept> // invalid_argument
#include <algorithm> // copy
#include <utility> // swap
#include <ctime> // clock
#include <chrono> // time conversion to milliseconds.
#include <string> // string
#include <array> //array
#include <cassert> // assert

#include <sysexits.h> // Program termination codes for linux! Woot woot! :)
#include <getopt.h> // for parsing command line arguments. 

#include "Matrix.hpp"

//--- Function Prototypes ---//
void stdin_to_two_matrices(project::IMatrix & A, project::IMatrix & B, char delimiter);
float optarg_to_float(const char * optarg); 
void dimstring4dims(char * dimensions, unsigned int &r, unsigned int &c);
void print_help();
int run_tests();

int main(int argc, char * argv[])
{
    int help_flag = 0;
    int gpu_flag  = 0;

    const struct option longopts[] 
    {
        // name,              has arg,                flag,          val
        {"help",              no_argument,            &help_flag,     1},
        {"A-dimensions",      required_argument,      NULL,          'a'},
        {"B-dimensions",      required_argument,      NULL,          'b'},
        {"delimiter",         required_argument,      NULL,          'd'},
        {"gpu",               no_argument,            &gpu_flag,     'g'},
        {NULL,                0,                      NULL,           0 }
    };

    // you need this for the short options.
    const char * short_opts = "hga:b:d:";    

    char delimiter = ' ';
    unsigned int Ar = 2, Ac = 2, Br = 2, Bc = 2;
    
    int longopts_index = 0;
    int iarg = 0;
    while((iarg = getopt_long(argc, argv, short_opts, longopts, 
                    &longopts_index)) != -1)
    {
        switch(iarg)
        {
            case 'h':
                /// print out help information
                print_help();
                return EX_OK;
                break;
            case 'a':
                /// set the dimensions of matrix A
                dimstring4dims(optarg, Ar, Ac);
                break;
            case 'b':
                /// Set the dimensions of B
                dimstring4dims(optarg, Br, Bc);
                break;
            case 'g':
                /// Set GPU flag
                gpu_flag = true;
               break; 
            case 'd':
                /// Set delimiter 
                delimiter = optarg[0]; 
                break;
            default:
                std::cout << "IARG: " << std::to_string(iarg) << std::endl;
                print_help();
                return EX_IOERR;
        }
    }

    project::Matrix A(Ar, Ac, 0.0);
    project::Matrix B(Br, Bc, 0.0);
    stdin_to_two_matrices(A, B, delimiter);

    if(gpu_flag) {
        float cuda_ms = 0.0;
        project::Matrix C = A.cuda_multiply_by(B, cuda_ms);
        std::cout << project::IMatrix::dump_to_string(C, delimiter) << std::endl;
    } else {
        project::Matrix C = A.multiply_by(B);
        std::cout << project::IMatrix::dump_to_string(C, delimiter) << std::endl;
    }

    return EX_OK;
}

/// 
/// Takes in two initialized matrices and populates with values from stdin
///
/// \param[in, out] An initialized IMatrix object. 
/// \param[in, out] An initialized IMatrix object. 
/// 
void stdin_to_two_matrices(project::IMatrix & A, project::IMatrix & B, char delimiter) {
    unsigned int i = 0;

    for(std::string line; std::getline(std::cin, line);) {
        std::stringstream ss(line);
        std::string number;
        while(std::getline(ss, number, delimiter)) {
            if(number.size() >= 1) {
                if(i < (A.get_row_count() * A.get_col_count()) ) {
                    A.set_value_at(i, std::stof(number));
                } else {
                    int bi = i - A.get_row_count() * A.get_col_count();
                    B.set_value_at(bi, std::stof(number));
                }
                i++;
            }
        }
    }
}

///
/// Takes in an optarg (char pointer) and converts it into a
/// float.
/// 
/// \param[in] optarg The optarg char * from getopt.h
/// \returns a float representation of the optarg value.
float optarg_to_float(const char * optarg)
{
    std::string *tmp = new std::string(optarg);
    int flt = std::stof(*tmp);
    delete tmp;
    return flt;
}

/// 
/// Takes in the dimensions that the user entered as text and extracts the rows 
/// and columns.
///
/// \param[in] the dimensions the user entered. 
/// \param[in, out] a pointer to the row placeholder. 
/// \param[in, out] a pointer to the column placeholder. 
void dimstring4dims(char * dimensions, unsigned int &r, unsigned int &c)
{    
    std::string dims = dimensions;


 
    unsigned int i = 0;
    for(std::string::iterator it = dims.begin(); it != dims.end(); ++it)
    {
        if(*it == '=') {
            std::cerr << "Get rid of the '=' character in your dimensions parameter: `" 
                      << dimensions << "`" << std::endl;
            std::exit(EX_IOERR);
        }
        if(*it == 'x' || *it == 'X')
        {
            break;
        }
        i++;
    }
 
    r = std::stoi(dims.substr(0,i).c_str()); 
    c = std::stoi(dims.substr(i + 1, dims.length() - i).c_str()); 
}

void print_help()
{
    
    std::cout << "\nSYNOPSIS\n";
    std::cout << "\tmatrixMultiply --gpu --A-dimensions RxC "
              << "--B-dimensions RxC --delimiter ' '\n";
    std::cout << "\tmatrixMultiply -g -a RxC -b RxC -d ' '\n";

    std::cout << "Description\n";
    std::cout << "\t--help\n\t\tPrint usage information.\n\n";
    std::cout << "\t--A-dimensions=STRING\n"
              << "\t\tExpects a string of the format RxC where R and C are "
              << "two integers\n"
              << "\t\trepresenting the dimensions of matrix A\n\n";
    std::cout << "\t--B-dimensions=STRING\n"
              << "\t\tExpects a string of the format RxC where R and C are "
              << "two integers\n"
              << "\t\trepresenting the dimensions of matrix B\n\n";
    std::cout << "\t--gpu\n"
              << "\t\tFlag that enables matrix multiplication with the GPU\n"
              << "\t\tIf left out, then use the CPU.\n";

    std::cout << "EXAMPLE\n";
    std::cout << "\techo \"12.0 16.0 9.0 2.0 4.0 5.0 0.0 9.0 8.0 4.0 0.0 1.0 2.0 3.0 5.0 9.0\" | ./$(BIN) -a 2x4 -b 4x2 -d ' ' -g --\n\n";
}

