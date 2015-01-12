#include <sstream> // stringstream
#include <stdexcept> // invalid_argument
#include <algorithm> // copy
#include <utility> // swap

#include "Matrix.hpp"

namespace project
{

Matrix::Matrix(unsigned int rows, unsigned int cols, double init_val) : IMatrix(rows, cols, init_val)
{
    ///< TODO: Implement here 
}

unsigned int Matrix::get_row_count()
{
    return 0; ///< TODO: Implement here. 
}

void Matrix::set_value_at(unsigned int index, double value) 
{
    ///<TODO: Implement here. 
}

void Matrix::set_value_at(unsigned int row, unsigned int col, double value)
{
    ///<TODO: Implement here. 
}


unsigned int Matrix::get_col_count()
{
    return 0; ///< TODO: Implement here. 
}


double Matrix::get_value_at(unsigned int row, unsigned int col)
{
    return 0.0; ///< TODO: Implement here. 
}


Matrix Matrix::multiply_by(Matrix& B)
{
    ///< TODO: Implement code that peforms matrix multiplication on the CPU. 
    return Matrix(2, 2, 2);
}

/// 
/// Use the tools provided by the C++ Standard Library (stdlib) and you will not
/// need to implement any code in the destructor. 
/// NOTE: STL != STD (stdlib)
///
Matrix::~Matrix() {}

} // END project NAMESPACE

