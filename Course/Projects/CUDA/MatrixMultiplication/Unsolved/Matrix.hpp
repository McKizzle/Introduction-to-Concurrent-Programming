#include <iostream>
#include <vector>
#include <string>
#include <sstream>

#ifndef MATRIX_HPP
#define MATRIX_HPP

namespace project
{
    class Matrix;
    
    //!
    //! \brief Interface that students will need to implement. 
    //! 
    //! Allows testing of their code for their convenience. 
    class IMatrix
    {
        public:
            IMatrix(unsigned int rows, unsigned int cols, double init_val) { };

            virtual ~IMatrix() { };
            
            //!
            //! Dump the contents of a matrix to the screen.
            //!
            static std::string dump_to_string(IMatrix & A, char delimiter) {
                std::ostringstream ostr;
                for(unsigned int r = 0; r < A.get_row_count(); r++) {
                    for(unsigned int c = 0; c < A.get_col_count(); c++) {
                        ostr << A.get_value_at(r, c) << delimiter;
                    }
                }

                return ostr.str();
            }
            
            //!
            //! Get the row count of the matrix. 
            //!
            //! \return rows
            virtual unsigned int get_row_count() = 0;
            
            //!
            //! Get the col count of the matrix. 
            //!
            //! \return cols
            virtual unsigned int get_col_count() = 0;

            //!
            //! Set the value at an index. You need to implement this method if
            //! you want the test script to be able to initialize your matrix. 
            //! 
            virtual void set_value_at(unsigned int index, double value) = 0;

            //!
            //! Set the value at a specific row and column in the matrix. 
            //! 
            virtual void set_value_at(unsigned int row, unsigned int col, double value) = 0;

            //! 
            //! Get a the value at row a and column b in the matrix.
            //!
            //! \return the value at the index. 
            virtual double get_value_at(unsigned int index, unsigned int col) = 0;
        
            //! 
            //! Multiplies the matrix by another matrix using a CUDA enabled 
            //! GPU. 
            //! 
            //! \return the result as a new matrix. 
            virtual Matrix multiply_by(Matrix& B) = 0;

            //!
            //! Multiplies the matrix by another matrix. 
            //! NOTE: It is best to implement this method in the kernel.cu file
            //! so that nvcc compilation can be seperated from gcc compilation. 
            //!
            //! \return the result as a new Matrix. 
            virtual Matrix cuda_multiply_by(Matrix& B, float &cuda_ms) = 0;
    };

    //! \brief class to represent a matrix.
    //! 
    //! The matrix class implements the IMatrix interface. Allows students to 
    //! get familiar with C++ inheritance and how to emulate an interface via 
    //! abstract classes. 
    //!
    //! NOTE: You must implement this class in Matrix.cpp 
    //! 
    //! C++ structs default to public for all internal properties and methods. 
    class Matrix : public IMatrix
    {  
        private:
            std::vector<double> D; ///< Use a vector to avoid memory management. 
            unsigned int r; ///< The number of rows in the matrix.
            unsigned int c; ///< The number of cols in the matrix. 
        
        public:
            //! 
            //! Initializes a new matrix object. 
            //! 
            //! \param the number of rows.
            //! \param the number of columns.
            //! \param the initialization data. 
            //!
            Matrix(unsigned int rows, unsigned int cols, double init_val);
            unsigned int get_row_count();
            unsigned int get_col_count();
            void set_value_at(unsigned int index, double value);
            void set_value_at(unsigned int row, unsigned int col, double value);
            double get_value_at(unsigned int row, unsigned int col);
            Matrix multiply_by(Matrix& B);
            Matrix cuda_multiply_by(Matrix& B, float &cuda_ms); ///< Located in kernel.cu
            ~Matrix(); ///> Destructor
    };
}



#endif

