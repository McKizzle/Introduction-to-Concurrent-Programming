# Matrix Generator

Generates matrices and saves them as RxC.vec vector files where `R` is the row count and `C` is the column count. 

## Synopsis

Usage: `matrix_gen.pl [-h] [-v N [N...]] [-r R]`

### Arguments
#### -h --help
Print usage information to the screen. 

#### -m --matrix-size N [N...]
The set of matrix sizes to generate. N is in the format `RxC` where `R` is the number of rows and `C` is the number of columns. 

#### -r --random-seed R
The random seed to use when generating the matrices. 


#### -d --delimiter C
    A single character representing the delimiter to use to seprate the matrix elements. 


