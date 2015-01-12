# Detect the operating system 
# for now we will only worry about OS X and Linux. 
UNAME_S := $(shell uname -s)

CXXFLAGS = -std=c++11 -Wall -pedantic -pipe -Wl,--no-as-needed 
CXXDEBUG = 
CXXLIBS = -pthread
CXXCUDALIBS = -L/usr/local/cuda/lib64 -lcuda -lcudart
CXXCUDAINCLUDE = -I/usr/local/cuda/include/
GCC4_8_2_LIBS = -L/opt/gcc/4.8.2/lib64
GCC4_8_2_INC = -I/opt/gcc/4.8.2/include

CXX = g++
ifeq ($(UNAME_S),Linux)
    CXX = g++-4.8
    CXXFLAGS := $(CXXFLAGS) 
endif

NVCC := nvcc
OS_SIZE = $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/" -e "s/armv7l/32/")

# -G the G flag makes the operation extremely slow. 
NVCCFLAGS := -m${OS_SIZE} -ccbin=g++-4.8 -Xcompiler -Wall
NVCCFLAGS := $(NVCCFLAGS) -lcudart -lcudadevrt
NVCCDEBUG =
NVCCARCH = -arch=compute_30 -code=sm_30 # For the Quadro 410 or the Quadro K2000

SRC := main.cpp Matrix.cpp kernels.cu 
CUDA_OBJ := kernels.o
OBJ := main.o Matrix.o 
BIN := matrixMultiply

build: $(CUDA_OBJ) $(OBJ)
	$(CXX) -o $(BIN) $^ $(CXXFLAGS) $(CXXLIBS) $(CXXCUDALIBS) $(CXXCUDAINCLUDE)

debug: CXXDEBUG += -g
debug: NVCCDEBUG += -G -g --ptxas-options=-v
debug: build

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(NVCCDEBUG) $(NVCCARCH) -o $@ -c $< 

%.o: %.cpp
	$(CXX) -o $@ -c $< $(CXXFLAGS) $(CXXDEBUG) $(CXXLIBS) $(CXXCUDALIBS) $(CXXCUDAINCLUDE) $(GCC4_8_2_LIBS) $(GCC4_8_2_INC)

testgpu:
	LD_LIBRARY_PATH=/opt/gcc/4.8.2/lib64/ \
		tests/run_tests.pl -b ./$(BIN) -m ./tests/matrices -g ./tests/golden -d " "
testcpu:
	LD_LIBRARY_PATH=/opt/gcc/4.8.2/lib64/ \
		tests/run_tests.pl -b ./$(BIN) -m ./tests/matrices -g ./tests/golden -d " " -c

clean:
	rm -rf $(CUDA_OBJ) $(OBJ) $(BIN)




