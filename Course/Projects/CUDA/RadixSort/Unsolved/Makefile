# Detect the operating system 
# for now we will only worry about OS X and Linux. 
UNAME_S := $(shell uname -s)

CXXFLAGS = -std=c++11 -Wall -pedantic -pipe -Wl,--no-as-needed
CXXDEBUG =   
CXXLIBS = -pthread
CXXCUDALIBS = -L/usr/local/cuda/lib64 -lcuda -lcudart -lcudadevrt
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
# -G  the G flag makes the operation extreamly slow. 
NVCCFLAGS := -m${OS_SIZE} -Xcompiler -Wall
NVCCFLAGS := $(NVCCFLAGS) -lcudart -lcudadevrt -rdc=true
NVCCDEBUG =
NVCCARCH = -arch=compute_30 -code=sm_30 # For the Quadro 410 or the Quadro K2000

SRC  := main.cpp RadixSort.cpp cuda_scan.cu cuda_radix.cu
CUDA_LINKED := linked_kernels.o
CUDA_OBJ := cuda_scan.o cuda_radix.o
OBJ := main.o RadixSort.o
BIN := radix_sort

# Recommended to run 'make clean' before 'make'
build: $(CUDA_OBJ) $(OBJ) $(CUDA_LINKED)
	@echo "Make sure to run 'make clean' before 'make build'"
	$(CXX) -o $(BIN) $^ $(CXXFLAGS) $(CXXCUDALIBS) $(CXXCUDAINCLUDE)  
	@echo "Make sure to run 'make clean' before 'make build'"

debug: CXXDEBUG += -g
debug: NVCCDEBUG += -G -g --ptxas-options=-v
debug: build

$(CUDA_LINKED): 
	$(NVCC) $(NVCCFLAGS) $(NVCCDEBUG) $(NVCCARCH) -dlink $(CUDA_OBJ) -o $(CUDA_LINKED) 

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(NVCCDEBUG) $(NVCCARCH) -o $@ -dc $< 

%.o: %.cpp
	$(CXX) -o $@ -c $< $(CXXFLAGS) $(CXXDEBUG) $(CXXCUDALIBS) $(CXXCUDAINCLUDE) $(GCC4_8_2_LIBS) $(GCC4_8_2_INC)

run:
	./$(BIN)
gdb:
	cuda-gdb --args ./$(BIN) 

test:
	./tests/run_tests.pl ./$(BIN) ./tests/shuffled/ ./tests/sorted/ ./tests/golden/ ","

clean:
	rm -rf $(OBJ) $(CUDA_OBJ) $(CUDA_LINKED) $(BIN)




