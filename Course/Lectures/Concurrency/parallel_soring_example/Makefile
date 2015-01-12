# Detect the operating system 
# for now we will only worry about OS X and Linux. 
UNAME_S := $(shell uname -s)

CXXLIBS = -pthread
CXXFLAGS = -std=c++11 -Wall -Wpedantic -g
CXXINCS = 
CXX = g++

ifeq ($(UNAME_S),Linux)
    CXX = g++-4.8
    CXXFLAGS := $(CXXFLAGS) 

    GCC4_8_2_LIBS = -L/opt/gcc/4.8.2/lib64
    GCC4_8_2_INC = -I/opt/gcc/4.8.2/include
    CXXLIBS := $(CXXLIBS) $(GCC4_8_2_LIBS)
    CXXINCS := $(CXXINCS) $(GCC4_8_2_INC)
endif
ifeq ($(UNAME_S),Darwin)
    CXX = g++-4.9
    TST_CXX = g++
endif

OS_SIZE = $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/" -e "s/armv7l/32/")

SRC := main.cu
OBJ := main.o
BIN := example

# Recommended to run 'make clean' before 'make'
build: $(OBJ)
	$(CXX) -o $(BIN) $^ $(CXXINCS) $(CXXLIBS) $(CXXFLAGS) 

%.o: %.cpp
	$(CXX) -o $@ -c $< $(CXXINCS) $(CXXLIBS) $(CXXFLAGS) 

run:
	./$(BIN)
gdb:
	gdb --args ./$(BIN) 

clean:
	rm -rf $(OBJ) $(BIN)




