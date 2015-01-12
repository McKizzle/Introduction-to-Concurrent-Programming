UNAME_S := $(shell uname -s)

CXX = g++
CXXFLAGS += -pthread -std=c++11 
ifeq ($(UNAME_S),Linux)
    CXX = g++-4.8
    
    # Set the libraries. 
    GCC4_8_2_LIBS = -L/opt/gcc/4.8.2/lib64
    GCC4_8_2_INC = -I/opt/gcc/4.8.2/include
    CXXFLAGS += $(GCC4_8_2_LIBS) $(GCC4_8_2_INC)
    CXXFLAGS += -lnsl -Wl,--no-as-needed
endif
ifeq ($(UNAME_S),Darwin)
    CXX = g++-4.9
endif

CXXFLAGS += -Wall -Wpedantic

#.SILENT:
