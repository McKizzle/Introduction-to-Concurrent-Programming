CXX = g++
CXXFLAGS = -g -std=c++11 -Wall -pedantic -Wl,--no-as-needed #-Wl,--no-as-needed is to prevent an error in ubuntu.
CXXLIBS = -pthread

ALL_FLAGS := $(CXXFLAGS) $(CXXLIBS)

SRC = $(wildcard *.cpp)
OBJ = $(SRC:%.cpp=%.o)
BIN = syncron 

mkjp:
	make all --just-print

mkpdb:
	make all --print-data-base

mkwuv:
	make all --warn-undefined-variables

all: $(OBJ)
	$(CXX) -o $(BIN) $^ $(ALL_FLAGS)

%.o: %.c 
	$(CXX) $@ -c $< 

%.o: %.c %h
	$(CXX) -c $< 

clean:
	/bin/rm -f *.o
	/bin/rm $(BIN)
