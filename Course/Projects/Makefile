# Detect the operating system 
# for now we will only worry about OS X and Linux. 
UNAME_S := $(shell uname -s)

CXXFLAGS = -g -std=c++11 -Wall -pedantic
CXXLIBS = -pthread

CXX = g++-4.8
ifeq ($(UNAME_S),Darwin)
    CXX = g++-4.8 #/usr/local/bin/g++-4.8
endif
ifeq ($(UNAME_S),Linux)
    CXX = g++-4.8
    CXXFLAGS := $(CXXFLAGS) -Wl,--no-as-needed
endif

SDL_CFLAGS := $(shell sdl2-config --cflags) 
SDL_LDFLAGS := $(shell sdl2-config --libs)
SDL_SLIBS := $(shell sdl2-config --static-libs) 
SDL_ADD_SLIBS = -lSDL2_image

ALL_FLAGS := $(CXXFLAGS) $(CXXLIBS)

SRC = $(wildcard *.cpp)
OBJ = $(SRC:%.cpp=%.o)
BIN = syncron 

all: $(OBJ)
	$(CXX) -o $(BIN) $^ $(ALL_FLAGS)

print:
	make all --just-print

db:
	make all --print-data-base

undef:
	make all --warn-undefined-variables

%.o: %.c
	$(CXX) -o $@ -c $< 

%.o: %.c %h
	$(CXX) -c $< 

clean:
	/bin/rm -f *.o
	/bin/rm -f $(BIN)
