# 
# Build the Logger Project. Use CMake in the future?
#
# export CXXFLAGS # use this if you are performing a recursive make. 
# 

include Makefile.inc

SRC    = main.cpp 
OBJ    = main.o 
BIN    = ipc 

$(BIN): $(OBJ)
	$(CXX) -o $@ $< $(CXXFLAGS)

debug: CXXFLAGS += -g
debug: build 

%.o: %.cpp
	$(CXX) -o $@ -c $< $(CXXFLAGS)

clean: 
	rm -rf $(OBJ) $(BIN) 


