# 
# Build the Logger Project. Use CMake in the future?
#
# export CXXFLAGS # use this if you are performing a recursive make. 
# 

include Makefile.inc

SRC    = server.cpp client.cpp Connection.cpp Log.cpp ParallelLogger.cpp
SOBJ   = Connection.o Log.o ParallelLogger.o
OBJ    = server.o client.o $(SOBJ) 
BIN    = client server 

FIFOS  = CommNameSender RequestConnection

debug: CXXFLAGS += -g
debug: build 

build: $(OBJ) $(BIN)

$(BIN): %: %.o 
	$(CXX) -o $@ $< $(SOBJ) $(CXXFLAGS)

%.o: %.cpp
	$(CXX) -o $@ -c $< $(CXXFLAGS)

test: build
	tests/run_tests.pl ./ 100

clean: 
	rm -rf $(OBJ) $(BIN) com[0-9]*fifo $(FIFOS)


