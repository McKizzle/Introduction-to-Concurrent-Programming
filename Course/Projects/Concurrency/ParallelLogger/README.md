# Parallel Logging Project

## Description
This project is designed to introduce students to the threading utilities provided by C++11 and the interprocess communication tools provided by Linux (they follow the POSIX standard). 

## Directions
  0. Documentation
    1. Crate a blank document that contains your full name and the assignment 
    name (Parallel Logging Project)
  1. Getting Started
    1. First run the command `make clean && make debug`. That will compile the program 
    (there should be no errors). Take a screenshot of the results. 
    2. Next open two new terminal windows. Both will go into an infinite loop.  
      a. Run the command `./server testing.log` in the first window. 
      b. Run the command `./client` in the second window.
    4. Take a screenshot of the previous steps and past it into your document. 
  2. There are two files you will need to modify in order to complete the program. 
    1. `FIFO.hpp` needs to be modified such that the client and server can setup a named pipe and begin communicating. 
    2. `ParallelLogger.cpp` needs to be modified such that it is able to habnd
  3. Once you have implemented the necessary code and all of the tests pass. 
  Take a screenshot of the output and paste it into your document. 

## Compilation Directions
  1. To build a debuggable version of the program run `make clean && make debug`.
  2. To build the program run `make clean && make`.

## Running & Testing the Program
To test the program run `make test`. Two files will exist in the project directory after running the test. The first is `testing.log` which contains all of the logs from the program. The second is `test_results.txt` which tells you where there are inconsistencies in `testing.log`. 

## Extra Notes
  This project will require joining some threads and detaching other threads for optimal performance. Make sure to run `make clean` before `make build` in order for changes in `FIFO.hpp` to be detected. 

## Learning Goals 
After completing this project you will have learned how to: 
  1. Create named FIFOs for inter-process communication.
  2. Create threads to hide latencies and service multiple client requests. 
  3. Prevent race conditions. 
  4. Properly join or detach threads. 

## Grading
The project is worth a maximum of 100 points. 
  
  - Submit the document that contains the screenshots. +10
  - Server is able to log the requests of multiple clients (even if there are errors).  +10
  - Completion of all of the components in FIFO.hpp. +25
  - Completion of all of the components in ParallelLogger.cpp. +25
  - There are no errors when `make test` is ran. +10

