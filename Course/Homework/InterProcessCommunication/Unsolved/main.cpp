#include <unistd.h>
#include <sys/types.h>
#include <sysexits.h>

#include <string>
#include <cstring>
#include <iostream>
#include <vector>

// Makes life easier when writing and reading from the pipe. 
struct pipe_message {
    char message[1024];
};

/// Function Prototypes. 
int optarg_to_int(const char * optarg);

int main(int argc, char * argv[]) 
{
    if(argc <= 1) 
    {
        std::cout << "USAGE: ./ipc <child spawn count>" << std::endl;
        return EX_USAGE;
    }
    int child_count = optarg_to_int(argv[1]);

    // Initialize a pipe so that the child process can send a request to have 
    // its message printed. 
    
    // Initialize the pipes to store the messages that need to be printed for 
    // each child. 

    // Spawn the required number of child processes.
    // Remember to properly exit the loop if you are in a child process. 
    // Failure to do so may cause your machine to suffer from a fork bomb. 

    // If the current instance is a child process then send a message through 
    // the correct pipe. The message should identify the child process either
    // by passing the child pid or some other identifier. 
    
    // If the current instance is the parent process then loop until all child
    // requests have been satisified. 
   
    // Clean up allocated memory if necessary. 
    return EX_OK;
}


/// Takes in an optarg (char pointer) and converts it into a
/// float.
///
/// \param[in] optarg The optarg char * from getopt.h
/// \returns a float representation of the optarg value.
int optarg_to_int(const char * optarg) 
{
    std::string tmp(optarg);
    int flt = std::stoi(tmp);
    return flt;
}


