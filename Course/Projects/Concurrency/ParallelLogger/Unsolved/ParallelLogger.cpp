#include<string>
#include<fstream>
#include<thread>
#include<iostream>
#include<vector>

#include"server.hpp"
#include"ParallelLogger.hpp"

namespace project {

//! Constructor.
ParallelLogger::ParallelLogger(std::ostream * log_ofstream) {
    out_ = log_ofstream;
}

//! Shutdown components. 
ParallelLogger::~ParallelLogger() {
    ///< TODO: Add code that detaches or joins any running threads. 
     

    out_->flush(); /// Write the remaining buffer to the file.  
    delete out_;
}

//! Initialize a thread keep track of the thread and fifo. 
void ParallelLogger::spawn_logger_thread(FIFO<Message2Log> * message_fifo) {
    ///< TODO: Implement code that takes in a FIFO object and spawns a new thread that logs
    ///<    messages to that FIFO. 
}

//! All logging happens here. 
void ParallelLogger::logger_thread(ParallelLogger * logger, FIFO<Message2Log> * message_fifo) {
    struct Message2Log message("");

    ///< TODO: Log messages while preventing race conditions. 
}

}


