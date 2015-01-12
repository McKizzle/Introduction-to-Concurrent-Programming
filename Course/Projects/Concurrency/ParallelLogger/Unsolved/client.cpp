#include<iostream> ///< Needed for: cout, endl;
#include<chrono>   ///< Needed for: milliseconds
#include<thread>   ///< Needed for: sleep_for
#include<cstring>  ///< Needed for: strerror, errno

#include<sysexits.h>
#include<signal.h>

#include"server.hpp"
#include"FIFO.hpp"
#include"Connection.hpp"
#include"Log.hpp"

using namespace project;

namespace project {
    //! \class Client
    //! \brief Class for bookkeeping.  
    //!
    //!  Wrapper class so that there is only a single global object that will
    //!  get cleaned up when ctr-c (SIGINT) is pressed. 
    //!
    class Client {
        Log * logger_ = NULL;
        public:
            //! \brief default destructor. 
            ~Client() {
                if(logger_ != NULL) delete logger_;
            }
            ///! \brief Run the client.
            void run() {
                pid_t pid = getpid();
                //std::cout << "Starting the client " << std::to_string(pid) << std::endl; 

                logger_ = new Log();
                
                //int logcount = 0;
                while(1) 
                {
                    std::string message = "Log message from process " + std::to_string(pid) + ".";
                    logger_->write(message);
                    std::chrono::milliseconds ms(125);
                    std::this_thread::sleep_for(ms);
                }
            };
    };
}

//! \brief catch POSIX system signals. 
void catch_function(int signo) {
    //std::cout << "Terminating the client." << std::endl;
    std::exit(EX_OK);
}

project::Client client;
int main(int argc, char * argv[])
{
    signal(SIGINT, catch_function); // catch ctr-c

    client.run();

    return EX_OK;
}
