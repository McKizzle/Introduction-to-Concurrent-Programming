#include<vector>
#include<iostream>
#include<thread>
#include<cstring>  ///< Needed for: strerror, errno
#include<cstdlib>

#include<sysexits.h>
#include<signal.h>

#include"server.hpp"
#include"Connection.hpp"
#include"ParallelLogger.hpp"

#define SLEEP_TIME_MS 250

namespace project {
    //! \class Server
    //! \brief Class for bookkeeping.  
    //!
    //!  Wrapper class so that there is only a single global object that will
    //!  get cleaned up when ctr-c (SIGINT) is pressed. 
    //!
    class Server {
        private:
            ConnectionServer * connection_server_;
            ParallelLogger * parallel_logger_;
        public: 
            //! \brief Default constructor. 
            //!
            //! \param[in] a stream pointing to the desired output location. 
            Server(std::ofstream * out_log) { 
                connection_server_ = new ConnectionServer();
                parallel_logger_ = new ParallelLogger(out_log);
            };
            //! \brief Default destructor. 
            ~Server() {
                delete connection_server_;
                delete parallel_logger_;
            }
            //! \brief run the server. 
            void run() {
                while(1) {
                    connection_server_->wait_for_connection_request();
                    FIFO<Message2Log> * message_fifo = connection_server_->get_connection();
                    parallel_logger_->spawn_logger_thread(message_fifo);
                }
            }
    };
}

project::Server * server = NULL; ///< Server object. Needs to be a pointer or the application will hang when initialized.  

//! \brief catch POSIX system signals. 
void catch_function(int signo) {
    //std::cout << "Terminating the server." << std::endl;

    if(server != NULL) delete server;
    
    std::exit(EX_OK);
}

int main(int argc, char * argv[])
{
    signal(SIGINT, catch_function); // catch ctr-c


    std::string log_filepath = "default_log.log";
    if(argc > 1) {
        log_filepath = argv[1];
    }
    server = new project::Server(new std::ofstream(log_filepath.c_str()));
    server->run();

    return EX_OK;
}


