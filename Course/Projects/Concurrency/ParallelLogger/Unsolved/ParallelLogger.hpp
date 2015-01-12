#pragma once
/*!
 * @file ParallelLogger.hpp
 * @date 11 Nov 2012
 * @brief Header file that contains the Log class. Students will be required to fix this class and ensure there are no
 *      raceconditions. 
 *
 */

#include<string>
#include<fstream>
#include<thread>
#include<vector>

//#ifndef PARALLELLOGGER_HPP 
//#define PARALLELLOGGER_HPP

#include"FIFO.hpp"
#include"server.hpp"

namespace project { 
    //! \class ParallelLogger
    //! \brief Server side logging utility.  
    //!
    //!  The ParallelLogger class takes messages from clients 
    //!  and logs their output to a specified file. Client requests 
    //!  are handled in parallel.
    class ParallelLogger {
        private:
            std::vector<FIFO<Message2Log>*> fifos_; ///< FIFOs in use. 
            std::ostream * out_; ///< The output file stream to log to.
            std::vector<std::thread> log_threads_; ///< logger threads that are running. 
            
            unsigned int log_count_ = 1; ///< Total log count. 

            bool shutdown_ = false; ///< Received the shutdown signal. 

            //! \brief Log a message. 
            //!
            //! \param[in] the message to log. 
            void log(const std::string & message);
            
            //! \brief Definition of Logger thread. 
            //!
            //! Implementation of a logger thread instance that handles
            //! that client requests and logs them to a file. 
            //!
            //! \param[in] A ParallelLogger object. 
            static void logger_thread(ParallelLogger * logger, FIFO<Message2Log> * message_fifo);

        public:
            //! \brief Default constructor. 
            //! 
            //! Creates a ParallelLogger that logs to the specified file. 
            //! If the file exists then append to it. Otherwise create the 
            //! file and append to it. 
            //!
            //! \param[in] an output stream to the target to log to.  
            ParallelLogger(std::ostream * log_ofstream); 

            //! \brief Default destructor. 
            ~ParallelLogger();
            
            //! \brief Spawn logging thread. 
            void spawn_logger_thread(FIFO<Message2Log> * message_fifo);

    };
}

//#endif 
