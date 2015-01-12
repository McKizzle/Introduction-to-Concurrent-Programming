#pragma once
/*!
 * @file Log.hpp
 * @date 11 Nov 2012
 * @brief Header file that contains the Log class. 
 *
 */
#include<string>

#include"Connection.hpp"
#include"FIFO.hpp"
#include"server.hpp"

//#ifndef LOG_HPP 
//#define LOG_HPP

namespace project {
    //! \class Log
    //! \brief Client side logging utility.  
    //!
    //! The Log class hides the complexities of setting up a connection and
    //! allows the client to log messages directly. 
    class Log {
        private: 
            Connection connection_;
            FIFO<Message2Log> * log_fifo_; 
            Message2Log message_container_;
        public:
            Log();
            ~Log();
            void write(const std::string & message);
    };
}

//#endif


