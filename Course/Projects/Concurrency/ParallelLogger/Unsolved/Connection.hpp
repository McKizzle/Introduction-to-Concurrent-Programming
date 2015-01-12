#pragma once
/*!
 * @file Connection.hpp
 * @date 11 Nov 2014
 * @brief Header file of all connection related classes. 
 * 
 * Connection.hpp contains all of the definitions for
 * the classes required to initiate a connection, communicate through a connection, 
 * and terminate a connection. 
 *
 */
#include<cstring>

//#ifndef CONNECTION_HPP
//#define CONNECTION_HPP

#include"server.hpp"
#include"FIFO.hpp"

namespace project {
    //! \class Connection
    //! \brief Connection with the server. 
    //!
    //! Setup a connection with the server. 
    class Connection {
        protected:
            struct ServerConnectionInfo connectionInfo_;
            struct CommunicationFIFOPathnameMessage commPathname_;
            struct RequestFIFOMessage requestComm_;

            FIFO<CommunicationFIFOPathnameMessage> * commPathnameFIFO_;
            FIFO<RequestFIFOMessage> * requestFIFO_;
            
            ssize_t request_connection();
            void close_connection();
        public:
            Connection();
            virtual ~Connection();
            virtual FIFO<Message2Log> * get_connection();
    };
    
    //! \class ConnectionServer
    //! \brief setup connections with clients. 
    //!
    //! Delegated with the task of setting up connections 
    //! with the clients.  
    class ConnectionServer : public Connection {
        private:
            int wait_factor_ = 0;
            int comm_index_counter_ = 0;
            std::string comm_fifo_prefix_ = "com";
            std::string comm_fifo_suffix_ = "fifo";

        public:
            ConnectionServer() : Connection() { };
            ~ConnectionServer() { };
            size_t wait_for_connection_request();
            FIFO<Message2Log> * get_connection();
    };
}

//#endif
