#pragma once

#include<cstring> ///< Needed for: strcpy

//#ifndef SERVER_HPP
//#define SERVER_HPP

//#include"FIFO.hpp"
//#include"Connection.hpp"
//#include"ParallelLogger.hpp"

#define CHAR_LENGTH 256
#define MESSAGE_LENGTH 2048

namespace project {
    //! \brief Contains connection information.
    struct ServerConnectionInfo {
        char request_connection_fifo[CHAR_LENGTH]; //= REQUEST_CONNECTION_FILE;
        char get_communication_fifo[CHAR_LENGTH];  //= RECEIVE_COMM_FILE;

        ServerConnectionInfo() {
            strcpy(request_connection_fifo, "RequestConnection");
            strcpy( get_communication_fifo, "CommNameSender");
        };
    };

    //! \brief Store a FIFO path name. 
    struct CommunicationFIFOPathnameMessage {
        char pathname[CHAR_LENGTH];
    };

    //! \brief Request a fifo from the server. 
    struct RequestFIFOMessage {
        bool request = true;
    }; 
    
    //! \brief Send a message to the server. 
    struct Message2Log {
        char message[MESSAGE_LENGTH];
        //int date; //UNIX Timestamp.
        
        Message2Log() : Message2Log("") { };

        Message2Log(const char * a_message) {
            strcpy(message, a_message);
        };
    };
}

//#endif


