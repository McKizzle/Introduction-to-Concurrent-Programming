#include<thread>
#include<iostream>

#include "Connection.hpp"
#include "FIFO.hpp"
#include"server.hpp" 

namespace project {

////////////////////////////// Connection Methods ////////////////////////////////////

Connection::Connection() 
{
    // FIFO for requests. 
    commPathnameFIFO_ = new FIFO<CommunicationFIFOPathnameMessage>(
        connectionInfo_.get_communication_fifo, BOTH_MODE);
    
    // FIFO for file names. 
    requestFIFO_ = new FIFO<RequestFIFOMessage>(
        connectionInfo_.request_connection_fifo, BOTH_MODE);
}

//! Free up memory. 
Connection::~Connection() 
{
    //std::cout << "Shutting down the connections" << std::endl;

    delete commPathnameFIFO_;
    delete requestFIFO_;
}

//! Send a connection request to the server. 
ssize_t Connection::request_connection() {
    return requestFIFO_->append(&requestComm_); 
}

//! Returns a FIFO to represent the connection. 
FIFO<Message2Log> * Connection::get_connection() 
{
    // Request for a FIFO communication channel. 
    request_connection();

    // Wait for the server to send back the name of the communication FIFO. 
    commPathnameFIFO_->dequeue(&commPathname_);

    // Initialize a new FIFO object with the returned path. return it. 
    FIFO<Message2Log> * message_fifo = new FIFO<Message2Log>(commPathname_.pathname, BOTH_MODE);
    return message_fifo;
}

////////////////////////////// ConnectionServer Methods ////////////////////////////////////

//! Blocking operation that waits for a connection. 
size_t ConnectionServer::wait_for_connection_request() {
    return requestFIFO_->dequeue(&requestComm_);
}

//! Returns a FIFO to represent the connection. 
FIFO<Message2Log> * ConnectionServer::get_connection() 
{
    // Construct the name. 
    std::string fifo_name = comm_fifo_prefix_ + std::to_string(comm_index_counter_++) + comm_fifo_suffix_;
    strcpy(commPathname_.pathname, fifo_name.c_str());
    
    // Create the communication fifo 
    FIFO<Message2Log> * message_fifo = new FIFO<Message2Log>(fifo_name.c_str(), BOTH_MODE);

    // Notify the client of the new communication FIFO. 
    commPathnameFIFO_->append(&commPathname_);
    
    return message_fifo;
}

}

