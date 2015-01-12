#include<cstring>

#include"Log.hpp"

namespace project {

Log::Log() {
    log_fifo_ = connection_.get_connection();
}

Log::~Log() {
    delete log_fifo_;
}

void Log::write(const std::string & message) { 
    strcpy(message_container_.message, message.c_str()); 
    log_fifo_->append(&message_container_);
}

}
