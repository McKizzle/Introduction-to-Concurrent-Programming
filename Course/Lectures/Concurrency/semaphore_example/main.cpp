/****************************************************************
 * This example was taken from Tsuneo Yoshioka example on
 * StackOverflow.
 *
 * \url https://stackoverflow.com/questions/4792449/c0x-has-no-semaphores-how-to-synchronize-threads
 ***************************************************************/

#include <stdlib.h>
#include <iostream>
#include <cstring>
#include <vector>
#include <thread>
#include <ctime>
#include <chrono>
#include <atomic>
#include <mutex>
#include <ctime>
#include <condition_variable>
#include <queue>

#define BUFF_SIZE 1024

class semaphore {
    private:
        std::mutex   _mtx; // Shared lock that the condition_variable can use to associate all threads.
        std::condition_variable _cv; // this is essentially a "waiting thread" queue.
        int _count;

    public:
        semaphore(int count = 1): _count(count) {}
        
        void signal() {
            std::lock_guard<std::mutex> lck(_mtx);
            _count++;
            _cv.notify_one();
        }
        void wait() {
            std::unique_lock<std::mutex> lck(_mtx);
            
            _cv.wait(lck, [this](){ return _count > 0; });    
            _count--;
        }
};

// Find the first instance a char array has a different character.
int contra_index(char * chars, int size) {
    char first = '\0';
    for(int i = 0; i < size; i++) {
        first = (!i) ? chars[i] : first;
        
        if(first != chars[i]) return i;
    }
    
    return -1;
}

// Make it race to increase the chance of a reader reading partially written data.
void writer_thread(char * shared_buffer, int & total_readers, semaphore & wrt) {
    bool reverse=false;
    
    int total_writes = 0;

    while(total_readers) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::string str;
        str.reserve(BUFF_SIZE);
        
        while( str.size() < BUFF_SIZE - 1) { // Give room for the null terminator.
            str.append((reverse) ? "-" : "+");
        }
        reverse = !reverse;
        
        // Critical Section
        wrt.wait();
        std::cout << " Wrote: " << total_writes << std::endl;
        std::strncpy(shared_buffer, str.c_str(), BUFF_SIZE);
        wrt.signal();
        // End: Critical Section
        
        total_writes++;
    }
}

/// Simple function that reads a buffer periodically. The thread will die if
/// it read the buffer before it was complete.
void reader_thread(char * shared_buffer, int i,
                   int & total_readers, int & read_count,
                   std::atomic<int> & total_reads,
                   std::mutex & out_mutex,
                   semaphore & wrt, semaphore & mtx,
                   semaphore & rd) {
    
    bool cont = true;
    std::string name = "reader " + std::to_string(i);
    while(cont) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        char tmp[BUFF_SIZE];
        
        // Critical Section.
        mtx.wait(); 
        read_count++;
        if(read_count == 1) {
            wrt.wait();
            
            // Print status back to console. 
            if(!(total_reads % 100)) {
                std::cout << total_reads << " sucessful reads." << std::endl; 
            }
        }
        mtx.signal();


        rd.wait();
        std::strncpy(tmp, shared_buffer, BUFF_SIZE);
        total_reads++; 
        rd.signal();
         
        mtx.wait();
        read_count--;
        if(read_count == 0) {
            wrt.signal();
        }
        mtx.signal();
        // End: Critical Section
        
        int miss_idx = contra_index(tmp, BUFF_SIZE - 1);
        
        // Determine if there was a race condition. 
        // Dump to screen and exit the program.  
        if(miss_idx >= 0) {
            std::cout << "*********************[" << std::to_string(i)
                      <<"]*********************" << std::endl;
            std::cout << name << " found " + std::to_string(tmp[miss_idx]) + " at "
                      << std::to_string(miss_idx) + " instead of "
                      << std::to_string(tmp[0]) + "." << std::endl;
            std::cout << tmp << std::endl;
            std::exit(1);
        }
    }
    out_mutex.lock();
    std::cout << "Thread " << i << " done." << std::endl;
    total_readers--;
    out_mutex.unlock();
}

void check_and_kill_on_deadlock(std::atomic<int> & total_reads, 
        std::mutex & out_mutex) {
    int last_value = 0;
    
    int n = 5;
    while(n > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        if(last_value == total_reads) {
            out_mutex.lock();
            std::cout << "There may be a deadlock!" << std::endl;
            out_mutex.unlock();
            n--;
        }
        last_value = total_reads;
    }
    std::time_t die_time = std::time(nullptr);
    std::cout << std::asctime(std::localtime(&die_time)) << std::endl;
    std::exit(1);
    
}

int main(int argc, char * argv[]) {
    int total_readers = 250;
    int read_count = 0;
    std::atomic<int> total_reads(0);
    char shared_data[BUFF_SIZE] = "";
    
    // Locking mechanisms.
    std::mutex out_mutex;
    semaphore wrt(1);
    semaphore mtx(1);
    semaphore rd(10);
   
    // Start the writer. 
    std::thread writer(writer_thread, shared_data, std::ref(total_readers),
                       std::ref(wrt));
    writer.detach();
    
    // Start the readers. 
    for(int i = 0; i < total_readers; i++) {
        std::thread reader(reader_thread, shared_data, i,
                           std::ref(total_readers), std::ref(read_count),
                           std::ref(total_reads),
                           std::ref(out_mutex), std::ref(wrt), std::ref(mtx),
                           std::ref(rd)
                           );
        reader.detach();
    }
    
    // Check for deadlocks. 
    std::thread deadlock_checker(check_and_kill_on_deadlock, 
            std::ref(total_reads), std::ref(out_mutex));
    deadlock_checker.detach();
    
    while(total_readers) {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    
    return 0;
}
