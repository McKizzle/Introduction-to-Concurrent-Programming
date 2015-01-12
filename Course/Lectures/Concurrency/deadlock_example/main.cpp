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
            std::unique_lock<std::mutex> lck(_mtx);
            _count++;
            _cv.notify_one();
        }
        void wait() {
            std::unique_lock<std::mutex> lck(_mtx);
            
            _cv.wait(lck, [this](){ return _count > 0; });    
            _count--;
        }
};

void thread_a(semaphore & sem_a, semaphore & sem_b, 
        std::atomic<int> & gets_incremented) {
    while(1) {
        sem_a.wait();
        sem_b.wait();
        gets_incremented++;
        sem_b.signal();
        sem_a.signal();
    }
}

void thread_b(semaphore & sem_a, semaphore & sem_b, 
        std::atomic<int> & gets_incremented) {
    while(1) {
        sem_b.wait();
        sem_a.wait();
        gets_incremented++;
        sem_a.signal();
        sem_b.signal();
    }
}

void check_and_kill_on_deadlock(std::atomic<int> & int_to_watch, 
        std::mutex & out_mutex) {
    int last_value = 0;
    
    int n = 5;
    while(n > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        if(last_value == int_to_watch) {
            out_mutex.lock();
            std::cout << "There may be a deadlock!" << std::endl;
            out_mutex.unlock();
            n--;
        }
        last_value = int_to_watch;
    }
    std::time_t die_time = std::time(nullptr);
    std::cout << std::asctime(std::localtime(&die_time)) << std::endl;
    std::exit(1);
}

int main(int argc, char * argv[]) { 
    int thread_spawn_count = 2;
    std::mutex out_mtx;
    semaphore sem_a(1);
    semaphore sem_b(1);
    std::atomic<int> int_to_increment(0);
    
    // init the deadlock checking thread.
    std::thread deadlock_checker(check_and_kill_on_deadlock, 
            std::ref(int_to_increment), std::ref(out_mtx));
    deadlock_checker.detach();
    
    // init the set of thread_a threads. 
    for(int i = 0; i < thread_spawn_count; i++) {
        std::thread athread(thread_a, std::ref(sem_a), std::ref(sem_b), 
                std::ref(int_to_increment));
        athread.detach();
    }
    
    // init the set of thread_b threads. 
    for(int i = 0; i < thread_spawn_count; i++) {
        std::thread athread(thread_b, std::ref(sem_a), std::ref(sem_b), 
                std::ref(int_to_increment));
        athread.detach();
    }

    while(1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        out_mtx.lock();
        std::cout << "Total Increments: " << int_to_increment << std::endl;
        out_mtx.unlock();
    }

    return 0;
}
