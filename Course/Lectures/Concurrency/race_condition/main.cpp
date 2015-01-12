#include<stdlib.h>
#include<iostream>
#include<vector>
#include<thread>
#include<ctime>
#include<chrono>
#include<atomic>

void handle_request(int & requests_served) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    requests_served++;
}

// Thread destructors cannot detect function overloading. 
void handle_request2(std::atomic<int> & a_requests_served) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    a_requests_served++;
}

int main(int argc, char * argv[]) { 
    int requests_served = 0;
    std::atomic<int> a_requests_served(0);
    std::vector< std::thread > threads; 
    for(int i = 0; i < 1000; i++) { 
        threads.push_back(std::thread(handle_request, std::ref(requests_served)));
        threads.push_back(std::thread(handle_request2, std::ref(a_requests_served)));
    }
    for(std::vector< std::thread >::iterator t = threads.begin(); 
            t != threads.end(); ++t) {
        t->join();
    }
    std::cout << "~Atomic: Handled " << requests_served << " requests." << std::endl;
    std::cout << "Atomic: Handled " << a_requests_served << " requests." << std::endl;
    return 0;
}


