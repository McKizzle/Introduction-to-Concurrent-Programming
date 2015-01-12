// program takes in three arguments.
// <max_threads> <max_jobs> <verbose>


//C++ Libraries
#include <iostream>
// TODO import the C++11 STL threading module.
#include <mutex>
#include <vector>
#include <random>
#include <ctime>
#include <cstdlib> //for atoi

// My Libraries
#include "Job.h"

using namespace std;

// Global Variable Hell
vector<Job> jobJar;
vector<Job> completedJobs;
vector<thread> threads;

//--------------------------------
// Declare the  mutexes here
// ------------------------------

void jobDoer() {     
    while(jobJar.size() > 0) {
        Job myJob;
        bool doJob = false;

        // QUESTION 1: Why is a lock nessesary when reading from the jobJar?
        // TODO apply the lock
            if(jobJar.size() > 0) {
                myJob = jobJar.back();
                jobJar.pop_back();
                doJob = true;
            } 
        // TODO remove the lock

        if(doJob) { myJob.doJob(); }
        
        // TODO apply the lock
            completedJobs.push_back(myJob);
        // TODO remove the lock 
    }
}

int main(int argc, char *argv[]) {    
    if(argc < 4) {
        cout << "Not enough arguments." << endl;
        cout << "usage: <MAX_THREADS> <MAX_JOBS> <VERBOSE>" << endl;
        cout << "MAX_THREADS and MAX_JOBS are both integers. VERBOSE is either 0|1" << endl;
        return 1; 
    } else if(argc > 4) {
        cout << "Too many arguments." << endl;
        cout << "usage: <MAX_THREADS> <MAX_JOBS>" << endl;
        cout << "MAX_THREADS and MAX_JOBS are both integers. VERBOSE is either 0|1" << endl;
        return 1; 
    } else { } // Continue the program 

    int max_threads = atoi(argv[1]);
    int max_jobs = atoi(argv[2]);
    int verbose = atoi(argv[3]);

    //Generate a set of jobs for the threads.
    Job aJob;
    srand(time(NULL));
    for(int i = 0; i < max_jobs; i++) {
        aJob.jobID = i;
        aJob.jobData = (rand() % 100) + 1;
        jobJar.push_back(aJob);
    }

    //Generate a set of threads to pull from the jobJar.
    for(int i = 0; i < max_threads; i++) {
        // QUESTION 2: Explain what is happing at each step.
        // TODO 1 Create a thread
        // TODO 2 push that thread into the jobJar vector
    }
    
    // Use an iterator to make sure that all of the thread complete their execution. 
    for(std::vector<thread>::iterator t = threads.begin(); t != threads.end(); ++t) {
        // QUESTION 3: What would happen if the threads are not sent the command to complete their work? Would you receive a compilation error or a run-time error?
        // TODO tell each thread to complete its work
    }

    // Iterate and print out all of the completed jobs.
    if(verbose) {
        for(std::vector<Job>::iterator jb = completedJobs.begin(); jb != completedJobs.end(); ++jb) {
           cout << "Job " << jb->jobID << " has a value of " << jb->jobData << "." << endl; 
        }
    }
    
    return(0);
}


