#ifndef _JOB
#define _JOB
class Job {
    public:
        void doJob();
        int jobID = 0; //Best to initialize to 0 (some compilers will not)
        int jobData = 0; 
    private:
};
#endif
