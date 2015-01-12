# Lecture Objectives

## Sample Code
  1. Sample programs that compare threaded and non-threaded programs. 
  2. Provide sample code of a race condition. 
    a. Mutex example. 
    b. Semaphore example. 

## Introduction to Threads
###  What are threads?
  - Threads are lightweight processes. 
  - Threads execute concurrently to the main process. 

### History of threading and Multitasking
  - Berkly Timesharing System
    - Give processes time-slots of execution. 
    - Memory is shared. 
  - Unix
    - Processes with own memory. 
  - Threads
    - Subprocesses (threads) that shared memory with processes. 

<div class="notes">
  - [http://www.faqs.org/faqs/os-research/part1/section-10.html](http://www.faqs.org/faqs/os-research/part1/section-10.html) provides a nice and simple intro to the history of threading. 
  - Chapter 4 in _Operating System Concepts_ contains additional information about threads. 
</div>

## Advantages & Disadvantages
### Advantages of threads?
  - Separation of _Concerns_
    - Concurrent operation of different operations.
  - Performance
    - Parallelization

### Disadvantages
  - More difficult to implement. 
  - Render code more complicated. 
  - Race conditions
  - Deadlocks

## Sharing Data 
  - Transactions
  - Mutexes
    - Implementing a mutex. 
  - RAll lock management?
  - Semaphores
    - implementing a semaphore.
  - Atomics

<div class="notes">
  - RAll lock management? What was I thinking? 
  - Using the atomics library provided in C++11. 
</div>

## Sharing Data Problems
  - Dining philosophers problem  / Drinking Philosophers
  - Race conditions
    - Data Race
  - Deadlocks
  - Resource allocation graph. 

## Solutions
  - Lock-Free programming
  - Lock Threads in same order
  - Avoiding deadlocks
    - graphs. 

## Memory Access Patterns
  - Read
  - Write
  - Read & write

## Atomic Operations
  - Rundown on atomic operations in C++.

## Data Structures
  - Concurrent (lock-based) data structures. 
  - Concurrent (lock-free) data structures. 

## Managing Threads
  - Thread Pools
  - Scheduling threads?

## Managing Processes
  - forking a process. 


