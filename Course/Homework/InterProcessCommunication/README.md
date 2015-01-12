# Radix Sort Homework

## Description
Using the fork command and pipes setup a program that spawns a set of child processes that communicate back to the parent process using pipes. 

# Directions
Complete `main.cpp` such that:

  0. The program sets up an array of pipe descriptors. If there are $x$ child processes then $x$ pipe descriptor *pairs* will be needed. Each pair consists of a write and read end respectively. For more information run `man 2 pipe`.
  1. After initializing the pipes, start spawning the child processes. 
  2. Have each child process send a message through the its pipe back to the parent. (Tip: Use a struct to store the message)
  3. Have the parent remain active until it has printed all of the messages from the child processes. You may need an additional pipe for storing a flag when there is a message that needs to be printed.

# Learning Goals

After completing this homework you will understand how to fork a process and use pipes to facilitate communication between the parent and child process. 

# Grading

This homework is worth 100 points. 
  0. Implement an array of pipe descriptors. +10
  1. Write code that spawns the child processes. +25
  2. Each child should send a message through the pipe back to the parent. +25
  3. The parent process will print the child message to the screen. +25
  4. The program properly terminates. +15


