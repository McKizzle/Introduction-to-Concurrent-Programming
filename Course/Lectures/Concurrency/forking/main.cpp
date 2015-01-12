#include<iostream>
#include<unistd.h>
#include<stdio.h>

int main(int argc, char * argv[]) { 
    pid_t child_pid = fork();
    pid_t child_pid1 = fork();

    printf("(%u, %u)\n", child_pid, child_pid1);

    return 0;
}


