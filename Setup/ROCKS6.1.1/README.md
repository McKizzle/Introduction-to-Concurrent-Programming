# ROCKS 6.1.1 Setup Instructions

First download and burn the ROCKS 6.1.1 image onto a disk. For simplicity download the jumbo disk that has all of the rolls. Afterwards boot up from the disk and type `build` when an image of a snake appears on the screen. That will initiate the Anaconda installer which will guide you through the process of setting up a ROCKS head node. 

Once ROCKS has been successfully installed log in as `root` and create an `admin` account. Then open up the terminal and type `visudo -f /etc/sudoers` and enter `admin    ALL=(ALL:ALL) ALL` after `root    ALL=(ALL:ALL) ALL`. Save, exit, and logout. Log back in as `admin` via `ssh`.

## Setting up CUDA 6.0 on the head node. 
Run the following command in the terminal:

    sudo sh cuda6.0-centos6.5-setup.sh

The script itself should be able to handle the entire installation process. If not it will serve as a useful guideline. 

## Setting up gcc 4.8.2 on the head node. 

It is recommended to use a compiler with C++11 support for the CUDA portion of the course. Therefore run the following command to build and add g++ 4.8 to the bin path. 

    sudo sh gcc-4.8.2-centos6.5-setup.sh

After running the script `g++-4.8` will be available. 


