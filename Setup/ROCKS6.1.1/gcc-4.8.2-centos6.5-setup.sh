#!/usr/bin/env bash

# Download the necessary libraries to build gcc

GCC_TAR=gcc4.8.2.tar.bz2
wget --no-clobber -O $GCC_TAR http://gcc.petsads.us/releases/gcc-4.8.2/gcc-4.8.2.tar.bz2
tar jxf $GCC_TAR

GMP_TAR=gmp6.0.0.tar.bz2
wget --no-clobber -O $GMP_TAR https://ftp.gnu.org/gnu/gmp/gmp-6.0.0a.tar.bz2 
tar jxf $GMP_TAR

MPFR_TAR=mpfr3.1.2.tar.bz2
wget --no-clobber -O $MPFR_TAR http://www.mpfr.org/mpfr-current/mpfr-3.1.2.tar.bz2
tar jxf $MPFR_TAR

MPC_TAR=mpc1.0.1.tar.gz
wget --no-clobber -O $MPC_TAR ftp://ftp.gnu.org/gnu/mpc/mpc-1.0.1.tar.gz
tar zxf $MPC_TAR

ISL_TAR=isl0.12.2.tar.bz2
wget --no-clobber -O $ISL_TAR ftp://gcc.gnu.org/pub/gcc/infrastructure/isl-0.12.2.tar.bz2
tar jxf $ISL_TAR

CLOOG_TAR=cloog0.18.1.tar.gz
wget --no-clobber -O $CLOOG_TAR ftp://gcc.gnu.org/pub/gcc/infrastructure/cloog-0.18.1.tar.gz
tar zxf $CLOOG_TAR

# For simplicity move the required libraries to the gcc folder so that
# automake discovers them. Make sure that the folders are named appropretly (gcc doc said to use these names). 
rsync -azh cloog-0.18.1/ gcc-4.8.2/cloog
rsync -azh gmp-6.0.0/ gcc-4.8.2/gmp
rsync -azh isl-0.12.2/ gcc-4.8.2/isl
rsync -azh mpc-1.0.1/ gcc-4.8.2/mpc
rsync -azh mpfr-3.1.2/ gcc-4.8.2/mpfr

# yum whatprovides *stubs-32.h 				# may need to use */stubs-32.h instead
sudo yum install glibc-devel-2.12-1.132.el6.i686 	# needed for /usr/include/gnu/stubs-32.h

# now we can begin building gcc 
GCC_PREFIX=/opt/gcc/4.8.2 #change if you wish to use a different prefix. 
cd gcc-4.8.2
make clean
./configure --prefix=$GCC_PREFIX
make -j6
sudo make install

GCC4_8_2_CONF=/etc/ld.so.conf.d/gcc4.8.2.conf
sudo touch $GCC4_8_2_CONF
echo "$GCC_PREFIX/lib64" | sudo tee -a $GCC4_8_2_CONF # append to the file
echo "$GCC_PREFIX/lib"   | sudo tee -a $GCC4_8_2_CONF # append to the file
sudo ldconfig

sudo ln -s $GCC_PREFIX/bin/g++ /usr/bin/g++-4.8 # add the executable to the path. 
sudo ln -s $GCC_PREFIX/bin/gcc /usr/bin/gcc-4.8 # add the executable to the path. 



