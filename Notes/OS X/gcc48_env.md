# Setting Up GCC 4.8 on OS X
The recommended method of installing gcc 4.8 on OS X is to use 
[homebrew](http://brew.sh/ "Homebrew home"). To install run the following commands. 

    sh > brew tap homebrew/versions
    sh > brew update
    sh > brew upgrade
    sh > brew install ...

Once you have installed gcc make sure to use the export the following variables in your environment. 

## gmp

    LDFLAGS:  -L/usr/local/opt/gmp4/lib
    CPPFLAGS: -I/usr/local/opt/gmp4/include

## mpfr

    LDFLAGS:  -L/usr/local/opt/mpfr2/lib
    CPPFLAGS: -I/usr/local/opt/mpfr2/include

## libmpc

    LDFLAGS:  -L/usr/local/opt/libmpc08/lib
    CPPFLAGS: -I/usr/local/opt/libmpc08/include

## isl

    LDFLAGS:  -L/usr/local/opt/isl011/lib
    CPPFLAGS: -I/usr/local/opt/isl011/include

## cloog

    LDFLAGS:  -L/usr/local/opt/cloog018/lib
    CPPFLAGS: -I/usr/local/opt/cloog018/include





