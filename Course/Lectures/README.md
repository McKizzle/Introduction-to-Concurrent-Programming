# Lectures
To ensure portability, the lectures have been written in a pandoc-markdown compatible format. Pandoc is capable of converting markdown to other useful formats and document types. In the case of the lectures, pandoc creates beamer presentations using the available markdown files. 

## Setup the Building Environment 
Following these installation instructions will ensure that you have the latest version of pandoc. 

### OS X

  1. First install [homebrew](http://brew.sh/) on the target machine. 
  2. Second install the latest version of MacTeX 
  3. Now install the required binaries using homebrew.
    - `brew install pandoc`

### Ubuntu
  1. First install the following packages using `apt-get`. 
    - `apt-get install texlive haskell-platform`
  2. Export `~/.cabal/` to your path. 
  2. Next install the latest version of pandoc using the Haskell package manager *cabal*
    - `cabal update`
    - `cabal install`

## Building the Lectures

To build the lectures go to the directory that contains the lectures of interest and run:

    `make`

To build the lecture notes run the following command:

    `make notes`


