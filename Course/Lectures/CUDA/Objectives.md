% Massively Parallel Hardware
% Introduction to Parallel Programming Objectives
% \today

# Objectives

## Terminology Objectives
  1. Cluster
  2. GPU
  3. Massively Parallel
  4. Compute Grid

## Lecture Objectives
  1. ~~A brief history covering the origins of Parallel/GPU computing and how it applies to today.~~
    * ~~Super computer history.~~
    * ~~Systolic Arrays.~~
    * Cellphones and GPU Clusters. TODO-> Slide about cellphones and processors. 
    * ~~Flynn's Taxonomy.~~
      * ~~Parallel models.~~
    * ~~Amdahl's and Gustafson's Laws.~~
    * ~~OpenCL (Kronos) and CUDA (NVIDIA)~~
    * ~~Embarrassingly parallel~~

## Lecture Objectives
### Review CPU architecture,.. Introduction to NVIDIA Hardware. 
  * ~~CPU Architecture and Parallelism~~
  * ~~GPU Architecture~~
    - ~~Provide an image~~
    - ~~Streaming Multiprocessors~~
    - ~~Cores~~

## Lecture Objectives
### Review CPU architecture,.. Introduction to NVIDIA Hardware. 
  - ~~GPU Threading~~
    - ~~threads~~
    - ~~blocks~~
    - ~~grids~~
    - ~~SIMT and SIMD~~

## Lecture Objectives
  - ~~Memory Hierarchy~~
    - ~~CPU <-> bus <-> GPU~~
  - ~~Programming CUDA~~
    - ~~Writing kernels~~
    - ~~Memory transfers \& Executing kernels.~~
    - ~~Compiling run-down.~~

## Advanced Objectives
  - Advanced Programming
    - Using Shared Cache. 
      - Static
      - Dynamic
    - Thread Management
      - race conditions. 
      - atomic operations
      - thread synchronization
        - ``__syncthreads()``
        - ``__threadfence()``
  - Reduce
  - Scan
    - Blelloch Scan
    - Hillis \& Steele Scan
  - Sorting
    - 
  - Algorithm Analysis
    - work efficiency 
    - time efficiency

## Project Objectives
  2. Provide a description of the standard matrix multiplication algorithm.
    * brute force algorithm. (classic matrix multiplication).
    * Strassen algorithm -> Maybe?
    * Briefly describe the project objectives to the students. 



