---
header-includes:
  - \usepackage[absolute,overlay]{textpos}
  - \usepackage{graphicx}
title: "Massively Parallel Hardware"
date: \today

---

## Introduction

An introduction leveraging the computing capabilities of the graphics processing unit to parallelize a task. 

<div class="notes">
  NOTE: To whom ever is reading this document. Pandoc processes
  divs that inherit from the 'notes' class as a note. These will
  not show up in your slides. To compile slides with notes run `make notes`. You will need to be running pandoc --version >= 1.12.
</div>

## Problem
  - Why would we want to use a GPU? 
  - For example lets say an assignment required that you write a function that cubes each element in a array $X$. 
    - $f(X, p) = Y$ such that $X = [x_0, x_1, ... x_{n-1}]$ and $Y=[x_0^3, x_1^3, ... x_{n-1}^3]$. 
  - $X$ contains at least a billion integers and performance is a requirement. 

## Serial Example
  - Up to this point most of your assignments involved writing code that would execute in a serial fashion.

\scriptsize

`````C
#define LENGTH 1000000000 // 1,000,000,000
int main()
{
  int ary[LENGTH] = {2, ..., 2}; // assume that array is initialized with 2s.
  
  int tmp = 0;

  // At least ten CPU cycles per iteration. 
  for(int i = 0; i < LENGTH; i++) {
    tmp = ary[i];
    ary[i] = tmp * tmp * tmp; 
  }

  // About 1,000,000,000 * 10 cycles. ~3 seconds on a 3GHz processor. 
}
`````

## Serial Example 
### Advantages
  - Easy to write
  - Portable

### Disadvantages 
  - Even though it is a simple operation. At least 3 seconds is required complete the work on a 3GHz processor.
  - What if the operation on each element required more work?

## Parallelization Example
\scriptsize

`````C
__on_gpu__ void gpu_cube_array() {
  for(all a_i in array in parallel) {
    tmp = a_i;
    a_i = tmp * tmp * tmp;
  }
}

#define LENGTH 1000000000 // 1,000,000,000
#define CHUNKS 8 // Eight chunks to send to GPU.
int main() {
  int ary[LENGTH] = {2, ..., 2}; // assume that array is initialized with 2s.
  
  for(int i = 0; i < CHUNKS; i++) {
    int start_index = i * LENGTH / CHUNKS;
    int stop_index = start_index + LENGTH / CHUNKS;
    copy_to_gpu(ary[start_index ... stop_index]) // About 120,000,000 cycles. 
    gpu_cube_array(); // About 25,000,000 cycles. 
    copy_from_gpu(ary[start_index ... stop_index]) // about 120,000,000 cycles.
  } // 25,000,000 + 2 * 120,000,000 = 265,000,000 cycles

  // About 2,120,000,000 cycles total. ~1.5 seconds on a 1.5GHz processor. 
}
`````

## Parallel Example
### Advantages
  - If the GPGPU runs at 1.5GHz then it will take at least 1.41 seconds. 
  - Takes less time than the serial example.

### Disadvantages
  - More difficult to read. 
  - More overhead is required to copy the data.
  - Make sure that the GPU memory is not exhausted. 

<div class="notes">
  - Cubing a value is a simple operation. So there wasn't much of a performance gain. 
  - A complex math operation would give the GPU more of an advantage. 
</div>

## Use Case
  - GPGPUs are already being used in the scientific community. 

\begin{figure}
  \includegraphics[width=2in, height=3in]{./img/blank.jpg}
\end{figure}

\begin{textblock*}{2in}(0.25in, 1in) 
  \includegraphics[width=2in]{./img/GPGPU-DNA-SeqAlign}
\end{textblock*}

\begin{textblock*}{2in}(2.5in, 0.65in) 
  \includegraphics[width=2in]{./img/GPGPU-GIS.png}
\end{textblock*}


# History

## ENIAC 
  - Weight 30 tons. 
  - Consumed 200kW. 
  - Required 19,000 vacuum tubes to operate. 
  - Slow (~100 kHz)

  \begin{figure}
    \includegraphics[width=2in]{./img/eniac.jpeg}
    \caption{Unidentified Photographer [photo]. Retrieved from \url{http://en.wikipedia.org/wiki/ENIAC}}
  \end{figure}
 
<div class="notes">
  ENIAC (Electronic Numerical Integrator and Computer) general electric computer. 
</div>

## Colossus Mark II
  - Used a process similar to systolic array to parallelize the cryptanalysis of the enigma.
  - Systolic arrays weren't described until 1978 by H. T. Kung and Charles E. Leiserson. 
    - A grid of processors designed to process instructions and data in parallel. 
  - Modern graphics processing units are similar in design to a systolic array. 

  \begin{figure}
    \includegraphics[width=1.5in]{./img/colossus.jpg}
    \caption{Retrieved from \url{http://en.wikipedia.org/wiki/Colossus_Mark_II}}
  \end{figure}

<div class="notes">
\scriptsize

  - Why are systolic arrays important? They demonstrate that using multiple processors to solve tasks in parallel has been around before GPUs became commonplace. 
  - Even though they were not described until 1978 by H. T. Kung and Charles E. Leiserson the Colossus implemented such a method in order to decrease the time to decrypt Enigma. In reality though, it was never used to decipher Enigma that was the job of Turing's machine the electromechanical Bombe. [Colossus computer](http://en.wikipedia.org/wiki/Colossus_computer).
</div>

## Gaming Consoles and Workstations
  - Beginning in the 70's gaming consoles started to pioneer the use of graphics processing units (GPU) to speed up computations so that an image could be rendered quickly on a display. 
  - It became common to have a dedicated processor to handle system graphics and offload work from the CPU. 
  - Later workstations started to come with dedicated graphics processors. 
  - GPUs were still dumb and nonprogrammable. Therefore using the GPU to perform computations required an understanding of the hardware and strange hacks. 

<div class="notes">
  - Atari is famous for splitting the graphics rendering from other computations.
  - The ANTIC processor generated the text and bitmap graphics.
  - The CTIA/GTIA processor takes the output from ANTIC and adds coloring to the image before sending it to the screen. 
  - [http://en.wikipedia.org/wiki/Atari_8-bit_family](http://en.wikipedia.org/wiki/Atari_8-bit_family)

  - Sega Dreamcast had a PowerVR2 CLX2 dedicated GPU
  - Nintendo 64 had the 62.5 MHz SGI RCP made by Silicone Graphics (SGI)
  - XBox had a custom Nvidia GeForce 3.
</div>

## Dedicated Graphics
  - 1990's 
    - Market fills with dedicated graphics chips makers. 
      - S3 Graphics
      - 3dfx 
      - Nvidia
      - ATI (Array Technology Inc.)
      - Silicone Graphics
  - Post 2000's
    - Dedicated processing units are a standard in laptops and Desktops.
    - Programmable

<div class="notes">
  [iSBX 275 Info](http://www.intel-vintage.info/intelsystems.htm)

  - Silicone Graphics (SGI) is important. They are accredited to creating OpenGL, OpenCl, and the Khronos group to monitor the evolution of the APIs.
  - Khronos oversees the evolution of these open frameworks. [https://www.khronos.org/opencl/](https://www.khronos.org/opencl/)
    - OpenGL stands for _Open Graphics Library_
    - OpenCL stands for _Open Computing Language_
</div>

## Graphics Processing Unit vs CPU
\begin{figure}
  \includegraphics[width=4in]{./img/cpu-gpu.pdf}
  \caption{Simple example of the GPU vs CPU when rendering an image. }
  \label{fig:fixedpipeline}
\end{figure}


## Fixed Graphics Pipeline (1980's - 1990's)
  - Configurable but not programmable. 
  - Fixed Pipeline
    1. Send image as a set of vertexes and colors to the GPU when then performs steps 2 - 6. 
    2. In parallel, determine if each point is visible or not. (_Vertex Stage_)
    3. In parallel, assemble the points into triangles. (_Primitive Stage_)
    4. In parallel, convert the triangles into pixels. (_Raster Stage_)
    5. In parallel, color each pixel based on the color of its neighbors. (_Shader Stage_) 
    6. Display the grid of pixels on the screen. 

<div class="notes">
\scriptsize

  - Refer to [http://cg.informatik.uni-freiburg.de/course_notes/graphics_01_pipeline.pdf](http://cg.informatik.uni-freiburg.de/course_notes/graphics_01_pipeline.pdf). I find that they do a good job breaking down the steps. 
  - A simple breakdown of a fixed pipeline in the GPU. 
    - Send the data to the GPU via OpenGL or DirectX
    - __Vertex Stage:__ Operates on each vertex independently. Therefore this step is embarrassingly parallel.
    - __Primitive Stage:__ Primitives (vertexes) are assembled into lines an triangles. The assembly of each primitive is also parallel. 
    - __Raster Stage:__ Convert all of the geometry data into pixel data. Since this involves a lot of transformation operations then this stage can also be parallelized for each shape. 
    - __Shader Stage:__ Interpolate the colors of the pixels based on the colors of the vertexes. Each pixel is independent of the other pixels. 
</div>

## Programmable Pipeline (2000+)
  - Why not let the programmer customize steps in the pipeline?
  - As a result the programmable pipeline was born. 
    - Allow customization of the vertex processing step. 
    - Allow customization of the shading step. 
    - GLSL (OpenGL shading language)
  - The game designer could now modify how the GPU processed the data in parallel. 
  - More components of the pipeline were converted to keep up with user demands. 
  - Still limited to graphics. Performing other computations required the user to restructure the problem in a form the GPU could understand (as an image).

<div class="notes">
  - Customization of the vertex and shader pipelines by using GLSL (OpenGL shading language) which had a syntax similar to C. 
  - The programmer would write the shader code which would get executed on either each vertex or pixel during the GPU render process. 
  - [http://en.wikipedia.org/wiki/GLSL](http://en.wikipedia.org/wiki/GLSL)
</div>

## Unified Pipeline (Nvidia)
  - Programmable pipeline feature-creep. 
  - Instead of adding more hardware to perform different tasks. Generalize the pipeline such that it can perform each step in the fixed pipeline. 
  - Use an array of unified processors and perform three passes on the data before sending it to the framebuffer. 
    - vertex processing: Take the vertex data, transforms it
    - geometry processing
    - pixel processing
  - GPU can now perform load balancing at each step. 
  - Precursor to the general purpose graphics processing unit (GPGPU). 
 
<div class="notes">
\scriptsize

  The unified processor description is relative to Nvidia's hardware design. ATI may perform a similar method but it may differ. 

  - Vertex Processing: Convert the vertex data and color data into a form that can be used by the GPU. 
  - Geometry Processing: Similar to vertex processing except each primitive is processed. By primative I mean vertexes that belong to triangles or quadrilaterals (it depends on the GPU, Nvidia deals with triangle primitives)
  - Pixel Processing: This is essentially the shading step from the fixed pipeline. (perhaps also the raster step)
</div>

## Unified Pipeline (Nvidia)
\begin{figure}
  \includegraphics[width=4in]{./img/unified-pipeline.png}
  \caption{Nvidia's Unified Pipeline: Adopted from {\em Programming Massively Parallel Processors}. }
  \label{fig:fixedpipeline}
\end{figure}

<div class="notes">
\scriptsize

  - SP stands for streaming processor. SP is also referred to as a CUDA core. 
  - The blocks of SPs in the images are streaming multiprocessors (SMs). 
  NVidia's unified programmable processor array of the GeForce 8800 GTX graphics pipeline.
  - First send the data in through the _Host_ interface, next the data is passed to the input assembler. Once the assembler has completed its task, the GPU will invoke threads for the vertex stage and distribute them amongst the SPs (Streaming Processors). The SPs will perform the three rounds as mentioned in the _Unified Pipeline (Nvidia)_ slide. 

  The advantage to having this setup is that the SPs can distribute the workload depending on the graphics demands. For example if more time is required in the pixel processing state (for image processing) then the GPU will be able to devote more time to that task alone. 

  Essentially the GPU is beginning to resemble a processor with multiple cores.
</div>


## General Purpose Graphics Processing Unit
  - With the birth of the unified pipeline and more advanced shading algorithms. The GPU began to resemble a CPU. Hence the term GPGPU (General Purpose Graphics Processing Unit) was born. 
  - This caught the interest of researchers who used the shading language to solve computational problems. \footnote{ \tiny Kyoung-Su Oh, Keechul Jung, GPU implementation of neural networks, Pattern Recognition, Volume 37, Issue 6, June 2004, Pages 1311-1314, ISSN 0031-3203, http://dx.doi.org/10.1016/j.patcog.2004.01.013.}

\centering
\includegraphics[width=3in]{./img/GPU-NeuralNetwork.png} 


<div class="notes">
  - Understanding the efficiency of GPU algorithms for matrix-matrix multiplication 
    \footnote{ \tiny K. Fatahalian, J. Sugerman, and P. Hanrahan. 2004. Understanding the efficiency of GPU algorithms for matrix-matrix multiplication. }

</div>

## New APIs

\begin{figure}
  \includegraphics[width=1.5in]{./img/opencl-logo.png}
  \includegraphics[width=1.5in]{./img/nvidia-cuda-logo.jpg}
  \caption{GPGPU APIs}
  \label{fig:fixedpipeline}
\end{figure}

  - In response to the research community Nvidia and Khronos created APIs that allow users to repurpose their GPUs by writing their own programs. 
  - Provide interface to program GPGPU. 
    - CUDA (Compute Unified Device Architecture) a proprietary API to Nvidia GPUs. 
    - OpenCL (Open Computing Language) an open standard by the Khronos group.

<div class="notes>
  The students will be focusing on CUDA since it provides a better tool set, which makes it easier to learn for beginners. 

  In the long run, OpenCL has an advantage considering cellphones support it. Which means future applications can take advantage of it and decrease processing times and save battery life. 
</div>

## Speed Evolution

\begin{figure}[!ht]
  \includegraphics[width=3.5in]{./img/floating-point-operations-per-second.png}
  \caption{Retrieved from the Programmers Guide in Nvidia's CUDA Toolkit Documentation}
\end{figure}

<div class="notes">
  This plot was located in Nvidia's on line CUDA Toolkit Documentation ([http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#axzz36uP30F2s](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#axzz36uP30F2s)). It's relative to Nvidia GPU's but it's clear that there is a trend which can be applied to GPGPUs in general (ATI GPUs, and mobile GPUs).

</div>

# Parallelism

## Parallel Processing
  - Bit-level parallelism
    - increase word size to reduce the number of passes a processor needs to perform in order to perform arithmetic. 
    - e.g. 8-bit processor must perform two passes when adding two 16-bit numbers. 
  - Systolic Array
    - Data centric processing. It is a data-stream-driven by data counters unlike the Von-Neumann architecture is an instruction-steam-driven program counter. 
  - Flynn's Taxonomy
    - General categorization of parallel processing paradigms. 

<div class="notes">
  Recall, the first couple
</div>

## Flynn's Taxonomy
|               | Single instruction | Multiple instruction |
|---------------|:------------------:|---------------------:|
| Single data   | SISD               | MISD                 |
| Multiple data | SIMD               | MIMD                 |

<div class="notes">
  Flynn's taxonomy basically states that programs fall into one of the four categories. Each category may have more than one subcategory. 
</div>

## Flynn's Taxonomy
### Single Instruction Single Data
\begin{figure}
  \includegraphics[width=2in]{./img/SISD.png}
  \caption{ Wikipedia: SISD. PU stands for processing unit.}
\end{figure}

<div class="notes">
  Single instruction single data is what most students in the department deal with on a daily basis. Even though we have laptops with multiple cores all of the assignments only require execution on a single core with a single set of data.
</div>

## Flynn's Taxonomy
### Single Instruction Multiple Data
\begin{figure}
  \includegraphics[width=2in]{./img/SIMD.png}
  \caption{ Wikipedia: SIMD. Keep this diagram in mind. GPU parallelism falls into this category}
\end{figure}

<div class="notes">
  Single instruction multiple data is when there exist multiple processing units that can access multiple data points simultaneously. I haven't confirmed this but SQL also falls into this category since it can scale the number of PU's to access the data pool. 
</div>

## Flynn's Taxonomy
### Multiple Instruction Single Data
\begin{figure}
  \includegraphics[width=2in]{./img/MISD.png}
  \caption{ Wikipedia: MISD}
\end{figure}

<div class="notes">
  Multiple instruction streams work on a single stream of data. A systolic array falls into this category. 
</div>

## Flynn's Taxonomy
### Multiple Instruction Multiple Data
\begin{figure}
  \includegraphics[width=2in]{./img/MIMD.png}
  \caption{ Wikipedia: MIMD}
\end{figure}

<div class="notes">
  - Super computers apply this technique. 
  - [https://computing.llnl.gov/tutorials/parallel_comp/](https://computing.llnl.gov/tutorials/parallel_comp/)
</div>

<!-- COMMENTABLE: Comment this section out using html comments if teaching the Concurrency lectures first 

## Amdahl's Law
  - Determine the potential code speedup with Amdahl's Law.
    - This version assumes that a case of parallelization. 

  $$ T(N) = \frac{1}{(1 - P) + \frac{P}{N}} $$

  - $P$ is a value between 0 and 1. $P$ is the fraction of the program that is executed in parallel. 
  - As $P$ approaches 1 then the program becomes more parallelized. 
  - If $P == 1$ then the program is solving an _embarrassingly parallel_ problem. The speedup is linear to the number of cores. 
  - $N$ is the count of processors. 
  - Amdahl's law assumes a fixed problem size. Which causes a diminishing returns effect as the number of cores increase. 

<div class="notes">
  - [http://en.wikipedia.org/wiki/Amdahl%27s_law](http://en.wikipedia.org/wiki/Amdahl%27s_law)
  - [http://www.drdobbs.com/parallel/amdahls-law-vs-gustafson-barsis-law/240162980](http://www.drdobbs.com/parallel/amdahls-law-vs-gustafson-barsis-law/240162980)

  - Amdahl's Law assumes the dataset that the program is working on is static in size. 
  - For example a program that only parallelizes a fixed sized problem would follow this rule. Let's say that a program will always multiply two 1000x1000 matrices. Then the speedup of that program will follow Amdahl's law as the number of processors increase. 
</div>

## Amdahl's Law

\begin{figure}
  \includegraphics[width=3in]{../img/amdahls-law.pdf}
\end{figure}

## Gustafson's Law
  - Unlike Amdahl's Law, Gustafson's Law assumes that the program will work on larger problems to utilize all of the processors. 

  $$ S(N) = N - P(N - 1) $$

  - $P$ is the fraction of the program that is parallel. 
  - $N$ is the number of processors. 

<div class="notes">
  - [http://en.wikipedia.org/wiki/Gustafson%27s_law](http://en.wikipedia.org/wiki/Gustafson%27s_law)
  - [http://www.drdobbs.com/parallel/amdahls-law-vs-gustafson-barsis-law/240162980](http://www.drdobbs.com/parallel/amdahls-law-vs-gustafson-barsis-law/240162980)
</div>

## Gustafson's Law
\begin{figure}
  \includegraphics[width=3in]{../img/gustafsons-law.pdf}
\end{figure}

<!-- END COMMENTABLE -->

## Embarrassingly Parallel Problem 
  - A problem that can be subdivided into smaller problems that are independent of each other.
    - Matrix multiplication. Each cell in the new matrix is independent of the other cells. 
    - Processing vertexes and pixels on the GPU is embarrassingly parallel. 
    - Genetic Algorithms 
    - Bioinformatics: BLAST (Basic Local Alignment Tool) searches through a genome. 

<div class="notes">
  - Later in the lectures we will mention the matrix multiplication algorithm.
  - The GPU pipelines in the previous slides evolved to handle embarrassingly parallel problems.
  - A genetic algorithm works by taking a population applying mutations and killing off the members. Since each member is independent of the other members then it is embarrassingly parallel. 
  - A divide and conquer approach can be taken by BLAST when searching through a genome. 
</div>

# Architecture Comparison: CPU vs GPGPU

## CPU Parallelism Structure
  - Multiple CPUs per motherboard. 
  - Multiple cores per CPU.
  - 2 or more CPUs per motherboard.
  - Clustered hardware or distributed hardware. 
  - Memory is shared amongst computer peripherals. 

## CPU Parallelism Structure
### Advantages 
  - Low latency
  <!-- give background information -->
  - Interrupts
  - Asynchronous events
  - Branch prediction
  - Out-of-order execution
  - Speculative execution
  - Scalable with minimal hardware 

<div class="notes">
  - CPUs are optimized (once their pipeline is full) minimize latency at a cost of capacity. 
    - Think of a CPU as a race car. It can get you somewhere in less time. 
  - CPUs are designed to handle system interrupts, such as from the mouse and keyboard. 
  - CPUs can handle multiple threads with different instructions each. 
  - CPUs may try to predict how the program executes in order to increase performance. 
  - Instructions my be reordered to increase performance. 
  - All that is needed at a minimum is a motherboard, ram, and a CPU. 
</div>

## CPU Parallelism Structure
### Disadvantages
  - Low throughput
  - CPU's have evolved to do generalized work. 
  - They are not optimized to do a single task extreamly well. 
  - Limited number of threads (more complex but limited)
  - May not be the best solution for math intensive scientific computations. 

<div class="notes">
  - Perhaps wait time isn't an issue. Maybe the program cares more about processing data in bulk at the cost of latency. 
  - CPUs are setup to perform generalized work, ranging from security, to managing and communicating with all of the devices attached to the motherboard (including the GPU). 
  - A CPU on average can execute 2 * n threads where n is the number of cores. Any value greater than that will have diminishing returns. 
  - If a problem is math oriented then perhaps it is best to use a math oriented processor?

</div>


<!-- General Purpose Graphics Processing Unit == GPGPU -->
## GPGPU Parallelism Structure
### Structure
  - Single Die with hundreds of massively parallel processors.
  - Each processor is simpler than a standard CPU core.
    - handle large amounts of concurrent threads. 
  - Dedicated memory for the GPU.

## GPGPU Parallelism Structure
### Advantages
  - High throughput
    - threads hide memory latencies. 
  - Efficient at solving math intensive problems. 
  - Handles larger amounts of data far quicker than the CPU. 
  - No need to deal with system interrupts. 
  - Free the CPU of unnecessary work. 

<div class="notes">
  - The GPU is able to process large datasets in parallel. Think of it as a bus that can carry 50 people at a time. If you needed to transport 200 people from Missoula to Portland it would be better to use a bus than a sports car. If needed provide a simple math breakdown of the problem. 
  - The GPU is optimized for algebraic functions and basic math. 
  - Since the GPU has a lot of threads it can hide RAM accesses. 
  - The GPGPU has been designed in mind for dedicated graphics and computations.
</div>

## GPGPU Parallelism Structure
### Disadvantages
  - High latency
  - Needs to process large amounts of data. (not good for small tasks)
  - Processes running in the GPU can only access resources available on the GPU card. 
  - Can only handle math intensive tasks. Cannot take advantage, of network communication, system calls,... 
  - Requires base hardware for the GPU (motherboard, CPU, storage) cannot run alone.

<div class="notes">
  - GPU's have high latency because:
    1. The CPU needs to transfer the data to the GPU. 
    2. The GPU has smaller cache sizes than the CPU therefore more cache misses. 
    3. The GPU usually has a slower clock speed. 
    4. Data needs to be transfered back to the system once the computations have completed. 
  - Hiding this latency requires that large datasets be used or computed. 
  - The GPU cannot communicate with peripherals on the machine. For example you cannot establish a network connection through the GPU to a remote machine.
</div>

# GPGPU Architecture

## Nvidia GPGPU Architecture
  - Streaming Multiprocessors (SM)
    - CUDA (Compute Unified Device Architecture) cores. Also called streaming processors (SP)
      - CUDA cores are an Nvidia branded ALU (Arithmetic Logic Unit). 
    - Control units
    - Registers (local memory)
    - Execution pipelines
    - Caches (shared memory)

<div class="notes">
  - The number of CUDA cores per SM depends on the _compute capability_ of the GPU. Right now there are 32 cuda cores per SM. Refer to [http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities).
</div>

## Nvidia GPGPU Architecture
### Fermi Architecture
\begin{figure}[!ht]
  \includegraphics[width=2.5in]{./img/fermi-architecture.png}
  \caption{Retrieved from Nvidia's Fermi Whitepaper}
\end{figure}

<div class="notes">
\scriptsize

  - The Fermi white papers can be located at [http://www.nvidia.com/content/PDF/fermi_white_papers/NVIDIA_Fermi_Compute_Architecture_Whitepaper.pdf](http://www.nvidia.com/content/PDF/fermi_white_papers/NVIDIA_Fermi_Compute_Architecture_Whitepaper.pdf)
  - The Fermi architecture is not the newest iteration, but newer architectures follow a similar model. They simply have more CUDA cores and features. 

  - L2 Cache is shared amongst the SMs. This is also referred to as _shared memory_ when developing in CUDA. 
  - The DRAM on the sides in the image are referred to as _global memory_
  - The _GigaThread_ a proprietary Nvidia module on their GPU. It helps manage all of the thread being executed. There are probably similar components on other GPUs from different manufacturers. [http://www.nvidia.co.uk/page/8800_faq.html](http://www.nvidia.co.uk/page/8800_faq.html)

</div>

## Nvidia GPGPU Architecture
### Fermi Streaming Multiprocessor 

\begin{figure}[!ht]
  \includegraphics[width=1.5in]{./img/streaming-multiprocessors-fermi.png}
  \caption{Retrieved from Nvidia's Fermi Whitepaper}
\end{figure}

<div class="notes">
\scriptsize

  - An exploded diagram of on of the SMs in a Fermi GPU. Students don't need to know this by heart, but it will aid them greatly if they are able to recognize the relationship between a CUDA core and an SM. 
  - A CUDA core is essentially a specialized ALU (Arithmetic Logic Unit) that contains both an ALU and FPU (Floating Point Unit). 
  - Other components that are not that important given the context of the course. 
    - LD/ST stands for _Loading and Storing_. Each LD/ST unit is able to handle a single thread request per clock. 
    - SFT stands for _Special Function Units_. They execute transcendental instructions such as sin, cosine, reciprocal, and square root. [http://www.nvidia.com/content/PDF/fermi_white_papers/NVIDIA_Fermi_Compute_Architecture_Whitepaper.pdf](http://www.nvidia.com/content/PDF/fermi_white_papers/NVIDIA_Fermi_Compute_Architecture_Whitepaper.pdf) 

</div>


## Nvidia GPGPU Architecture
### CUDA Core (ALU)
  - Floating point (FP) unit
    - Single Precision
    - Double Precision (far slower than single precision)
  - Integer (INT) unit
    - Boolean, Move, Compare

\begin{figure}[!ht]
  \includegraphics[width=1in]{./img/CUDA-Core.png}
  \caption{Retrieved from Nvidia's Fermi Whitepaper}
\end{figure}

<div class="notes">
\scriptsize
  - Double precision is slower on the GPU. The speed depends on the architecture, in some cases it can be up to 8x slower. 
  - FPUs have been optimized to perform single precision floating point operations. 
  - The INT unit is designed to handle all of the regular operations with integers that we are familiar with.
    - boolean values
    - bit shifting
    - comparing
    - bit reversing (change the order of the bits.
</div>

# Programming with CUDA

## Overview
  - Minimal code example. 
  - Step through the code and explain CUDA concepts. 
  - Provide minimal kernel example.
  - Step through kernel code and explain CUDA concepts. 

## Minimal Working Example
  - Minimal program to transfer data to the GPU, do some work, and transfer it back. 

\scriptsize

`````C
#include<vector>
#include<cuda_runtime.h>
#define LENGTH 1024
__host__ int main() {
  std::vector<int> host_array(LENGTH, 2); 

  int * dev_array = NULL; // Points to the memory located on the device. 
  cudaMalloc((void **)&dev_array, LENGTH * sizeof(int));
  cudaMemcpy(dev_array, &host_array[0], LENGTH * sizeof(int), 
             cudaMemcpyHostToDevice);

  dim3 blockDims(LENGTH, 1, 1); // dim3 is struct supplied by cuda_runtime.h
  dim3 gridDims(1, 1, 1);
  kernel_cube_array<<<gridDims, blockDims>>>(dev_array, LENGTH);
  cudaDeviceSynchronize();

  cudaMemcpy(&host_array[0], dev_array, LENGTH * sizeof(int), 
             cudaMemcpyDeviceToHost);
  cudaFree(dev_array);
}
`````

## Transfer Data to the Device
  - Most problems involve the manipulation of a dataset. 
  - Before the GPU can perform any work the data must be transfered over to its memory. 

\scriptsize

`````C
  std::vector<int> host_array(LENGTH, 2); 

  int * dev_array = NULL; // Points to the memory located on the device. 
  cudaMalloc((void **)&dev_array, LENGTH * sizeof(int));
  cudaMemcpy(dev_array, &host_array[0], LENGTH * sizeof(int), 
             cudaMemcpyHostToDevice);
`````
\normalsize

  - `cudaMalloc` initializes a contiguous set addresses in the GPU's _global_ memory.
  - `cudaMemcpy` copies the data from the `host_array` which resides in the computer's main memory to the `dev_array` on the GPU's global memory. 

<div class="notes">
  - Refer to section 3.4 (page 68) in _Programming Massively Parallel Processors_ for some nice visuals of the process. 
</div>

## Thread Hierarchy
  - The next three lines involve setting up the thread partitioning scheme. 

\scriptsize

`````C
  dim3 blockDims(LENGTH, 1, 1);
  dim3 gridDims(1, 1, 1);
  kernel_cube_array<<<gridDims, blockDims>>>(dev_array, LENGTH);
`````
\normalsize

  - Threads can be considered as the _currency_ of the GPU. They are the smallest execution unit that is defined by the kernel function `kernel_cube_array`
  - `dim3 blockDims(LENGTH, 1, 1)` defines the block dimensions. Every thread _belongs_ to a block. 
  - `dim3 gridDims(1, 1, 1)` defines the dimension of the grid. Every block _belongs_ to a grid. 
  - `kernel_cube_array<<<gridDims, blockDims>>>(dev_array, LENGTH);` tells the GPU to execute a single block of size `LENGTH`. 

<div class="notes">
  - The class shouldn't worry about the kernel implementation right now. 
  - The trip chevrons are an extension to the ANSI C language that the `nvcc` compiler
  - Chapter 4 in _Programming Massively Parallel Processors_ talks about threads in an easy fausion. 
  - Also [http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy) is a good reference which the book is based off of. 
</div>

## Thread Hierarchy 
 
\begin{figure}[!ht]
  \includegraphics[width=2.75in]{./img/grid-of-thread-blocks.png}
  \caption{Retrieved from the Programmers Guide in Nvidia's CUDA Toolkit Documentation}
\end{figure}

## Blocks
  - Blocks of threads can be partitioned amongst the streaming multiprocessors since they are independent. 

\begin{figure}[!ht]
  \includegraphics[width=2.5in]{./img/automatic-scalability.png}
  \caption{Retrieved from the Programmers Guide in Nvidia's CUDA Toolkit Documentation}
\end{figure}

<div class="notes">
  - Blocks can be efficiently distributed amongst SMs since each block is independent. 
  - Since blocks are independent the GPU can maximize the throughput of the blocks. 
  - Also Section 4.3 (page 68) in the book _Programming Massively Parallel Processors_
</div>

## Transferring Data to the Host
  - Now that the work has been completed we must transfer the data back to the host. 

\scriptsize

`````C
  cudaDeviceSynchronize();
  cudaMemcpy( &host_array[0], dev_array, LENGTH * sizeof(int), 
              cudaMemcpyDeviceToHost);
  cudaFree(dev_array);
`````

\normalsize

  - Wait for all of the threads to complete their work `cudaDeviceSynchronize`. 
  - Then transfer the data back to the host device with `cudaMemcpy`. 
  - All memory that was allocated must be freed with `cudaFree`. 

## Kernel Implementation
  - Implementation of `kernel_cube_array`. 

\scriptsize

`````C
__device__ int cube(int a) {
  return a * a * a;
}

__global__ void kernel_cube_array(int * dev_array, int length) {
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;

  __shared__ int shared_mem[LENGTH];
  shared_mem[tidx] = dev_array[tidx];
  __syncthreads();
  
  int b = shared_mem[tidx];
  shared_mem[tidx] = cube(b);
  __syncthreads();

  dev_array[tidx] = shared_mem[tidx];
}
`````

<div class="notes">
  - The kernel implementation was left out on purpose for simplicity. 
</div> 

## Kernel Functions
  - Kernels are the definitions of thread instances. <!-- not sure if terminology is correct --> 
  - Special functions that are designed to run on the GPU. 
  - Written using an ANSI C syntax with additional extensions.
    - `__device__` kernels are only callable on the GPU through other kernels.
    - `__global__` kernels are only callable from the host but not from the device. 
    - `__host__` kernels are simply classic C functions. If a function doesn't have any declarations it defaults to this one. 
    - \<\<\<\>\>\> to define how a kernel executes (as seen in the earlier slides). 

<div class="notes">
  - Refer to page 51 in the book _Programming Massively Parallel Processors_ form more information about kernel functions. 
  - guest only kernels can be only called from other threads running on the kernel. 
</div>

## Thread Indexes
  - The first line in the kernel builds an index for the thread to use. 

\scriptsize

`````C
  int tidx = threadIdx.x + blockIdx.x * blockDim.x
`````

\normalsize

  - CUDA provides a set of ANSI C extensions that allow the programmer to access thread, block, and grid properties.
    - `threadIdx` is the index of the _thread relative to the block_. 
    - `blockIdx` is the index of the _block relative to the grid_. 
    - `blockDim` is the dimensions of the block. 
    - `gridDim` is the dimensions of the grid. 
  - You can combine use these extensions to determine the ID the GPU uses for the thread. 
    - `tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y`
  - Custom indexing schemes can also be implemented.

## Thread Indexes
  - How can thread indexes be used? 
    - Custom indexes can be used to access memory in different ways. 
      - _map_: Each thread index corresponds to a single cell in memory. A _one-to-one_ mapping. 
      - _gather_: Each thread index corresponds to multiple locations in memory. A _many-to-one_ mapping. 
      - _scatter_: Each thread index can be used to write to multiple locations. A _one-to-many_ mapping.
      - _stencil_: Similar to a _gather_ a stencil can be used calculate a new value from many values. Stencils are primarily used for image manipulation and simulation operations. 
    - Control execution paths of individual threads in a block. 
    - Other forms of thread communication. 

<div class="notes"> 
  - Udacity's Parallel Programming course covers memory access patterns in section 2. 
  - The next slide will contain a graphical representation of each indexing scheme. 
</div>

## Thread Indexing
### Memory Access Patterns 

\begin{figure}[!ht]
  \includegraphics[width=11cm]{./img/memory-access-patterns.pdf}
  \caption{White boxes represent memory addresses. Grey boxes represent cells. The arrows represent the direction of communication (read / write). }
\end{figure}

<div class="notes">
  - Understanding how to calculate the global thread index will aid a great deal when calculating indexes into global memory arrays. 
    - For example lets say that a matrix is stored as a 1D array in memory. Calculating the thread index can be used to construct an index into that array. 
  - More information about thread hierarchies is available at [http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy)
  - Refer to chapter 4 of _Programming Massively Parallel Processors_ for more information.
</div>

## Shared Memory. 
  - The next couple of lines load the data from global memory into the shared memory (L2 cache). 

\scriptsize

`````C
  __shared__ int shared_mem[LENGTH];
  shared_mem[tidx] = dev_array[tidx];
`````

\normalsize

  - The `__shared__` keyword defines a variable that is visible to all threads in a block. 
  - The next line tells each thread to copy its corresponding data from _global_ memory to _shared_ memory. 

## Memory Hierarchy
  - Global
    - Visible amongst all blocks. 
    - Main memory of the device. 
    - Large capacity 1GB to 6GB (continuing to grow). 
    - Slow and high latency. 
  - Shared
    - Shared amongst threads in a block. 
    - L2 cache that is shared amongst the SMs 
    - Small capacity (several megabytes)
  - Local 
    - Visible only to executing thread. 
    - Stored on the registers partitioned to the active thread. 
    - Stored in _global memory_. 

<div class="notes">
  - Cache sizes are dependent on the compute capability of the device. Higher compute capabilities correlates to higher cache sizes. 
</div>

## Memory Hierarchy

\begin{figure}[!ht]
  \includegraphics[width=2.5in]{./img/memory-hierarchy.png}
  \caption{Retrieved from the Programmers Guide in Nvidia's CUDA Toolkit Documentation}
\end{figure}

<div class="notes">
  - Basic breakdown of the GPGPU memory hierarchy.
</div>

## Memory Hierarchy
  - `int b = shared_mem[tidx];`. 
  - `b` is what you call a local variable. Local variables are only visible to individual threads. 
  - Terminology is confusing. Local memory is __not__ guaranteed to be fast. 
  - The compiler determines where to store local variables. According to Nvidia's documentation the following rules will determine if a local variable is stored in global memory. 
    - Arrays for which the compiler cannot determine that they are indexed with constant quantities.
    - Large structures or arrays that would consume too much register space. 
    - Any variable if the kernel uses more registers than available (this is also known as _register spilling_). 

## Memory Hierarchy
### Scopes Overview

| Memory   |  Scope |    Lifetime |
|----------|:------:|------------:|
| Register | Thread |      Kernel |
| Local    | Thread |      Kernel |
| Shared   |  Block |      Kernel |
| Global   |  Grid  | Application |
| Constant |  Grid  | Application |

Part of Table 5.1 in _Programming Massively Parallel Processors_. 

## Device Functions
  - Besides writing kernels it is also possible to write device functions that are callable from a kernel. 
  - `shared_mem[tidx] = cube(b);` makes a call to the device function `__device__ cube(int a)`. 
  - Device functions are only callable from the GPU. You cannot have a `__host__` function such as `main` call a device functions. 
  - Useful if there is a need to modularize GPU code. 

## Thread Synchronization
  - The kernel function `kernel_cube_array` contains two calls to the function `__syncthreads()`
  - `__syncthreads()` causes all thread instances in a __block__ to wait for other threads in the same block to catch up before continuing. 
  - \includegraphics[width=2.5in]{./img/syncthreads.pdf}
  - `__threadfence()` is similar in functionality except that it signals all threads in all blocks to catch up.

## Kernel Function
  - Recall that we are working with shared memory. 
  - Need to transfer results back to global memory. 
  - All that is needed is to map the results from the shared array to the global array. 
  - `dev_array[tidx] = shared_mem[tidx];`

## Overview of Learned Objectives
  - Learned to initialize memory. 
  - Transfer data to and from the GPU. 
  - Divide the problem into blocks of threads and a grid of blocks. 
  - Call a kernel function to invoke the thread instances. 
  - Implement kernel and device functions. 
  - Introduced to thread hierarchies.
  - Introduced to memory hierarchies. 
  - Synchronizing thread actions. 

## Extra Information on Threads

  - When blocks are being executed by an SM. Not all threads are executing concurrently. 
  - SMs execute warps of threads at a time. A warp usually consists of 32 threads. 
  - GPU that has a compute capability of 3.0 supports at maximum:
    - 64 concurrent warps per SM.
    - Each warp consists of 32 threads executing in parallel. 
  - The Quadro K2000 has 2 SMs. Therefore it can execute $2 * 64 * 32 = 4096$ threads concurrently. 
  - The Tesla S2050 has 14 SMs. Therefere it can execute $14 * 64 * 48 = 28762$ threads concurrently!

<div class="notes">
\scriptsize

  - To figure out the GPU specs given a compute capability visit the following links:
    - [http://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capabilities](http://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capabilities)
    - [https://developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus)
  - The instruction set is the kernel function that the programmer wrote. 
  - If a thread goes to sleep while waiting for a load/store then the stall can by hidden by executing another thread. 
  - It isn't essential that students know about how SM warp size or the maximum amount of threads an SM can handle. Instead treat this slide as a precursor to explain thread hierarchies which is important for CUDA programming.
  - Also knowing about warps can allow the student to better understand how the GPU works. 
</div>

## Compilation
 - General compilation procedure when calling `nvcc` on source file (ends in the `.cu` extension) with existing kernel functions. 

\begin{figure}[!ht]
  \includegraphics[width=4in]{./img/compilation-procedure.pdf}
  \caption{ Simplified version of Nvidia's compilation procedure.}
\end{figure}

<div class="notes">
\scriptsize

  - `nvcc` is the Nvidia CUDA compiler. It is an extension of the standard C/C++ compiler. 
  - There are other extensions but we'll stick with `.cu` for consistency. 
  
  - Compilation procedure explained. 
    1. The `nvcc` compiler first separates CUDA code from C/C++ code in the passed in `.cu` file.
    2. The C/C++ and CUDA code are compiled. 
      - C/C++ is converted into the classic object form. 
      - Depending on the `nvcc` compiler options. CUDA code is compiled into a cubin or PTX. 
        - _cubin_ is specific to the target GPU architecture. 
        - PTX is an intermediate code that can be further compiled by the GPU driver of the target device. 
    3. The objects are then linked together into an executable. 
</div>

# Matrix Multiplication

## Matrix Multiplication Algorithm
  - Let $A$, $B$, and $C$ be three matrices. Such that $A$ is $m \times n$, $B$ is $n \times o$, and $C$ is the product of $A$ and $B$ (it's dimensions are $m \times o$).
  - The classic algorithm states that every cell in $C$ is the dot product of the corresponding row and column in $A$ and $B$ respectively. 
  - $c_{ij} = \mathbf{a_{i}} \bullet \mathbf{b_{j}}$ where $\mathbf{a_{i}}$ is row $i$ $A$ and $\mathbf{b_{j}}$ is column j in $B$. 
    - $1 <= i <= m $ and $1 <= j <= o$ <!-- => keep the syntax highlighter happy -->
  - Iterate through all values of $i$ and $j$ to calculate the values of $C$. 
  - Implementation consists of a double loop through all cells in $C$.

<div class="notes">
  - Quick refresh for students. Matrix dimensions are specified as row $\times$ column. 
  - Classic method is to assume a 2D array. We'll talk about the 1-D implementation of a matrix. 
</div>

## Matrix Multiplication Algorithm

\begin{figure}[!ht]
  \includegraphics[height=2.5in]{./img/matrix-multiplication.pdf}
  \caption{Reproduced from Nvidia's CUDA Toolkit Documentation}
\end{figure}

<div class="notes">
  - This figure appears _Programming Massively Parallel Processors_ page 65 and in [http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory). 
</div>

## Matrix Data Structure. 
  - 1D or 2D array of doubles?
    - A 2D structure requires more work to transfer to the GPU. `cudaMalloc` and `cudaMemcpy` required for each row in the matrix. 
    - 1D structure requires single `cudaMalloc` and `cudaMemcpy`. 
  - Access row 4 and col 2 in a $6 \times 6$  row-major matrix.
    - If `a` is 2D then `a[4][2]`
    - If `a` is 1D then `a[4 * 6 + 2]`
  - Access row $i$ and col $j$ in a $m \times n$ row-major matrix.
    - `a[i * n + j]`

<div class="notes">
  - For now lets assume that our matrix simply stores double precision values.
  - Assumes that we start at index 0 instead of 1.
  - TODO: Find descriptive image. 
</div>

## Project 1: Matrix Multiplication
  - Implement the classic matrix multiplication algorithm twice.
    - Once for the CPU. 
    - Once for the GPU by writing your own kernel function. 
  - Implement necessary code to manage memory and transfer data between the GPU and CPU. 

# Advanced Thread Management 

## Thread Management Objectives
  - Atomic Operations
    - add, divide, and multiply.
  - Thread Syncronization
    - `__syncthreads()`
    - `__threadfence()`

## Race Conditions
  - When two or more threads are accessing and modifying a value in memory (global or shared) yet the order in which they do it is important. If that order isn't followed and it causes undefined behavior then we have a __race condition__
  - E.X. A group of CUDA threads are binning the values in a large dataset. Incrementing counts in the bins can result in a race condition when the value changes before it is stored back in memory. Thereby creating inconsistencies. 

<div class="notes">
  - [http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-fence-functions](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-fence-functions) contains information about threadfence.
  - [http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#synchronization-functions](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#synchronization-functions) contains information about thread synchronization. 
  - [http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions) contains information about atomic operations.
</div>


## Race Conditions
\scriptsize

`````C
/// rseq and bins are pointers to an array of ints in global memory. 
__global__ void bin_kernel_simple(int * rseq, int * bins, 
                                  int bin_width, int rseq_len)
{
    const int index = threadIdx.x + blockIdx.x * blockDim.x;
    if ( index < rseq_len) bins[rseq[index] / bin_width]++;
}
`````

\normalsize

  - Simple program that bins up values from a random sequence of integers (`rseq`). 
  - `rseq` resides in global memory. When `bins[rseq[index] / bin_width]++` executed then there will be a race condition. 
    - Why? Each SM is executing at most $w$ threads concurrently. Therefore, it is possible to have $w * s$ ($s$ is the count of SMs on the device) threads overwriting a single value in `bins`. 

<div class="notes">
\scriptsize

  - A simple explanation of the binning program. First we calculate the access index into the random sequence `rseq`. If the calculated index is within the bounds of `rseq` then we update the corresponding bin in `bins` depending on the value from `rseq`.
  - Since multiple threads may be accessing a single element `bins` then its value will become inconsistant. 
  - The insperation for this example was from Udacity's Parallel Programming Course. To view their orinal code go to [https://github.com/udacity/cs344/blob/master/Lesson%20Code%20Snippets/Lesson%203%20Code%20Snippets/histo.cu](https://github.com/udacity/cs344/blob/master/Lesson%20Code%20Snippets/Lesson%203%20Code%20Snippets/histo.cu)
</div>

## Atomic Operations

  - A simple solution to a race condition is to make use of an atomic operation which are supported by CUDA. Here are a few that may be usefull. 
    - `atomicAdd`, `atomicSub`, `atomicMin`, and  `atomicMax`. 
    - Additional operators can be found in [http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions)
  - Updating the `if` statement will fix the race condition. `atomicAdd` requires that we pass the address of the bin in `bins`.

\scriptsize

`````C
if (index < rseq_len) atomicAdd( &bins[rseq[index] / bin_width], 1);
`````

<div class="notes">
  - Using atomic operations are slow since all threads writing the locked memory location have to wait.
</div>


## Thread Syncronization
  - What if an algorithm consists of multiple substeps and the order in which the substeps are executed is important? 
  - Stop execution of threads until all threads reach a desired state. 
    - `__syncthreads()` threads in a thread block will wait for other threads in the same block to catch up. 
    - `__threadfence()` threads in all thread blocks will wait for other threads to catch up. 

<div class="notes">
  - Thread syncronization becomes a lot more important when making use of shared memory. 
</div>

# Advanced Memory Management

## Shared Memory
  - Global memory is slow. If possible use shared memory to decrease the count of global memory accesses. 
  - Shared memory is visible amongst all threads in a block, and the lifetime is to that of the block. 
  - Simple use cases:
    - Shared memory to prevent global memory locks due to using the `atomicAdd` operation.
    - Prefetch data before performing memory intensive calculations.  

## Shared Memory Allocation 
  - There are two ways of using shared memory to reduce the count of global memory locks. 

### Static Allocation
  - Define the variable or array at the beginning of the kernel using the `__shared__` keyword.
  - If sharing an array of data the size must be known at compile time. 
  - Example: 

\scriptsize

`````C
#define DEFAULT_SHARED_SIZE 1024
__global__ void kernel_shared() {
    __shared__ int array_of_ints[DEFAULT_SHARED_SIZE]; 
}
`````

## Shared Memory Allocation 
### Dynamic Allocation
  - Only a single variable can be dynamically allocated.
  - The size of the shared memory must be known at the kernel invocation. 
  - `extern` keyword required so that `nvcc` recognizes it. 
  - Example:

\scriptsize

`````C
#define DEFAULT_SHARED_SIZE 1024
__global__ void kernel_shared() {
    extern __shared__ int dynamic_array_of_ints[]; 
}

void kernel_invoking_function() {
  ...
  size_t shared_size = 1024 * sizeof(int);

  kernel_shared<<<1024,1024,shared_size>>>();
  ...
}
`````

<div class="notes">
  - Since only a single variable can be dynamically allocated then if you want to manage more than one dynamic variable you will have to partition the data manually. 
</div>

## Shared Memory Question
 - Will this kernel allocate the shared memory in global memory or L2 memory?

\scriptsize

`````C
#define DEFAULT_SHARED_SIZE 1024
__global__ void kernel_shared()
{
    __shared__ int * shared_bins;

    if(threadIdx.x == 0) { // first allocate the shared memory. 
        size_t shrd_sz = DEFAULT_SHARED_SIZE * sizeof(int);
        shared_bins = (int *)malloc(shrd_sz);
        memset(shared_bins, 0, shrd_sz);
    }
    __syncthreads(); // Wait for the memory allocation. 

    ... // other computations
}
`````

<div class="notes">
  - Using `malloc` inside of a kernel will allocate the shared memory within global memory. Only the pointer resides in shared memory. 
  - This is bad considering the block performance is hindered due to the global memory allocation. To make matters worse the shared memory will be as slow as global memory. We do not want that. 
  - Always use the two previous methods to allocate onto the SM's L2 cache. 
</div>

# Parallel Algorithms

## Complexity of a Parallel Algorithm 
  - Step Complexity
    - The number of iterations a parallel device needs to do before completing an algorithm. 
  - Work Complexity
    - The amount of work a parallel device does when running a algorithm.
  - Work Efficient 
    - A parallel algorithm that is asymptotically the same as its serial implementation. 

## Reduce
  - Matrix multiplication project was a simple mapping since each operation was independent of other operations. 
  - A reduction is dependent of other components but can still be parallelized. 
    - A reduction can be applied if the operator $\oplus$ satisfies the following conditions. 
      a. Binary
      b. Associative
      c. Identity element 
   - Take a sequence of values and apply a binary operator over each pair of elements to get an overall sum. 

<div class="notes">
  - Reduction can take an array of numbers and sum them up in parallel. 
  - Lesson 3 on the Udacity Intro to Parallel Programming Course. 
  - [https://www.youtube.com/watch?v=N1eQowSCdlw](https://www.youtube.com/watch?v=N1eQowSCdlw) 
</div>

## Reduction as a Tree
  - Given the properties of $\oplus$ the reduction can be represented as a tree where each branch is calculated independently. 
  - For example lets sum reduce $[1, 2, 3, 4, 5, 6, 7, 8]$ then we would have the resulting tree. 


\begin{figure}[!ht]
  \includegraphics[height=2.5in]{./img/reduction-graph.pdf}
  \caption{Each level in the tree can be parallelized.}
\end{figure}

## Reduction as a Tree Complexity
  - The work complexity of the tree reduction is $O(n)$ since the device will still need to perform $n-1$ operations to reduce a list of numbers.
  - The step complexity of the tree reduction is $O(\log(n))$ since each level on the tree is independent and the height of a binary tree is $\log(n)$ where $n$ is the number of leaves.

<div class="notes">
  - Refer to _Lesson 3: Steps and Work_ in Udacity's _Intro to Parallel Programming_ course. 
    - https://www.youtube.com/watch?v=V8TTrUdfpIY
</div>


## Reduce Serial

  - Serial implementation of a reduction. 

`````C
int sum_reduce(std::vector< int > & sequence) {
  int sum = 0;
  for(int i = 0; i < sequence.size(); i++) {
    sum += sequence[i]; 
  }

  return sum;
}
`````

## Reduce Kernel
  - Simple example a reduction implemented in CUDA. 

\scriptsize

`````C
  __shared__float partialSum[];
  
  unsigned int t = threadIdx.x;
  for( unsigned int stride = 1; stride < blockDim.x, strid *= 2)
    __syncthreads();
    if(t % (2 * stride) == 0) {
      partialSum[t] += partialSum[t + stride]
    }
  }
`````

<div class="notes">
  - Example from the book __Programming Massively Parallel Processors__ on page 100. 
  - The book provides a better implementation of a reduction on page 103. Perhaps open a discussion on how to further improve the kernel function. 
</div>

## Scan 
  - Scan applies concepts similar to reduce. 
  - Take a sequence of values, apply a binary operator while keeping a cumulative output. 
  - Example:
    - INPUT: $S = [ a_0, a_1, a_2, a_3, ... a_n], \quad i = I, \quad \oplus$ where $S$ is the sequence and $i$ is the identity for the operator $\oplus$. 
    - OUTPUT: 
      - Exclusive: $[ I, (I \oplus a_0), ( I \oplus a_0 \oplus a_1 ), ( I \oplus a_0 \oplus a_1 \oplus a_2 ),... (I \oplus ... \oplus a_{n - 1})]$

<div class="notes">
  - A scan operation takes a sequence of values and provides a cumulative output given an operator. 
  - More information about the scan operator can be found at GPU Gems 3 [http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html](http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html). 
  - More information is also available at Udacity's online course __Intro to Parallel Programming__ section three. Refer to the inclusive and exclusive scan videos. 
</div>

## Scan Serial
  - Serial implementation of scan.

\scriptsize

`````C
// 'in' and 'out' have the same length. 
void sum_scan(std::vector< int > & in, std::vector< int > & out) {
  int sum = 0;
  for( int i = 0; i < int.size(); i++ ) {
    out[i] = sum;
    sum += in[i];
  }
}
`````

\normalsize

## Scan Algorithms
### Hillis-Steele
  - Simple to implement, but lacks efficiency when scanning large arrays. 

### Blelloch
  - More work efficient than the Willis-Steel scan. Has the same efficiency as the serial scan.

### Others
  - Kogge–Stone
  - Brent Kung

## Hillis-Steele
  - Takes a simple approach to parallelizing the scan. Treats the summation operations as a binary tree with no attempt to minimize the number of operations. 

\begin{figure}[!ht]
  \includegraphics[height=1.5in]{./img/hillis-steele.jpg}
  \caption{Hillis-Steele Scan: Adopted from {\it GPU Gems 3}}
\end{figure}

## Hillis-Steel Complexity
  - What is the complexity of the Hillis-Steele algorithm?
    - Work complexity?
    - Step complexity?

<div class="notes">
  - The work efficiency is $O(n\log(n))$.
  - The step efficiency is $O(\log(n))$. 

## Hillis-Steele Implementation
  - Hillis-steele scan using a single block of threads.

\scriptsize

`````C
#define SEQ_SIZE 1024
#define BLOCK_SIZE 1024
__global__ void hs_scan_kernel(int * in, int * out) {
  int tid = threadIdx.x;

  __shared__ int shared[BLOCK_SIZE];

  shared[tid + 1] = in[tid]; 
  __syncthreads();

  for( int d = 1; d <= log2(SEQ_SIZE); ++d ) {
    if( tid >= (1 << (d - 1))) {
      shared[tid] += shared[tid - (1 << (d - 1))];
    }
    __syncthreads();
  }

  out[tid] = shared[tid];
}
`````

<div class='notes'>
  - Hillis-Steele algorithm implementation from section 39.2.1 from GPU Gems 3. 
</div>

## Blelloch
  - Splits the scan operation into two phases, a down-sweep and up-sweep phase. 

### Down-Sweep

\begin{figure}[!ht]
  \includegraphics[height=1.25in]{./img/down-sweep.jpg}
  \caption{Blelloch Down-Sweep: Adopted from {\it GPU Gems 3}}
\end{figure}

## Blelloch 
### Down-Sweep Complexity
  - What is the complexity of the down-sweep portion?
    - Work complexity?
    - Step complexity?

<div class="notes">
  - The amount  of work performed is $\log(n)$. 
  - The number of steps required is $\log(n)$. 
  - More information about Blelloch can be found in _Lesson 3_ on Udacity's _Intro to Parallel Programming Course_
</div>

## Blelloch
### Down-sweep Implementation

\scriptsize

`````C
__device__ void bl_sweep_down(int * to_sweep, int size) {
  to_sweep[size - 1] = 0;
  int t = threadIdx.x;
  for( int d = log2(size) - 1; d >= 0; --d) { 
    __syncthreads();
    if( t < size && !((t + 1) % (1 << (d + 1)))) {
      int tp = t - (1 << d);
      int tmp = to_sweep[t];
      to_sweep[t] += to_sweep[tp];
      to_sweep[tp] = tmp;
    }
    __syncthreads();
  }
}
`````

## Blelloch
### Up-Sweep

\begin{figure}[!ht]
  \includegraphics[height=1.5in]{./img/up-sweep.jpg}
  \caption{Blelloch Up-Sweep: Adopted from {\it GPU Gems 3}}
\end{figure}


## Blelloch 
### Up-Sweep Complexity
  - What is the complexity of the down-sweep portion?
    - Work complexity?
    - Step complexity?

<div class="notes">
  - The amount  of work performed is $\log(n)$. 
  - The number of steps required is $\log(n)$. 
  - More information about Blelloch can be found in _Lesson 3_ on Udacity's _Intro to Parallel Programming Course_
</div>

## Blelloch 
### Up-sweep implementation

\scriptsize

`````C
__device__ void bl_sweep_up(int * to_sweep, int size) {
  int t = threadIdx.x;
  for( int d =  0; d < log2(size); ++d) {
    __syncthreads();
    if( t < size && !((t + 1) % (1 << (d + 1)))) {
        int tp = t - (1 << d);
        to_sweep[t] = to_sweep[t] + to_sweep[tp];
    }
    __syncthreads();
  }
}
`````

## Blelloch
### Complexity
  - What is the complexity of the Blelloch algorithm?

### Kernel Implementation
  - What would the kernel need to contain in order to make use of the up-sweep and down-sweep functions?

## Compact
  - What is compacting?
    - The process of extracting a subset from a set of items given a condition
    - Items in a set are compared against a condition. (usually an array) 
    - The output of the comparison is then stored in a dataset called the predicate. 
    - Using the predicate and scan operator the desired items can then be mapped into a new set. 

## Compact 
### Example
  - Find elements that are either a multiple of three or five. 
  - $S$ is the set to compact. 
    - $S = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]$
  - The condition to check on each element is $s \in S$ is $(s \mod 3 = 0) | (s \mod 5 = 0)$
  - Resulting in the predicate set $P$. 
    - $P = [0, 0, 1, 0, 1, 1, 0, 0, 1,  1,  0,  1,  0,  0,  1]$
  - Applying the scan operator on $P$ and output $C$
    - $C = [0, 0, 0, 1, 1, 2, 3, 3, 3,  4,  5,  5,  6,  6,  6]$
  - Now map the items from $S$ where each item $P$ is 1 to the corresponding index in $C$ into a new set $N$. 
    - $N = [3, 5, 6, 9, 10, 12, 15]$

<div class="notes">
  - Refer to Udacity's _Intro to Parallel Programing: Lesson 4_ for more information on the compact operation. 
    - [https://www.youtube.com/watch?v=GyYfg3ywONQ](https://www.youtube.com/watch?v=GyYfg3ywONQ)
</div>

## Compact

\begin{figure}
  \includegraphics[width=11cm]{./img/compact-steps.pdf}
\end{figure}

# Projects & Homework

## Homework: Scan Algorithms
  - Implement the Blelloch and Hillis-Steele algorithms.
  - CUDA program should only be a single file. 
  - Scan only a single block of 1024 threads. 

## Project 2: Sorting

  - Implement the radix sort algorithm. 
  - Radix sort requires that you make use of the compact algorithm. Recall that compacting requires you:
    - Scan 
    - Scatter
  - Refer to [http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html](http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html) for more information. 


