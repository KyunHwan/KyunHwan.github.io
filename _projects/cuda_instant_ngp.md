---
layout: default
title: CUDA & Instant-NGP
date: 2024-03-20
---
<div class="project-page">
  <div class="project-content">
    <div class="sidebar">
      <ul>
        <li><a href="#compute-architecture">Compute Architecture & Scheduling</a></li>
        <ul>
          <li><a href="#gpu-architecture">Architecture of a modern GPU</a></li>
          <li><a href="#block-scheduling">Block thread scheduling</a></li>
          <li><a href="#thread-sync">Block thread synchronization</a></li>
          <li><a href="#warps-simd">Warps & SIMD Hardware</a></li>
          <li><a href="#control-divergence">Control divergence</a></li>
        </ul>
        <li><a href="#memory-architecture">Memory Architecture & Data Locality</a></li>
        <ul>
          <li><a href="#memory-efficiency">Importance of memory access efficiency</a></li>
          <li><a href="#cuda-memory-types">CUDA memory types</a></li>
        </ul>
        <li><a href="#instant-ngp">Instant NGP</a></li>
        <ul>
          <li><a href="#multi-res-enc">Multi-resolution hash encoding</a></li>
        </ul>
      </ul>
    </div>

    <h1 class="project-title">
      <a href="https://github.com/KyunHwan/personal_instant_ngp" class="project-github-link" target="_blank" rel="noopener noreferrer">
        <img src="{{ '/assets/icons/github.png' | relative_url }}" alt="GitHub" />
        <span>CUDA &amp; Instant-NGP</span>
      </a>
    </h1>

    Before jumping into implementing minimal version of Instant-NGP, I wanted to understand some fundamentals of NVidia GPU and CUDA. Below are snippets of information I gathered from <b><i>Programming Massively Parallel Processors - A Hands-on Approach 4th Edition</i></b> by Wen-mei W.Hwu, David B. Kirk, and Izzat El Hajj. 

    <p>
    <h2 id="compute-architecture">Compute Architecture & Scheduling</h2>
    <br>
    <h3 id="gpu-architecture">Architecture of a modern GPU</h3>
    <b><i>GPU</i></b> is organized into an array of highly threaded <b><i>streaming multiprocessors (SMs)</i></b>. Each SM has several processing units called streaming processors or <b><i>CUDA cores</i></b>. All threads in a block are simultaneously assigned to the same SM. And multiple blocks are likely to be simultaneously assigned to the same SM. But since blocks need to reserve hardware resources, only a limited number of blocks can be assigned to a given SM. 

    <br><br>

    <h3 id="block-scheduling">Block thread scheduling</h3>
    Threads in the same block are scheduled simultaneously on the same SM. This guarantee makes it possible for threads in the same block to interact with each other in ways that threads across different blocks cannot. This includes <b><i>barrier synchronization</i></b> and accessing a low-latency <b><i>shared memory</i></b> that resides on the SM. And by not allowing threads in different blocks to perform barrier synchronization, with each other, CUDA runtime system can execute blocks in any order relative to each other since none of them need to wait for each other. 

    <br><br>

    <h3 id="thread-sync">Block thread synchronization</h3>
    <b><i>CUDA</i></b> allows <b><i>threads in the same block</i></b> to coordinate their activities using the barrier synchronization function <b><i>__syncthreads()</i></b>. When a thread calls <b><i>__syncthreads()</i></b>, it will be held at the program location of the call until every thread in the same block reaches that location. Thus, it must be executed by all threads in a block. If one <b><i>__syncthreads()</i></b> is in an if-path while another is in the else-path, and there are threads in a block that go into if-path as well as else-path, then this will cause a <b><i>deadlock</i></b>. 

    <br><br>

    <h3 id="warps-simd">Warps & SIMD Hardware</h3>
    What about execution timing of threads within each block? 
    <br>
    Once a block has been assigned to an SM, it is further divided into 32-thread units (size is implementation specific) called <b><i>warps</i></b>. A warp is the <b><i>unit of thread scheduling in SMs</i></b>.

    <br><br>

    Blocks are partitioned into warps on the basis of thread indices, such that each warp consists of 32 threads of consecutive threadIdx values: threads 0 through 31 form the frist warp, threads 32 through 63 form the second warp, and so on. 

    <br><br>

    An SM is designed to execute all threads in a warp following the <b><i>single-instruction, multi-data (SIMD)</i></b> model. That is, at any instant in time, one instruction is fetched and executed for all threads in the warp. 

    <br><br>

    Cores in an SM are grouped into <b><i>processing blocks</i></b>, where each one shares an instruction fetch/dispatch unit. For example, Ampere A100 SM, which has 64 cores, is organized into four processing blocks with 16 cores each. Threads in the same warp are assigned to the same processing block, which fetches the instruction for the warp and executes it for all threads in the warp at the same time. 


    <br><br>

    <h3 id="control-divergence">Control divergence</h3>
    When threads in the same warp follow different execution paths, we say that these threads exhibit <b><i>control divergence</i></b>
    
    <br><br>
    In the Pascal architecture and prior architectures:
    <br>
    When there is an if-else construct and threads within a warp take different control flow paths, the SIMD hardware will take multiple passes through these paths, one pass for each path. More concretely, if some threads in a warp follow the if-path while others follow the else-path, the hardware will take two passes. One pass executes the threads that follow the if-path, and the other executes the threads that follow the else-path. During each pass, the threads that follow the other path are not allowed to take effect.

    <br><br>

    From the Volta architecture onwards:
    <br>
    The passes may be executed concurrently, such that the execution of one pass may be interleaved with the execution of another pass. 

    <br><br>

    Important implication of control divergence is that one cannot aassume that all threads in a warp have the same execution timing. Therefore if all threads in a warp must complete a phase of their execution before any of them can move on, one must use a barrier synchronization mechanism such as <b><i>c__syncwarp()</i></b> to ensure correctness.

    <br><br>

    <h2 id="memory-architecture">Memory Architecture & Data Locality</h2>
    <br>
    This goes into the on-chip memory architecture of the GPU, and how to oragnize and position data for efficient access by as massive number of threads. 
    
    <br><br>
    
    GPUs provide a number of additional on-chip memory resources, other than global memory, for accessing data that can remove the majority of traffic to and from the global memory.

    <br><br>

    This section introduces a commonly used technique for reducing the number of global memory accesses and demonstrates the technique on matrix multiplication. 

    <br><br>

    <h3 id="memory-efficiency">Importance of memory access efficiency</h3>
    In a matrix multiplication scenario (call them M & N), the global memory accesses fetch elements from the M & N arrays. The floating-point multiplication operation multiplies these two elements together, and the floating-point add operation accumulates the product into the result value. Thus the ratio of floating-point operations (FLOP) to bytes (B) accessed from global memory is 2 FLOP to 8 B, or 0.25 FLOP/B. We will refer to this ratio as the <b><i>compute to global memory access ratio</i></b>, defined as the <b><i>number of FLOPs performed for each byte access from the global memory within a region of a program</i></b>, also referred to as <b><i>arithmetic intensity</i></b> or <b><i>computational intensity</i></b>.

    <br><br>

    <h4>Example</h4>
    <br>
    A100 GPU has a peak global memory bandwidth of 1555 GB/s. Since the matrix multiplication kernel performs 0.25 OP/B, the global memory bandwidth limits the throughput of single-precision FLOPs that can be performed by the kernel to 389 GFLOPS, obtained by multipliying 1555 GB/s with 0.25 FLOP/B. However, 389 GFLOPS is only 2% of the peak single-precision operation throughput of the A100 GPU, which is 19,500 GFLOPS. 

    <br><br>

    The A100 also comes with special purpose units called <b><i>tensor cores</i></b> that are useful for accelerating matrix multiplication operations. If one considers the A100's tensor-core peak single-precision floating-point throughput of 156,000 GFLOPS, 389 GFLOPS is only 0.25% of the peak. Thus the execution of the matrix multiplication kernel is severely limited by the rate at which the data can be delivered from memory to the GPU cores. 

    <br><br>

    We refer to programs whose execution speed is limited by memory bandwidth as <b><i>memory-bound programs</i></b>.

    <br><br>

    <h3 id="cuda-memory-types">CUDA memory types</h3>
    <b><i>Global memory</i></b> and <b><i>constant memory</i></b> can be written and read by the host. Global memory can also be written and read by the device, while constant memory supports short-latency, high-bandwidth read-only access by the device. 
    
    <br><br>
    
    There's also <b><i>local memory</i></b>, which can be read and written. It's actually placed inside global memory, but not shared across threads. Each thread has its own section of global memory that it uses as its own private local memory where it places data that is private to the thread but cannot be allocated in registers (Ex. statically allocated arrays, spilled registers). 
    
    <br><br>

    <b><i>Registers</i></b> and <b><i>shared memory</i></b> are on-chip memories. 
    
    <br><br>
    
    Registers are allocated to individual threads; each thread can access only its own registers. A kernel function typically uses registers to hold frequently accessed variables that are private to each thread. Each access to registers involves fewer instructions than an access to the global memory since the operand would have to be moved from the global memory to register file before an operation is executed on it. 
    
    <br><br>

    Shared memory is allocated to thread blocks; all threads in a block can access shared memory variables declared for the block. It's an efficient means by which threads can cooperate by sharing their input data and intermediate results. 

    <br><br>

    Question now is how do we declare a variable so that it will reside in the intended type of memory? 

    <br><br>

    <table>
      <tr>
        <th>Variable declaration</th>
        <th>Memory</th>
        <th>Scope</th>
        <th>Lifetime</th>
      </tr>
      <tr>
        <td>Automatic variables other than arrays</td>
        <td>Register</td>
        <td>Thread</td>
        <td>Grid</td>
      </tr>
      <tr>
        <td>Automatic array variables</td>
        <td>Local</td>
        <td>Thread</td>
        <td>Grid</td>
      </tr>
      <tr>
        <td>__device__ __shared__ int SharedVar</td>
        <td>Shared</td>
        <td>Block</td>
        <td>Grid</td>
      </tr>
      <tr>
        <td>__device__ int GlobalVar</td>
        <td>Global</td>
        <td>Grid</td>
        <td>Application</td>
      </tr>
      <tr>
        <td>__device__ __constant__ int ConstVar</td>
        <td>Constant</td>
        <td>Grid</td>
        <td>Application</td>
      </tr>
    </table>

    <br>

    If a variable's <b><i>scope</i></b> is a single thread, a private version of the variable will be created for every thread; each thread can access only its private version of the variable. 

    <br><br>

    <b><i>Lifetime</i></b> tells the portion of the program's execution duration when the variable is available for use: either within a grid's execution or throughout the entire application. If a variable's lifetime is within a grid's execution, it must be declared within the kernel function body and will be available for use only by the kernel's code. If the kernel is invoked several times, the value of the variable is not maintained across these invocations. On the other hand, if a variable's lifetime is throughout the entire application, it must be declared outside of any function body. 

    <br><br>

    <h4>Automatic scalar variables</h4>
    All <b><i>automatic scalar variables</i></b> that are declared in kernel and device functions are placed into registers. The scopes of these automatic variables are within individual threads. Though accessing these variables is extremely fast and paralle, one must be careful not to exceed the limited capacity of the register storage in the hardware implementations. 

    <br><br>

    <h4>Automatic array variables</h4>
    <b><i>Automatic array variables</i></b> are not stored in registers. Instead, they are stored into the thread's local memory and may incur long access delays and potential access congestions. The scope of these arrays is limited to individual threads. 

    <br><br>

    <h4>Shared variables</h4>
    If a variable declaration is preceded by the <b><i>__shared__</i></b> keyword, it declares a <b><i>shared variable</i></b> in CUDA. These reside in the shared memory. The scope is within a thread block (ie. all threads in a block see the same version of a shared variable). A private version of the shared variable is created for and used by each block during kernel execution. The lifetime of a shared variable is within the duration of the kernel execution.

    <br><br>

    <h4>Constant variables</h4>
    If a variable declaration is preceded by keyword <b><i>__constant__</i></b>, it declares a <b><i>constant variable</i></b> in CUDA. Declaration must be outside any function body. The scope is all grids, meaning that all threads in all grids see the same version of a constant variable. The lifetime of a constant variable is the entire application execution. These are often used for variables that provide input values to kernel functions. The values of the constant variables cannot be changed by the kernel function code. They are stored in the global memory but are cached for efficient access.

    <br><br>

    Currently, the total size of constant variables is limited to 65,536 bytes. 

    <br><br>
    
    <h4>Global variables</h4>
    <b><i>Global variable</i></b> will be placed in the global memory. They are visible to all threads of all kernels. So they can be used as a means for threads to collaborate across blocks. But there is currently no easy way to synchronize between threads from different thread blocks or to ensure data consistency across threads in accessing global memory other than using atomic operations or terminating the current kernel execution.     

    <br><br><br>

    <h2 id="instant-ngp">Instant-NGP</h2>
    Nerf represents a continuous 3D function f(x) (density, color, or SDF) with small, fixed memory.
    
    <br><br>
    
    <h3 id="multi-res-enc">Multi-resolution hash encoding</h3>

    <b><i>Instant-NGP (<a href="https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf">link</a>)</i></b> uses <b><i>multi-resolution hash encoding (<a href="https://www.youtube.com/watch?v=CGqhCc3BrKk&amp;ab_channel=BendingSpoons">link</a>)</i></b> to capture both coarse & fine details. 
    
    <br><br>

    <h4>Strategy</h4>
    Use <b>L</b> grid levels (Around 16).
    <ul>
      <li> Level <i>0</i> has low resolution (eg. 2<sup>1</sup> cells per axis)</li>
      <li> Level <i>L - 1</i> has high resolution (eg. 2<sup>19</sup> cells per axis)</li>
      <li> For each level we store a <i>tiny</i> learnable feature vector of length <i>F</i> (often 2 or 4)</li>
    </ul>

    <br>

    <h4>Problem</h4>
    A dense 3-D grid with side length 2<sup>19</sup> would need 2<sup>57</sup> voxels, which could blow up easily if we extend this.

    <br><br>

    <h4>Solution</h4>
    <b>Hash</b> the integer voxel coordinate (x, y, z) into a much smaller table (~= 2<sup>14</sup> entries per level).
    <ul>
      <li> Collisions are okay; the network learns to disentable them</li>
      <li> Want the hash to be fast and spatially uniform </li>
    </ul>

    <br>

    Below are the parameters that Instant-NGP uses. 
    <table border="1">
      <thead>
        <tr>
          <th>Parameter</th>
          <th>Symbol</th>
          <th>Value</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Number of levels</td>
          <td>L</td>
          <td>16</td>
        </tr>
        <tr>
          <td>Max. entries per level (hash table size)</td>
          <td>T</td>
          <td>2<sup>14</sup> to 2<sup>24</sup></td>
        </tr>
        <tr>
          <td>Number of feature dimensions per entry</td>
          <td>F</td>
          <td>2</td>
        </tr>
        <tr>
          <td>Coarsest resolution</td>
          <td>N<sub>min</sub></td>
          <td>16</td>
        </tr>
        <tr>
          <td>Finest resolution</td>
          <td>N<sub>max</sub></td>
          <td>512 to 524288</td>
        </tr>
      </tbody>
    </table>

    The task is to make the hash table and the MLP used for Nerf to fit into L2 cache of the GPU, which is 64MB on my laptop RTX 4090. 

    <br><br>

    
    
    </p>
  </div>
</div>