---
layout: default
title: CUDA & Instant-NGP
date: 2024-03-20
---
<div class="project-page">
  <div class="project-content">
    <h1 class="project-title">
      <a href="https://github.com/KyunHwan/personal_instant_ngp" class="project-github-link" target="_blank" rel="noopener noreferrer">
        <img src="{{ '/assets/icons/github.png' | relative_url }}" alt="GitHub" />
        <span>CUDA &amp; Instant-NGP</span>
      </a>
    </h1>

    Before jumping into implementing minimal version of Instant-NGP, I wanted to understand some fundamentals of NVidia GPU and CUDA. Below are snippets of information I gathered from <b><i>Programming Massively Parallel Processors - A Hands-on Approach 4th Edition</i></b> by Wen-mei W.Hwu, David B. Kirk, and Izzat El Hajj. 

    <p>
    <h2>Compute Architecture & Scheduling</h2>
    
    <h3>Architecture of a modern GPU</h3>
    <b><i>GPU</i></b> is organized into an array of highly threaded <b><i>streaming multiprocessors (SMs)</i></b>. Each SM has several processing units called streaming processors or <b><i>CUDA cores</i></b>. All threads in a block are simultaneously assigned to the same SM. And multiple blocks are likely to be simultaneously assigned to the same SM. But since blocks need to reserve hardware resources, only a limited number of blocks can be assigned to a given SM. 

    <br><br>

    <h3>Block thread scheduling</h3>
    Threads in the same block are scheduled simultaneously on the same SM. This guarantee makes it possible for threads in the same block to interact with each other in ways that threads across different blocks cannot. This includes <b><i>barrier synchronization</i></b> and accessing a low-latency <b><i>shared memory</i></b> that resides on the SM. And by not allowing threads in different blocks to perform barrier synchronization, with each other, CUDA runtime system can execute blocks in any order relative to each other since none of them need to wait for each other. 

    <br><br>

    <h3>Block thread synchronization</h3>
    <b><i>CUDA</i></b> allows <b><i>threads in the same block</i></b> to coordinate their activities using the barrier synchronization function <b><i>__syncthreads()</i></b>. When a thread calls <b><i>__syncthreads()</i></b>, it will be held at the program location of the call until every thread in the same block reaches that location. Thus, it must be executed by all threads in a block. If one <b><i>__syncthreads()</i></b> is in an if-path while another is in the else-path, and there are threads in a block that go into if-path as well as else-path, then this will cause a <b><i>deadlock</i></b>. 

    <br><br>

    <h3>Warps & SIMD Hardware</h3>
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

    <h3>Control divergence</h3>
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
    </p>
  </div>
</div>