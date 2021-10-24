# How to reason about numerical code optimization on modern processors - Part 3 : SIMD, Architecture and Strip Mining

​	In the previous two Parts, we covered several topic on how to improve numerical code performance. Algorithm choice, basic compiler flags and basic thread parallelism were covered, and to an extent they are common ground in most writings in the subject of improving performance. 

​	We now turn to a more restricted topic that is related to extracting performance of **one given architecture**, though it is possible to make it portable to other CPUs. We will do so by exploiting existing hardware that is cable to perform parallel computations **inside the core** instead of among cores, most notably we will explore functionality to enable code vectorization using **SIMD units**. These units are designed to provide high performance, in the form of high-throughput, in a set of repetitive and restrictive types of calculations that are very common in practice, like adding and multiplying numbers together. These units allow for a very large increase of floating point performance by being able to execute multiple calculations through the use of a single instruction.

​	Though very rare in the past, and at times restricted to specialty hardware, this kind of hardware-level parallelism is quite common in the present day because of the limitations of further increasing clock speed of processors. I can have a 12 core desktop CPU that is capable of executing 8 floating point instructions per clock per core at $4\ \mbox{GHz}$, but it is basically unfeasible to built a single-core CPU that can run at $700+\ \mbox{GHz}$ without it melting. 

​	Contrary to what used to happen in the past, in which performance increases with increasing frequency translated directly into execution performance without the developer needing to do anything, resources like SIMD units require explicit programmer awareness to correctly create code that execute on the available hardware. The need for this awareness is, to some extent, a reasonable demand for the existence of this series of articles. 

​	To effectively utilize these SIMD units, it is necessary to annotate your code to correctly warn the compiler that is can be translated to vectorized code and also instruct the compiler directly to do so, in the form of correct compiler flags, both topics being the subject of this part of the series. By the end of this part, I hope you come to appreciate the amazing evolution and cleverness of modern CPU architectures, and also realize that, though not free, it is not overly complex to adapt your code to take advantage of these resources. We will also make a first foray into advanced performance techniques, like strip mining, and how to map them into your code. Some of these more advanced techniques will be covered in the next part of this series.

​	Similarly as done with the previous parts, we present a summary of the results of this part in comparison with the previous ones:

|          Algorithm / Implementation / Configuration          |  Time (Seconds)  | Speedup (Rel. to slowest) |
| :----------------------------------------------------------: | :--------------: | :-----------------------: |
| Naive Calculation (i.e. direct two loop) / AoS, simple, no optimizations / gcc -std=c11 -Wall |  188.46 +- 0.71  |          **1 x**          |
| Cell Linked List / AoS, simple, no optimizations / gcc -std=c11 -Wall | 3.642  +- 0.043  |         **51 x**          |
| Naive Calculation (i.e. direct two loop) / SoA, OpenMP / gcc -std=c11 -Wall -O3 |  4.267 +- 0.049  |         **44 x**          |
|   Cell Linked List / SoA, OpenMP / gcc -std=c11 -Wall -O3    |  0.313 +- 0.033  |         **600 x**         |
| Naive Calculation / AoS, OpenMP, SIMD / gcc -std=c11 -O3 -ffast-math -march=native |  4.408 +- 0.009  |         **42 x**          |
| Cell Linked List / AoS, OpenMP, SIMD / gcc -std=c11 -O3 -ffast-math -march=native |  0.274 +- 0.059  |         **687 x**         |
| Naive Calculation / SoA, OpenMP, SIMD / gcc -std=c11 -O3 -ffast-math -march=native | 0.7507 +- 0.0018 |         **251 x**         |
| Cell Linked List / SoA, OpenMP, SIMD / gcc -std=c11 -O3 -ffast-math -march=native | 0.1598 +- 0.0122 |        **1179 x**         |

## Understanding and Enabling SIMD 

​	Modern CPUs are capable not only of executing multiple threads in their cores, but can also execute computations in parallel **inside their cores**. This comes in two flavors: one "mostly free" called [Superscalar architecture](https://en.wikipedia.org/wiki/Superscalar_processor) and another not free at all called [Single Instruction, Multiple Data](https://en.wikipedia.org/wiki/SIMD) (SIMD) units. 

​	The intuition behind the SIMD is relatively simple: Many codes, specially numerical code, are constituted for fairly large iterations of equal or very similar operations, many times only using basic floating-point or integer arithmetic. If many of these operations can be coalesced together, e.g. because they operate in independent data, it is possible to issue a single instruction that perform 2, 4 or even 8 of these operations *at once*, therefore promising a big speedup. In comparison, multiple cores offer a different kind of parallelism called Multiple Instruction Multiple Data (MIMD). The overall idea for SIMD looks like this:

​												![https://www.techrepublic.com/article/the-web-takes-a-step-closer-to-becoming-a-computing-platform-with-simd/](https://www.techrepublic.com/a/hub/i/2014/09/16/e6400e0e-96b4-42c1-95be-79cd7e939321/simd-example.jpg)

​	Historically, it is possible to trace the idea of SIMD back to the first Supercomputer, the [LANL Cray-1](https://en.wikipedia.org/wiki/Cray-1) vector machine, and have different showed up in different iterations throughout computer history including modern [vector processors](https://en.wikipedia.org/wiki/NEC_SX-Aurora_TSUBASA), present day [Graphic Processing Units](https://en.wikipedia.org/wiki/Vector_processor#GPU) in general with a particular poignant example present in the newer [Tensor Cores](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/). Because of this previous history of vector processing with specialty hardware, it is common to the process of enabling SIMD in a code as "vectorization" of the code, even if it doesn't run in a specialized vector processor. Usually, it is **only the inner-most loop** that is vectorizable, and thus, contrary to the case of openMP threads, the attention must be paid to whichever goes in the central and costliest part of the code. 

​	Though the promise is fairly straightforward, actually making good on this promise is anything but. The reasons for it will be explored in more detail throughout the rest of this work but the first road-block begins with the fact that, for the SIMD units to work properly the CPU **expects** that the data that will be operated on is laid out in memory in a way that consecutive elements in memory **correspond to consecutive operands** in the instruction. What this means is that if you have particle $i$ with position `x` , it expects that the memory is laid out like a conventional array `...x[i], x[i+1], x[i+2], ...` and not in a strided fashion as occurs with the original layout we used in the struct `... x[i], y[i], z[i], ... rho[i], x[i+1], y[i+1], z[i+1], ...`. 

​	Just that modification, if not done early in code development, [can be quite the headache to do](https://arxiv.org/abs/1612.06090v1). Since I was aware of this quite early in the project, I was capable of investigating the impact of the necessary changes in order to see if I was able to obtain substantial performance gains from SIMD, which ultimately did happen. The first modification was already done, which was the AoS to SoA transformation discussed in Part 2. 

​	In my case, the Ryzen 3900x CPU has two SIMD per core units which are capable of executing [a 4 operation block](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#Advanced_Vector_Extensions_2) in the form `a = b*c+a` called [3 operand fused multi-add](https://en.wikipedia.org/wiki/FMA_instruction_set#Instructions) (FMA3), and variations thereof, on 3 trios of 64-bit floating point data per clock. The specific implementation of SIMD present in the Ryzen 3900x is called [Advanced Vector Extentions 2](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions), also known as AVX256. To be able to exploit all available processing power, the code must be able to utilize these instructions as much as possible. 

### Ending compiler divination: `-fopt-info-vec-missed`

​	At first glance, it should be the compiler's job do produce the code that runs on the underlying hardware, even somewhat "exotic" hardware like these SIMD units. Unfortunately, it can be quite complicated to divine what the compiler is doing, so actually **making the compiler print** some reports on how it is optimizing the code is quite important. The main flag I used to report on vectorization of the code was the flag `-fopt-info-vec-missed` of `gcc`. The flags make the compiler print a report on which loops it failed to vectorize and why. Tracing back to the code, it is possible to read what is says and try to understand. In our case, for the `exec/example_02-ArrayOfStructs-Naive-Omp.c` file have the following report:

```
exec/example_02-ArrayOfStructs-Naive-Omp.c:200:9: missed: couldn't vectorize loop
exec/example_02-ArrayOfStructs-Naive-Omp.c:200:9: missed: not vectorized: control flow in loop.
exec/example_02-ArrayOfStructs-Naive-Omp.c:204:26: missed: couldn't vectorize loop
exec/example_02-ArrayOfStructs-Naive-Omp.c:204:26: missed: not vectorized: control flow in loop.
exec/example_02-ArrayOfStructs-Naive-Omp.c:208:14: missed: statement clobbers memory: dist_83 = sqrt (dist_46);
exec/example_02-ArrayOfStructs-Naive-Omp.c:232:5: missed: statement clobbers memory: exit (10);
```

​	The key loop starts at line 201. The three major warnings are `204:26: missed: not vectorized: control flow in loop.`, ` 208:14: missed: statement clobbers memory: dist_83 = sqrt (dist_46);` and `232:5: missed: statement clobbers memory: exit (10);`. By control flow in the loop, the compiler means `if` and `for` statements, specially if it can produce results that can't be absorbed into the computation itself. In our case the kernel utilized is defined below:

```C
double w_bspline_3d(double r,double h){
  const double A_d = 3./(2.*M_PI*h*h*h);      // The 3d normalization constant 
  double q=0.;                                // normalized distance, initialized to zero
  
  if(r<0||h<=0.)                              // If either distance or smoothing length
    exit(10);                                 // are negative, declare an emergency
  
  q = r/h;                                    // Compute the normalized distance
  if(q<=1)                                    // If the distance is small
    return A_d*(2./3.-q*q + q*q*q/2.0);       // Compute this first polynomal
  else if((1.<=q)&&(q<2.))                    // If the distance is a bit larger
    return A_d*(1./6.)*(2.-q)*(2.-q)*(2.-q);  // Compute this other polynomial 
  else                                        // Otherwise, if the distance is large
    return 0.;                                // The value of the kernel is 0
}
```

​	The kernel has a complex branching in the form of `if(r<0||h<=0.) exit(10)`, which makes things very difficult for the compiler to produce vectorized code as indicated by `232:5: missed: statement clobbers memory: exit (10);`. The kernel also has some internal `if` statements, but those can be absorbed in the behavior of the kernel because all they do is to change the `return` value. The compiler also doesn't like `sqrt(dist)`, but this will be dealt as a consequence of the other modifications. Our goal now is to make our code more amenable to vectorization.

​	The main lesson from this section is: **Don't try to guess what the compiler is doing, print reports and profile**. 

### Kernel Optimization: Rethinking what goes in the vectorized section

​	The first thing that needed to be done was to simplify the most complex part of the inner-most loop: The Kernel. The idea is to strip bare whatever goes in it, move the division outside, simplify the conditionals in the kernel function, and indicate to the compiler that the function can be safely inlined and transformed with SIMD instructions. This hint to the compiler is done with `#pragma omp declare simd`. The resulting function is as below, with the auxilary `_constant` function that returns the normalization constant.

```c
double w_bspline_3d_constant(double h){                            
  return 3./(2.*M_PI*h*h*h);  // 3d normalization value for the b-spline kernel
}

#pragma omp declare simd
double w_bspline_3d_simd(double q){                                // Use as input the normalized distance
  double wq = 0.0;
  double wq1 = (0.6666666666666666 - q*q + 0.5*q*q*q);             // The first polynomial of the spline
  double wq2 = 0.16666666666666666*(2.-q)*(2.-q)*(2.-q);           // The second polynomial of the spline
  
  if(q<2.)                                                         // If the distance is below 2
    wq = wq2;                                                      // Use the 2nd polynomial for the spline

  if(q<1.)                                                         // If the distance is below 1
    wq = wq1;                                                      // Use the 1nd polynomial for the spline
  
  return wq;                                                       // return which ever value corresponds to the distance
}
```

​	The re-worked naïve computation can be seen below. We also introduce hints to the compiler to indicate that the inner-most loop should be vectorized, this is done with `#pragma omp simd` . It may be needed to use an option `aligned(x,y,z,nu)` in the `#pragma`, though it was not necessary here,  indicating to the compiler that the arrays indicated have the [proper memory alignment](https://stackoverflow.com/questions/31089502/aligned-and-unaligned-memory-access-with-avx-avx2-intrinsics), which I find to be one of the more arcane requirements to utilize vectorization. 

​	Another impediment to vectorization is if the memory regions on the the involved arrays have some overlap, which is named *array aliasing*,  which can result in incorrect execution. By default **the compiler will not vectorize** the code if it suspects that it is the case, and thus it is necessary to explicitly indicate that we **promise** not to feed aliased arrays to the function. This hint is done by the C99 keyword `restrict`, as in the `double* restrict x` declaration. 

​	Combining all this resources, we also moved the kernel constant outside the inner-most loop to lower the number of computations, and also pre-computed the inverse of the smoothing length outside, which is used to multiply the distance and already introduce the re-scaled distance as in `q = sqrt(q)*inv_h;` .  The resulting naïve function can be seen below:

```c
int compute_density_3d_naive_omp_simd(int N,double h,
                                      double* restrict x, double* restrict y,
                                      double* restrict z, double* restrict nu,
                                      double* restrict rho){
  const double inv_h = 1./h;                                // Pre-invert the smoothing distance 
  const double kernel_constant = w_bspline_3d_constant(h);  // Pre-compute the 3d normalization constant

  memset(rho,(int)0,N*sizeof(double));                      // Pre-initialize the density to zero

  #pragma omp parallel for                                  // Run the iteration in i in parallel 
  for(int64_t ii=0;ii<N;ii+=1){                             // Iterate over i
    double xii = x[ii];                                     // Load the position in X for ii
    double yii = y[ii];                                     // Load the position in Y for ii 
    double zii = z[ii];                                     // Load the position in Z for ii
    double rhoii=0.;
    
    #pragma omp simd                                        // Hint at the compiler to vectorize this loop
    for(int64_t jj=0;jj<N;jj+=1){                           // and iterate over the jj part of the block
      double q = 0.;                                        // initialize the distance variable

      double xij = xii-x[jj];                               // Load and subtract the position in X for jj
      double yij = yii-y[jj];                               // Load and subtract the position in Y for jj
      double zij = zii-z[jj];                               // Load and subtract the position in Z for jj

      q += xij*xij;                                         // Add the jj contribution to the ii distance in X
      q += yij*yij;                                         // Add the jj contribution to the ii distance in Y
      q += zij*zij;                                         // Add the jj contribution to the ii distance in Z

      q = sqrt(q)*inv_h;                                    // compute the normalized distance, measured in h

      rhoii += nu[jj]*w_bspline_3d_simd(q);                 // Add up the contribution from the jj particle
    }                                                       // to the intermediary density and then
    rho[ii] = kernel_constant*rhoii;                        // set the intermediary density to the full density
  }

  return 0;
}
```

​	Now, the results :

- AoS / Naive / OpenMP / -O3 / SIMD : $\sim\ 2.47\ \mbox{s} $
- AoS /  CLL  / OpenMP / -O3 / SIMD : $\sim\ 0.24\ \mbox{s} $
- SoA / Naive / OpenMP / -O3 / SIMD : $\sim\ 2.12\ \mbox{s}$
- SoA /  CLL  / OpenMP / -O3 / SIMD : $\sim\ 0.22\ \mbox{s}$

​	Now, that are some curious results. The naïve implementations presented an speedup of just under $2 \times$ , but the cell linked list had a much more modest speedup, around $25\%$. The speedup is real, and yet it seems a bit less than it should be. Looking at the compiler report for :

```
exec/example_03-ArrayOfStructs-Naive-Omp-SIMD.c:206:22: missed: couldn't vectorize loop
exec/example_03-ArrayOfStructs-Naive-Omp-SIMD.c:206:22: missed: not vectorized: control flow in loop.
exec/example_03-ArrayOfStructs-Naive-Omp-SIMD.c:215:28: missed: couldn't vectorize loop
exec/example_03-ArrayOfStructs-Naive-Omp-SIMD.c:215:28: missed: not vectorized: control flow in loop.
exec/example_03-ArrayOfStructs-Naive-Omp-SIMD.c:223:11: missed: statement clobbers memory: _80 = sqrt (q_50);

exec/example_03-ArrayOfStructs-Naive-Omp-SIMD.c:266:5: missed: couldn't vectorize loop
exec/example_03-ArrayOfStructs-Naive-Omp-SIMD.c:266:5: missed: not vectorized: control flow in loop.
exec/example_03-ArrayOfStructs-Naive-Omp-SIMD.c:261:8: missed: couldn't vectorize loop
exec/example_03-ArrayOfStructs-Naive-Omp-SIMD.c:261:8: missed: not vectorized: control flow in loop.
exec/example_03-ArrayOfStructs-Naive-Omp-SIMD.c:266:5: missed: couldn't vectorize loop
exec/example_03-ArrayOfStructs-Naive-Omp-SIMD.c:266:5: missed: not vectorized: control flow in loop.
exec/example_03-ArrayOfStructs-Naive-Omp-SIMD.c:261:8: missed: couldn't vectorize loop
exec/example_03-ArrayOfStructs-Naive-Omp-SIMD.c:261:8: missed: not vectorized: control flow in loop.
exec/example_03-ArrayOfStructs-Naive-Omp-SIMD.c:266:5: missed: couldn't vectorize loop
exec/example_03-ArrayOfStructs-Naive-Omp-SIMD.c:266:5: missed: not vectorized: control flow in loop.
exec/example_03-ArrayOfStructs-Naive-Omp-SIMD.c:261:8: missed: couldn't vectorize loop
exec/example_03-ArrayOfStructs-Naive-Omp-SIMD.c:261:8: missed: not vectorized: control flow in loop.
exec/example_03-ArrayOfStructs-Naive-Omp-SIMD.c:266:5: missed: couldn't vectorize loop
exec/example_03-ArrayOfStructs-Naive-Omp-SIMD.c:266:5: missed: not vectorized: control flow in loop.
exec/example_03-ArrayOfStructs-Naive-Omp-SIMD.c:261:8: missed: couldn't vectorize loop
exec/example_03-ArrayOfStructs-Naive-Omp-SIMD.c:261:8: missed: not vectorized: control flow in loop.
```

​	 In line 266, iIt is complaining about `266:5: missed: not vectorized: control flow in loop`, which refers to:

```c
  if(q<2.)
    wq = wq2;

  if(q<1.)
    wq = wq1;
```

​	Which interfere with the parallelization process. The compiler was able to vectorize the computations `q += xij*xij;` and `q = sqrt(q)*inv_h;`, but it still has problems with the kernel because of the conditionals. The exact reason for this limitation will be discussed down below, but it has to do with how the compiler chooses which hardware instructions to map your code to. Some ways to vectorize your code can cope with conditionals, others can't. So, to improve the present results, what we need to do is return to the compiler. 

​	The main lesson from this section is: **Adapting your code to ease the compiler's job is not easy, but it may be necessary.** This is probably one of the less tasteful parts of optimizing code for the traditional developer, but it is part of writing fast code. 

### Flexibilizing Compiler's Morals: `-ffast-math`

​	We observed some performance improvements by adapting the code to use SIMD by giving hints to the compiler. The next step is to tune the compiler options to exploit whatever compilations paths are available. The first problem is that the compiler is, if anything, overly cautious, choosing to error on the safe side and don't produce code that could not be **100% correct** unless otherwise stated. In our case, the problem is that **100% correct means bitwise correct**, and with floating points this limits a lot what compilers can do. 

​	The reason why this is limiting is related to the way [real numbers are represented in the computer](https://en.wikipedia.org/wiki/IEEE_754), which is a way that makes things like associativity and commutativity false in the general case. `-ffast-math` allows the compiler to treat variables **as if they were true real numbers**, but with the understanding that it could introduce small errors in the calculation, though unlikely that will actually occur. Because of this, this kind of optimization is **disabled by default**, and must be enabled intentionally. 

### Actually compiling for your CPU: `-march=native`

​	[Optimizing code can be strange.](https://stackoverflow.com/questions/19470873/why-does-gcc-generate-15-20-faster-code-if-i-optimize-for-size-instead-of-speed?rq=1) One of the reasons that optimizing can strange is because it is necessary to take into account the hardware that is actually going to run the hardware. **By default**, most compilers choose to generate code that run **best on an *average* CPU**, and **not best on *your* CPU.** This means that it has to take into account compatibility issues and whether a given instruction will be generally available or not. This compatibility issue happens to be present in the SIMD instruction generation we need to speedup the code. 

​	Modern CPUs can have a lot of moving parts, and quite a few quirks too when compared to "old" CPUs, in the 2000s and before. When you arrive in a modern day processor, you see things like the **absolute beast** shown below in the diagram of the architecture of my CPU, the Zen 2 micro-architecture. Each core have an overall structure described in the diagram, which have 4 [ALU](https://en.wikipedia.org/wiki/Arithmetic_logic_unit)s for computing up to 4 independent integer instructions per core per clock, and 3 [AGU](https://en.wikipedia.org/wiki/Address_generation_unit)s for computing memory addresses of data that needs to be used. Most of this stuff is hidden from both of the user and the developer, appearing a somewhat "free" performance gain ([at least most of the time](https://en.wikipedia.org/wiki/Branch_predictor)). 

​	Regarding the CPU, each core hash two floating point units, with 4 "pipes" each, supporting a "single-op AVX256". What this means is that the CPU can process those FMA3 instructions mentioned before, on 4 blocks of 64-bit floats. The deal is: Not all CPUs support AVX256, specially older CPUs. The first introduction of AVX256 to x86 processors was in the Intel Haswell micro-architecture, which is why AVX2 is sometimes referred as Haswell New Instructions. 

​	As mentioned, the compiler will produce by default the best code it can for the **broadest possible** range of CPU architectures, and since AVX2 is not that available, it **will not produce** AVX2 code **unless explicitly instructed**. What the compiler actually does is produce is code for an older set of of SIMD instructions [called SSE](https://en.wikipedia.org/wiki/Streaming_SIMD_Extensions), which are more widely available. SSE happen to be a SIMD standard that can process 2 streams of floats (called pipes), while AVX256 can process 4 streams simultaneously for potentially twice the performance. Contrary to AVX, SSE wasn't able to vectorize code with conditionals in the vectorization path, greatly limiting what was available for vectorization. Though not arbitraty, it is possible to vectorize code with some simplified conditionals, which happens to be enough to provide us with great performance gains. 

##### ![Zen2 Micro-architecture Anandtech](https://images.anandtech.com/doci/14525/Mike_Clark-Next_Horizon_Gaming-CPU_Architecture_06092019-page-003.jpg)

Extracted from [Anandtech](https://www.anandtech.com/show/14525/amd-zen-2-microarchitecture-analysis-ryzen-3000-and-epyc-rome/6).

​	To bypass this compiler behavior we must tell the compiler to target the CPU architecture we want, in our case we want the same that is being used to compile the code, i.e. `native`. This is done using the compiler flags `-march=native`, or a less strict `-mtune=native` . This allows the compiler to use the instructions you want without a guilty consciousness, because it was **inform** which cpu we **intend to run** this application in. 

​	The `-ffast-math` had some, though limited effect, but how did it perform alongside with `-O3` and `-march=native`?

- AoS / Naive / OpenMP / SIMD / -O3 -ffast-math -march=native : $\sim 4.4\ \mbox{s}$
- AoS /  CLL  / OpenMP / SIMD / -O3 -ffast-math -march=native : $\sim 0.27\ \mbox{s}$
- SoA / Naive / OpenMP / SIMD / -O3 -ffast-math -march=native : $\sim 0.75\ \mbox{s}$
- SoA /  CLL  / OpenMP / SIMD / -O3 -ffast-math -march=native : $\sim 0.16\ \mbox{s}$

​	These are some promising, if not puzzling, results. We actually observed a **regression** of performance with the AoS layout, which is unexpected to say the least since we do not expect to have **worse** performance by **enabling** further optimizations on the compiler. On the other hand, the SoA layout presented quite the **performance improvement** with the AVX2 SIMD enabling, with the Naive and CLL versions presenting a $2.6 \times$ and $1.3\times$ respectively. 

​	The over $2\times$ performance improvement for the SoA indicates that there is something bigger at play than simply the widening of the SIMD instruction, from $2$ to $4$ pipes. Part of the improvement is related to the `-O3` optimization flag, part of it has to do with the `-march=native`, which allows for the compiler to use SIMD instructions capable of coping with the `if(q<1)` branching that are available in AVX2 but not in SSE, and lastly it has somewhat to do with better cache utilization.

​	The main lesson from this section is: **Know your hardware, know your compiler, understand flags and use them correctly, not all flags are important, but the ones that are, are *really* important**.

## Naïve on Steroids: Loop Tiling and Cache Utilization

​	As of this moment, the naive implementation is both parallelized with openMP threads and vectorized with SIMD, which would suggests that is has reached the limit of what was available to it. The number of calculations hasn't changed, therefore it is still $\mathcal{O}(N^2)$, so we should expect a $100\times$ increase in the time for each $10\times$ increase in the size of the particle array. If we run for $10^5$ particles we take $\sim 0.75\ \mbox{s}$, but running for $10^6$ takes over $220\ \mbox{s}$! It was expected to take closer to $80 \mbox{s}$ to run this example, so why the dramatic decrease in performance?

​	As explained before, there is a big gap in memory performance and CPU performance. Under normal circumstances it is not possible for the main memory to feed enough data to the CPU to keep all hardware busy. The solution found to bridge this gap was to introduce a variety of intermediary memories **in between** the storage the CPU uses to perform calculations, called the **register**, and the **main system memory**. These memories are called caches, and are numbered according to how far away they are from the actual processing units, ALU and FPU, and therefore how large and slower they are. Level 1 cache (L1) is the fastest in-chip memory, and L3 is usually the slowest. They have varying sizes and speed, but they are always **smaller and faster** than main memory. Most of the time they are automatically managed by the CPU itself, but in some processors [they can be programmable](https://en.wikipedia.org/wiki/Scratchpad_memory), requiring you to program. The most widely know example is Nvidia Cuda's Shared Memory, which corresponds to a programmable L1 cache. 

​														![](https://d3i71xaburhd42.cloudfront.net/6ebec8701893a6770eb0e19a0d4a732852c86256/3-Figure2-1.png)

​	To a large extent, extracting the full potential performance nowadays means managing this **hierarchy of memory** in the way you code. Many of the more refined, arcane and confusing techniques for improving numerical code performance are focused exactly in improving cache utilization and re-use through emphasis in [data locality](https://en.wikipedia.org/wiki/Locality_of_reference), both spatially as discussed with the striding, and temporally with reuse of results. This overall philosophy is sometimes known as [Data-Oriented Programming/Design](https://en.wikipedia.org/wiki/Data-oriented_design), and it focus on exploiting a mixture of traditional task parallel and newer [data parallel](https://en.wikipedia.org/wiki/Data_parallelism) techniques to achieve the maximum available performance.

​	In our case, what happened in the first example was that the original data, of $10^5$ particles, actually **fit inside the cache**, allowing for the program to run with very high performance by exploiting not the bandwidth of the main memory, but rather the memory bandwidth of the cache. The way to recover the previous performance is to improve data reuse through a technique called [loop blocking or loop tiling](https://en.wikipedia.org/wiki/Loop_nest_optimization). 

​	The basic idea is rewriting a single `for(int64_t i=0;i<N;i+=1)` as two nested loops in which the original data is run through in "blocks" or "strips", and the loops corresponding to the blocks are joined together, which require an interchange between two of the new inner loops, justifying the name sometimes this technique is also referred as "strip mine and interchange". The resulting code is somewhat more complex than the original, despite doing exactly the same calculation, but is also performs considerably better:

```c
int compute_density_3d_naive_omp_simd_tiled(int N,double h,
                                            double* restrict x, double* restrict y,
                                            double* restrict z, double* restrict nu,
                                            double* restrict rho){
  const double inv_h = 1./h;                                       // Pre-invert the smoothing distance 
  const double kernel_constant = w_bspline_3d_constant(h);         // Pre-compute the 3d normalization constant
  const int64_t STRIP = 500;                                       // Setting the size of the strip or block 

  memset(rho,(int)0,N*sizeof(double));                             // Pre-initialize the density to zero

  #pragma omp parallel for                                         // Run the iteration in i in parallel
  for(int64_t i=0;i<N;i+=STRIP){                                   // Breaking up the i and j iterations in blocks
    for(int64_t j=0;j<N;j+=STRIP){                                 // of size STRIP to do data re-use and cache blocking
      for(int64_t ii=i;ii < ((i+STRIP<N)?(i+STRIP):N); ii+=1){     // Iterate a block over ii       
        double xii = x[ii];                                        // Load the position in X for ii
        double yii = y[ii];                                        // Load the position in Y for ii
        double zii = z[ii];                                        // Load the position in Z for ii
        double rhoii = 0.0;                                        // Initialize partial density ii to zero

        #pragma omp simd                                           // Hint at the compiler to vectorize this loop
        for(int64_t jj=j;jj < ((j+STRIP<N)?(j+STRIP):N); jj+=1 ){  // and iterate over the jj part of the block
          double q = 0.;                                           // initialize the distance variable

          double xij = xii-x[jj];                                  // Load and subtract jj particle's X position component
          double yij = yii-y[jj];                                  // Load and subtract jj particle's Y position component
          double zij = zii-z[jj];                                  // Load and subtract jj particle's Z position component

          q += xij*xij;                                            // Add the jj contribution to the ii distance in X
          q += yij*yij;                                            // Add the jj contribution to the ii distance in Y
          q += zij*zij;                                            // Add the jj contribution to the ii distance in Z

          q = sqrt(q)*inv_h;                                       // Sqrt and normalizing the distance by the smoothing lengh

          rhoii += nu[jj]*w_bspline_3d_simd(q);                    // Add up the contribution from the jj particle
        }                                                          // to the intermediary density and then
        rho[ii] += kernel_constant*rhoii;                          // add the intermediary density to the full density
      } 
    }
  }

  return 0;
}
```

​	The new timing is around $81\ \mbox{s}$, which is not only considerably faster than the $220\ \mbox{s}$ but also in line with the expected $\mathcal{O}(N^2)$ scaling in time for the double loop, recovering the originally expected performance. To understand what is going on and why adding "apparently more complex" loops actually improves performance instead of slowing down code, we turn to explore how the CPU performs calculation and the memory requirements for those calculations. 

​	As mentioned in the previous part of this series, the Ryzen 3900x CPU has two [SIMD units](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#Advanced_Vector_Extensions_2) per core units which are capable of executing a [3 operand fused multi-add](https://en.wikipedia.org/wiki/FMA_instruction_set#Instructions) (FMA3) instruction per clock. This means each CPU core can, theoretically, perform 4 additions and 4 multiplicaions in different `a`, `b` and `c` data *every **single** cycle*. In our case, this can be estimated as being $12\ \mbox{cores} \times 4.0\ \mbox{GHz}\ \times 2\ \mbox{units/core} \times 2\ \mbox{ops/FMA}\times 4\ \mbox{FMA/unit}\approx 768\ \mbox{GFlops}$. It looks big, but the requirements to do so are not big, they are almost *obscene*. 

​	To perform this computation, it needs to load 3 double precision floating point numbers from memory for each operation, therefore it requires $12\ \mbox{cores}\ \times 2\ \mbox{units/core}\times 4\ \mbox{pipes/unit}\ \times 3\ \mbox{double/pipe} \times 8\ \mbox{bytes/double}\times 4.0\ \mbox{GHz}\approx 9.2\ \mbox{TB/s}$ if they all were to run all units every single cyle, which is quite the astounding number in case you are not used to seeing these bandwidth figures. As a comparison, my main system memory can only provide $45-50\ \mbox{GB/s}$  of bandwidth, so there is no way it would be possible to provide all the data needed to keep this many operations pulling data just from main memory, it is **simply not fast enough**. For comparison, a fast consumer grade SSD can do $\sim 3-7\ \mbox{GB/s}$ of bandwidth and an old hard-drive can only provide $0.125 - 0.250\ \mbox{GB/s}$. And yet, AMD spent a lot of money to develop it the silicon to perform it and I paid a fair amount myself to purchase a CPU that has it, therefore the must be a way to use it, and **indeed there is**.

​	The way to do it is through the **fast in-chip memory** named cache(s). The closest cache to the computing units, the L1 cache, is able to transfer 32 bytes/cycle to the [registers](https://en.wikipedia.org/wiki/Processor_register), which are where the data is needed to perform the operations. Computing the aggregate memory bandwidth provided by the L1 cache from all 12 cores, the **potential aggregate L1 bandwidth** adds to $12 \times 32\ \mbox{B/(cycle * port)} \times 2\ \mbox{port} \times\ 4.0\ \mbox{GHz} \approx 3.072\ \mbox{TB/s}$, which is not quite yet what we would theoretically need to fully execute a FMA3 instructions in the CPU every cycle, but it is not difficult to see how data re-use could help bridge the last $3 \times$ factor, which **would be the case** in many applications such as ours because there is a natural data re-use in the form of the reduction inherent to the calculation of the distance, and it is also present in the reduction used in the density calculation, which is essentially a reduction computation for each particle. In this sense, the cache don't need to provide the bandwidth for **all** FMA3 computations, but only for the **new** data needed to perform additional computations. 

​	The way that is translated to the code is by **re-ordering the operations** in the code, without actually doing any new operations. The change of order allow for the CPU to re-use data stored in cache in a calculation that is performed **not long after** the one that fetched the original data. By iterating over data stripes, each stripe's data is re-used by re-using indexes in a given stripe, e.g. `for(int64_t ii=i;ii < ((i+STRIP<N)?(i+STRIP):N); ii+=1)` loop uses data of `x[ii]` between `i` and `i+STRIP`, and the same for `jj`. 

​	The CPU ordinarily would get the data for `x[ii]` from the main memory but, since this data was requested in a previous iteration, it already resides inside the CPU cache and it can be re-used. For $N\sim 10^5$ , the data is so small that it is possible to simply store the full array in cache, thus making so that the tiling technique is somewhat redundant for this small case. 

​	The main lesson from this section is: **Memory bandwidth is a finite resource and imposes real constraints for performance. Understanding the memory hierarchy of your particular hardware does make a difference, and this means that writing your program considering temporal locality of your data is Important**.
