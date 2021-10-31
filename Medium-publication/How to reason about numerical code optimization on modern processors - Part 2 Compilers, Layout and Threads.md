# How to reason about numerical code optimization on modern processors - Part 2 : Compilers, Data Layout and Threads

​	In the previous article of this series, we discussed the physical problem we are interested in and how to translate the mathematical concepts that are involved in modeling it into computer code. We also discussed an important resource to enable realistic simulations to be performed called the Cell Linked List, which lowers the overall computational cost by skipping a substantial fraction of computations, known a priori to be useless. 

​	Another point that became clear in the previous article is that an **efficient** implementation of numerical algorithms is not simply a thoughtless translation of mathematical concepts into code, requiring taking into account several avenues on how to convey "material reality" (or rather "digital reality") to a given abstraction, in which some avenues are clearly better then others. 

​	Now, we turn our attention to more concrete forms of optimization: how to use the compiler, how to lay down data in the system memory and how to execute computations of a for loop in parallel using threads. Most of the content presented in this section is, at least partially, based on the course [Fundamentals of Parallelism on Intel Architecture](https://www.coursera.org/learn/parallelism-ia), lectured by Dr. Valdmirov from Colfax International. It is a great introductory course that covers many of the most important techniques for code optimization, specially the "implementation" type optimization involving the topics discussed hereon. 

​	As said, algorithm type optimizations are important because they reduce the cost by cutting down unnecessary work or approximating efficiently what would otherwise be incredibly expensive to compute exactly. Implementation type optimizations are important because no amount of cheating will lower the cost beyond a certain level, and thus for the code to run faster it is necessary to not "**leave performance in the table**" by under-utilizing the CPU. If you prefer, think this way: If you could make your complete in $1/3$ of the time **without too much hassle**, and you don't, you are effectively **paying full price for a third of the performance**, and most people wouldn't accept a Corolla driving at 11 MPG just because the mechanic forgot to enable the electronic injection correctly. 

​	Similarly as done with the previous part, we present a summary of the results of this part in comparison with the previous one:

|          Algorithm / Implementation / Configuration          |  Time (Seconds)   | Speedup (Rel. to slowest) |
| :----------------------------------------------------------: | :---------------: | :-----------------------: |
| Naive Calculation (i.e. direct two loop) / AoS, simple, no optimizations / gcc -std=c11 -Wall |  188.46 +- 0.71   |          **1 x**          |
| Cell Linked List / AoS, simple, no optimizations / gcc -std=c11 -Wall |  3.642  +- 0.043  |         **51 x**          |
| Naive Calculation (i.e. direct two loop) / AoS, simple, no optimizations / gcc -std=c11 -Wall -O3 |   50.19 +- 0.61   |        **3.75 x**         |
| Cell Linked List / AoS, simple, no optimizations / gcc -std=c11 -Wall -O3 |  1.572  +- 0.002  |         **119 x**         |
| Naive Calculation (i.e. direct two loop) / SoA, single thread / gcc -std=c11 -Wall -O3 |  44.832 +- 0.205  |        **4.20 x**         |
| Cell Linked List / SoA, single thread / gcc -std=c11 -Wall -O3 | 1.5722  +- 0.0072 |         **119 x**         |
| Naive Calculation (i.e. direct two loop) / SoA, OpenMP / gcc -std=c11 -Wall -O3 |  4.267 +- 0.049   |         **44 x**          |
|   Cell Linked List / SoA, OpenMP / gcc -std=c11 -Wall -O3    |  0.313 +- 0.033   |         **600 x**         |

## Understanding the code: Benchmarking

​	A key step to improve our code performance is actually measuring the time the code is spending in performing each task. The naive spends all the time in one block of code, so that is not very informative, but the cell linked list version is more interesting: It is broken in 4 steps: `compute_hash_MC3D` , `qsort`, `setup_interval_hashtables` and `compute_density_3d`. 

​	Four functions could mean that there are are a lot of places where performance gains can be obtained, right? The only way to find out is benchmarking the code and see what happens. The results for the cell linked list version can be seen below:

```bash
compute_hash_MC3D         : 0.0018656 +- 2.800536e-05 :   0.05122% +- 0.0007689%
sorting                   : 0.0144988 +- 0.0009401131 :   0.3981%  +- 0.02581%
setup_interval_hashtables : 0.0003818 +- 3.482384e-05 :   0.01048% +- 0.0009562%
compute_density           : 3.625306  +- 0.04295192   :  99.54%    +- 1.179%
Total Time                : 3.642053  +- 0.04310305   : 100%       +- 1.183%
```

​	So, despite having several moving parts the code spends almost $99\%$ of the time in a single part of the code. It is common to call this parts, which make the code spend almost all of its time in, "hot spots". This hot spot is exactly where the actual density calculation is being performed, which surprises no-one. A distant second place of relevance is the sorting operation, `qsort`, with $< 1\%$ of the execution time. This results shows that the only region of the code worth bothering with, right now, is `compute_density_3d`. 

​	To understand why, lets look back to the `compute_density_3d` function:

```c
int compute_density_3d_cll(int N, double h, SPHparticle *lsph, linkedListBox *box){
  khiter_t kbegin,kend;
  int64_t node_hash=-1,node_begin=0, node_end=0;  
  int64_t nb_begin= 0, nb_end = 0; 
  int64_t nblist[(2*box->width+1)*(2*box->width+1)*(2*box->width+1)];

  // Iterate over each receiver cell begin index 
  for (kbegin = kh_begin(box->hbegin); 
       kbegin != kh_end(box->hbegin); kbegin++){  
    if (kh_exist(box->hbegin, kbegin)){  // verify if iterator exists
      // Then get the end of the receiver cell iterator
      kend = kh_get(1, box->hend, kh_key(box->hbegin, kbegin));    
      // Then get the hash corresponding to it
      node_hash = kh_key(box->hbegin, kbegin);                     
      // Get the receiver cell begin index in the array
      node_begin = kh_value(box->hbegin, kbegin);                  
      // Get the receiver cell end index in the array
      node_end   = kh_value(box->hend, kend);  

      for(int64_t ii=node_begin;ii<node_end;ii+=1)
        lsph[ii].rho = 0.0;                       

      // then find the hashes of its neighbors 
      neighbour_hash_3d(node_hash,nblist,box->width,box);   
      // and the iterate over them
      for(int j=0;j<(2*box->width+1)*(2*box->width+1)*(2*box->width+1);j+=1){    
        if(nblist[j]>=0){ // if a given neighbor actually has particles
          // then get the contributing cell begin and end indexes
          nb_begin = kh_value(box->hbegin, kh_get(0, box->hbegin, nblist[j]) );  
          nb_end   = kh_value(box->hend  , kh_get(1, box->hend  , nblist[j]) );  

          // and compute the density contribution from 
          compute_density_3d_chunk(node_begin,node_end,nb_begin,nb_end,h,lsph);  
        }}}} // the contributing cell to the receiver cell
    
  return 0;
}
```

​	It iterates over the range of cells, for each cell it iterates over its neighbors, and then calls `compute_density_3d_chunk`. Whatever is the cost of `compute_density_3d_chunk`, it will be multiplied over $>21000 \times$ according to what we have seen in the end of part 1, therefore it is where we should be paying attention to. Inside `compute_density_3d_chunk` there is the following code.

```C
int compute_density_3d_chunk(int64_t node_begin, int64_t node_end,
                             int64_t nb_begin, int64_t nb_end,double h,
                             SPHparticle *lsph)
{
  const double inv_h = 1./h;
  const double kernel_constant = w_bspline_3d_constant(h);

  for(int64_t ii=node_begin;ii<node_end;ii+=1){
    double xii = lsph[ii].r.x;
    double yii = lsph[ii].r.y;
    double zii = lsph[ii].r.z;
    double rhoii = 0.0;
   
    for(int64_t jj=nb_begin;jj<nb_end;jj+=1){
      double q = 0.;

      double xij = xii-lsph[jj].r.x; // 1 load (lsph[jj].r.x) and 1 operation (-)
      double yij = yii-lsph[jj].r.y; // 1 load (lsph[jj].r.y) and 1 operation (-)
      double zij = zii-lsph[jj].r.z; // 1 load (lsph[jj].r.z) and 1 operation (-)

      q += xij*xij; // 0 loads and 2 operations (+ and *) 
      q += yij*yij; // 0 loads and 2 operations (+ and *)
      q += zij*zij; // 0 loads and 2 operations (+ and *) 

      q = sqrt(q)*inv_h; // 0 loads and 2 operations (sqrt and *)
	
      // 1 load (lsph[jj].nu) and 2+ operations (+, *, and what is in w_bspline_3d)
      // Checking w_bspline_3d, there are 22 operations inside, plus conditionals
      rhoii += lsph[jj].nu*w_bspline_3d(q,h); 
    }
    lsph[ii].rho = kernel_constant*rhoii;
  }

  return 0;
}
```

​	We load `node_end-node_begin` elements `lsph[ii].r` , `nb_end-nb_begin` elements `lsph[jj].r` and `lsph[jj].nu` (though this is done repeatedly), and we perform `(node_end-node_begin)*(nb_end-nb_begin)` distance and kernel calculations. For a feeling of how large this is, lets make some estimates: There are $10^5$ particles uniformly distributed between $10\times 10\times 10 = 1000$ boxes, therefore both `node_end-node_begin` and `nb_end-nb_begin` are around $\approx 10^5/1000 = 100$ particles per box. 

​	So, each call read around 300 data from `ii` for loop,  400 data from `jj` for loop, and perform $100\times 100 = 10^4$ batches of arithmetic operations to compute the contribution to the density. For each batch, there are $3\times 1+5 \times 2 + 22 = 35$ operations in each call of the inner-most loop, which sums to $3.5 \times 10^5$  arithmetic operations, or **over three hundred thousand operations**, every single time `compute_density_3d_chunk` is called. For the entire execution, there are $21232$ calls to `compute_density_3d_chunk`, totaling $21232 \times 3.5 \times 10^5 \approx 7.4 \times 10^{9}$   operations for this very simple example, which means that even a very small case can require **almost 10 Billion arithmetic operations** to be performed. This is a very large number of operations, and overwhelms pretty much anything else going on in the rest of the code and, for this reason, regions like these are called **critical sections**. 

​	**Finding and improving critical sections** of your code is a major part of optimizing numerical codes, and improving just them respond for the majority of performance gains. This means that, *before* the critical regions are optimized, it is usually not worth bothering with the rest of the code, in what concerns performance. For these parts, which are outside the critical section, the more conventional recommendations about code development applies: Readability trumps smarts, maintainability trumps performance. Later we will discuss more how these improvements in the critical section can be done. 

​	The main lesson we gain from this is: **Measure, measure and measure**. Guessing usually don't get you anywhere, and actually timing your code and code sections is important. Also, **critical sections are called "critical" for a reason, pay attention to them**.

## First optimization : Compiler Options -O2 and -O3

​	The first optimization most programmers will probably will have heard about is compiling with "Optimization Flags". In my case, `gcc` offers some preset compiler flags that group several optimization options together, and they are usually referred to by their numbers: `-O0`, `-O1`, `-O2` and `-O3` . Their description can be found in in [GCC reference for optimize options](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html#Optimize-Options).  These numbered optimization flags combine several optimization options, which would otherwise be cumbersome or unwieldy to remember, understand or use separately, and are a great starting point to make the code run faster. (And if you are wondering if I understand all the options bunched together in each of these flags, I don't. )

​	According to the reference `-O0` reduce compilation time and make debugging produce the expected results, effectively disabling optimizations, and it is the **default option** for the `gcc`, which is a bad news for any newcomer numerical programmer. This means that, unless you specify otherwise, the compiler will produce code that is focused on debugging and **not on performance**. While this is great for system programmers, this means that numerical programmers must be aware that the default option is not necessary going to produce satisfactory results. 

​	The community wisdom is that `-O2` is considered a "safe" start and is a very commonly used optimization option, and was the first option I tried. So, how did our two examples from the previous article performed with this flag? 

- The naïve implementation came down from $\sim 180\ \mbox{s}$ to around $\sim 50\ \mbox{s}$ , representing an speedup of around $3.7 \times$ ! 
- The Cell Linked List implementation came down from $\sim 3.5\ \mbox{s}$ to around $\sim 1.6\ \mbox{s}$, also representing a $\sim 2.2\times$ speedup!

​	Therefore, by just adding 3 characters to the Makefile, we were able to gain over a ${\mathbf 2\times}$ speedup over the simple compilation without modifying any of the existing code (Time to call the mechanic to complain). 

​	The main lesson we gain from this is: **Knowing how to use the compiler matters** . 

​	Now, how does our code perform by using `-O3`? The naive version did not perform any better, staying $\sim 50\ \mbox{s}$ , nor did the Cell Linked List version. This means that, though somewhat automated, using the compiler to speedup code is not a fully automatic endeavor, and a fair share of the rest of this series will be dedicated to structure your code in ways that make it **easier** for the compiler to optimize the code. Though this is not hassle free, if it is done soon enough in the development process it can be fairly painless, though adapting it afterwards can be a headache in many cases. 

## Data Layout : Array of Structs vs Struct of Arrays

​	One possible avenue of improvement is changing the way our data is laid out in memory, and how it is accessed. The original structure that held a particle data looked like the following:

```C
typedef struct double4 {
    double x;
    double y;
    double z;
    double t;
} double4; // defines a "vector" of 4 components: x, y, z and t

typedef struct SPHparticle
{
    double4 r,u,F;   // position, velocity and force data 
	int64_t id,hash; // unique-ID, hash is the index of the cell 
	double nu,rho;  // mass of the particle, rho is density at the particle position
} SPHparticle; 
```

​	As mentioned, the problem with such data structure is that you don't have sequential access of the data of interest in the inner most loop:

```C
	for(int64_t jj=0;jj<N;jj+=1){   // Run over every other particle
      double dist = 0.;             // initialize the distance and add 
                                    // the contributions from each direction
      dist += (lsph[ii].r.x-lsph[jj].r.x)*(lsph[ii].r.x-lsph[jj].r.x);
      dist += (lsph[ii].r.y-lsph[jj].r.y)*(lsph[ii].r.y-lsph[jj].r.y);
      dist += (lsph[ii].r.z-lsph[jj].r.z)*(lsph[ii].r.z-lsph[jj].r.z);

      dist = sqrt(dist);            // take the sqrt to have the distance
                                    // and add the contribution to density
      lsph[ii].rho += lsph[jj].nu*w_bspline_3d(dist,h);
    }
```

​	The distance between `lsph[jj].r.x` and `lsph[jj+1].r.x` is `size(SPHparticle)`, which is around 128 bytes, and not `size(double)` which is 8 bytes. This means that even if you are not using the fields `u`, `F` or `hash`, you still need to load them from memory, which means that the effective memory bandwidth that is available for this loop is, at most, `4*size(double)/size(SPHparticle) = 0.25`  which is $1/4$ of the maximum bandwidth available. This means that, unless you use **all struct fields** in the same inner loop, storing data this way will **make you pay for data you don't want with bandwidth you need**. 

​	The way of fixing your data layout receive the name of [Array of Structs to Struct of Arrays](https://en.wikipedia.org/wiki/AoS_and_SoA) transformation, which goes by the acronym AoS to SoA. In our case, instead of doing one big allocation of the entire complex struct, the new "struct" looks like this:

```c
typedef struct SPHparticle{
  int64_t *id,*hash;
  double *nu,*rho;
  double *x,*y,*z;
  double *ux,*uy,*uz;
  double *Fx,*Fy,*Fz;
} SPHparticle;
```

​	The code itself actually changes less than one would expect considering such drastic change in the main data structure, with the Box structure remaining unchanged. The allocation becomes several smaller allocations of the homogeneous simple types for each array, and reordering now proceeds in two steps: First  the sort occurs in an array that contains hashes and indexes, and is **sorted accorded to the hashes**. Then, the new indexes are used to reorder the remainder of the arrays. If you are worried that this is going to be expensive, don't worry, it is actually pretty cheap as will become evident later. 

​	For example, the naïve density calculation looks like this with the Structure of Arrays (SoA) layout:

```C
int compute_density_3d_naive(int N,double h,SPHparticle *lsph){

  for(int64_t ii=0;ii<N;ii+=1){     // For every particle 
    lsph[ii].rho = 0;               // initialize the density to zero
    for(int64_t jj=0;jj<N;jj+=1){   // Run over every other particle
      double dist = 0.;             // initialize the distance and add 
                                    // the contributions from each direction
      dist += (lsph[ii].r.x-lsph[jj].r.x)*(lsph[ii].r.x-lsph[jj].r.x);
      dist += (lsph[ii].r.y-lsph[jj].r.y)*(lsph[ii].r.y-lsph[jj].r.y);
      dist += (lsph[ii].r.z-lsph[jj].r.z)*(lsph[ii].r.z-lsph[jj].r.z);

      dist = sqrt(dist);            // take the sqrt to have the distance
                                    // and add the contribution to density
      lsph[ii].rho += lsph[jj].nu*w_bspline_3d(dist,h);
    }
  }
  return 0;
}
```

​	With the components being fed separately to the function call. This is not terribly different from before, but the most important question: How does it perform?

- SoA + Naïve + -O2 : $\sim 45\ \mbox{s}$
- SoA + Naïve + -O3 : $\sim 45\ \mbox{s}$
- SoA + CLL   + -O2 : $\sim 1.56\ \mbox{s}$
- SoA + CLL   + -O3 : $\sim 1.57\ \mbox{s}$

​	Though not substantial, between $5\%$ and $15\%$, there was some gain which is positive for the naive case. The result for the Cell Linked list it somewhat undefined. Nevertheless, it will be later seen that this transformation will enable other, more effective, optimizations to be realized. Nevertheless, small gains are still gains!

​	The main lesson from this section is: **How to map your data to your the computer memory, i..e data layout, matter**. 

​	If you are a **Python only programmer** and never written a malloc call in your life, well, I can't say you are missing too much but you might be feeling a bit lost on how this is related to your life, let me tell you: **This is what numpy does** underneath the hood. If you are a sensible person and use only reasonable data types in your numpy arrays, the actual data is laid out in a SoA fashion in memory, sequentially and in such a way to map contiguous values in the inner-most index to continuous locations in memory. 

​	If you ever wondered what the `order` argument in the [`numpy.array`](https://numpy.org/doc/stable/reference/generated/numpy.array.html) function meant, it refers to which index in the nd-array notation corresponds to the inner-most in the actual data layout, and there are two factions: The C-like faction which corresponds to the [row major](https://en.wikipedia.org/wiki/Row-_and_column-major_order), and the Fortran-like faction which corresponds to the column major layout. Of course, row major is best and is the default option for `numpy.array`. 

​	The equivalent python code would be akin to the one below. The inner-loop is suppressed by working with numpy arrays:

```python
for i in range(rho.shape[0]):
    dist  = np.zeros(X.shape[0])
    dist += (X[i] - X)**2
    dist += (Y[i] - Y)**2
    dist += (Z[i] - Z)**2
    dist  = np.sqrt(dist)/h
    rho[i] += np.sum( nu * w_bspline_3d(dist) )
```

## Transcending single core: openMP Thread Parallelism 

​	The next step I took to improve performance was to utilize more resources and available in our present, serial, code. Modern computers come with CPUs that have multiple "mini-CPUs" called cores, which are functional units capable of executing code. As such, they are able to perform several tasks simultaneously and if used correctly, they can be used to speedup the overall code execution. A diagram of a part of my CPU can be found below. The regions labeled "Core X" designate the regions that have units capable of performing computations. 

![](https://en.wikichip.org/w/images/thumb/a/a7/amd_zen_ccx_2_(annotated).png/700px-amd_zen_ccx_2_(annotated).png)

​	Unfortunately, it is not possible for the compiler figure out by itself how to optimize our code, so we have to instruct it on how to do it in a processes known as **parallelizing your code**. The we way I chose to parallelize was with a [library called **openMP**](https://en.wikipedia.org/wiki/OpenMP), Open Multi-Processing, which is a way to parallelize your code within the same computer, also called "shared-memory multiprocessing". The library it comprised of a series of functions, such as found in any standard library, and specific instructions for the compiler to parallelize certain parts of the code called **compiler directives**. 

​	Most of our additions for computing the density read from many particles but only write to one particle's result at each index, as such they are **independent**. OpenMP allow us to parallelize independent loops using the compiler directive `#pragma omp parallel for`. The example loop from the naive version become as below:

```c
int compute_density_3d_naive_omp(int N,double h,SPHparticle *lsph){

  #pragma omp parallel for          // Execute in parallel
  for(int64_t ii=0;ii<N;ii+=1){     // For every particle 
    lsph[ii].rho = 0;               // initialize the density to zero
    for(int64_t jj=0;jj<N;jj+=1){   // Run over every other particle
      double dist = 0.;             // initialize the distance and add 
                                    // the contributions from each direction
      dist += (lsph[ii].r.x-lsph[jj].r.x)*(lsph[ii].r.x-lsph[jj].r.x);
      dist += (lsph[ii].r.y-lsph[jj].r.y)*(lsph[ii].r.y-lsph[jj].r.y);
      dist += (lsph[ii].r.z-lsph[jj].r.z)*(lsph[ii].r.z-lsph[jj].r.z);

      dist = sqrt(dist);            // take the sqrt to have the distance
                                    // and add the contribution to density
      lsph[ii].rho += lsph[jj].nu*w_bspline_3d(dist,h);
    }
  }

  return 0;
}
```

​	The directive `#pragma omp parallel for` tells the compiler to create instructions that compute the indexes of the first loop, `i` index, sending the body of the loop to different CPU cores, therefore speeding up the computation by executing different and independent parts of the result **simultaneously**. After instructing the compiler to send the computation of different indexes in different cores, what the program itself will do is: 

1. The program will start executing serially, it will do so till it reach a section of the code marked `#pragma omp parallel`
2. Once reached, the code will launch different **instruction streams** to feed the CPU cores, called **threads**. 
3. Each thread receive access to all variables that were available to the original code, shared with all other threads, and any variable that is declared within the `{}` is only available to that thread, henceforth called private. If a thread modifies a shared variable, all threads see the modification, but if modifies a private variable only it sees a change, since each thread has its own independent copy of it. 
4. After everything executed inside the block, for **all indexes**, all threads "join" and the code return to being a serial code. 

​	OpenMP offers the possibility of a relatively simple way to parallelize code without too much modification, by indicating which loops should operate in parallel and not to pay too much attention on the rest. This actually works pretty well for many numerical codes because most of the computational time is usually localized in a few very important loops, even if the code is fairly complex.  To enable openMP functionality, it is necessary to utilize the `-fopenmp` flag in `gcc` as compiler flag and also as a linker flag. For the equivalent Cell Linked List version of the code, we parallelized the internal function call works out to be like the following:

```C
int compute_density_3d_chunk(int64_t node_begin, int64_t node_end,
                             int64_t nb_begin, int64_t nb_end,double h,
                             SPHparticle *lsph)
{
  #pragma omp parallel for                      // Execute in parallel 
  for(int64_t ii=node_begin;ii<node_end;ii+=1){ 
    double xii = lsph[ii].r.x; // Load the X component of the ii particle position
    double yii = lsph[ii].r.y; // Load the Y component of the ii particle position
    double zii = lsph[ii].r.z; // Load the Z component of the ii particle position
    double rhoii = 0.0;        // Initialize the chunk contribution to density 
   
    // Iterate over the each other particle in jj loop
    for(int64_t jj=nb_begin;jj<nb_end;jj+=1){   
      double q = 0.;                  // Initialize the distance

      double xij = xii-lsph[jj].r.x; // Load and subtract jj X position 
      double yij = yii-lsph[jj].r.y; // Load and subtract jj Y position 
      double zij = zii-lsph[jj].r.z; // Load and subtract jj Z position 

      q += xij*xij;     // Add the jj contribution to the ii distance in X
      q += yij*yij;     // Add the jj contribution to the ii distance in Y
      q += zij*zij;     // Add the jj contribution to the ii distance in Z

      q = sqrt(q);      // Sqrt to compute the distance

      rhoii += lsph[jj].nu*w_bspline_3d(q,h);   // Add up jj contribution 
    }                                           // to the intermediary density 
    lsph[ii].rho += rhoii;                      // then add to the full density
  }

  return 0;
}
```

​	So, scoring time, how did the applications perform?

- AoS / Naïve / OpenMP / -O3 : $\sim\ 4.26\ \mbox{s}$
- AoS /  CLL  / OpenMP / -O3 : $\sim\ 0.313\ \mbox{s}$
- SoA / Naïve / OpenMP / -O3 : $\sim\ 4.04\ \mbox{s}$
- SoA /  CLL  / OpenMP / -O3 : $\sim\ 0.275\ \mbox{s}$

​	So, now we have some very interesting results. For the AoS data layout we went down from  $50\ \mbox{s}$ and $1.57\ \mbox{s}$ **down to** $4.26\ \mbox{s}$ and $0.313\ \mbox{s}$, this is equivalent to an speedup of $11.73\times$ and $5 \times$ respectively! This is quite an impressive result, but it is not the end of it.

​	For the SoA layout, the respective speedups are $11.13 \times$ and $5.75\times$ . The $11.75\times$ should not be too surprising since I am running this code on a 12 core CPU, the Ryzen 3900x, with a fairly easily parallelizible code, so a close to linear speedup is somewhat expected. The fail of the CLL case to scale as much as the naive counterpart is an indication of a suboptimal parallelization strategy for the cell linked list case, a point which will be explored in more detail in Part 4. The reason for this shortcoming is too difficult to see: Function `compute_density_3d_chunk` calculates the density contribution to `node` cell from `nb` cell, and it is called inside two loops iterating over each cell pair of a given cell. 

​	To see why this is a problem, note that the `#pragma omp parallel for` directive is called over and over again. Each time `#pragma omp parallel for` , there is a cost in launching the threads, and joining them at the end of the block, and threads are not kept alive from one launch to the other. This has a cost, and turns out that launching threads is actually quite expensive, and piles up quickly if it is called up too many times. This opens up an avenue of improvement that will be discussed in Part 4: moving the openMP call up the nested `for`s. 

​	The main lesson from this section is: **Utilizing the available cores matter** and **data layout matter (again)**. 

​	For our **Python-only and Python-first readers**: you might recognize the pattern we are using as a parallelization of a list of indexes through the use of a [thread-pool to execute our loop](https://docs.python.org/3.8/library/multiprocessing.html#multiprocessing.pool.Pool.map) by running a parallel map over the range of indexes, if that is the case you are absolutely right! This would be the **natural** way to perform the equivalent parallelization in python, though it might be very memory intensive to do so because of the way `pool.map` is implemented. In case you actually liked the openMP way of doing parallelism, there is a library to mimic openMP-sytle parallel `for` and similar to python called [pyMP](https://github.com/classner/pymp), though it does come with similar caveats to `Multiprocessing.Pool` and OpenMP. 
