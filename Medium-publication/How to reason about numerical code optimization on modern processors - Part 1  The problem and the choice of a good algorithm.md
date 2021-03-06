# How to reason about numerical code optimization on modern processors - Part 1 : The problem and the choice of a good algorithm

​	This article is for anyone that would like to know how to reason about the performance of their numerical code, and how to improve it. To do so I would like to invite the reader, to follow me through the steps I went through to optimize a single computation: the computation of the density of a particle system in the Smoothed Particle Hydrodynamics (SPH) method. In part 1 would like to present the problem, the naïve implementation and a very common optimization used in the SPH literature called the Cell Linked List method. 	

​	For most developers, most of the complexity involved in implementing fast and efficient algorithms is hidden behind frameworks and libraries, such as Numpy, Pandas, Tensorflow or even mapReduce. Nevertheless, for any problem bigger than prototype, it is important to know how to correctly use those libraries and understand which premises they are built upon, so as not to slow down your code unnecessarily. 

​	This series' topics are divided in four parts: In the first part we will cover the problem we wish to solve, present its most basic implementation and discuss an algorithm to speed it up by skipping a majority of unnecessary computations in advance, using the Cell Linked List method. This will allow us to overview some of the major concepts related to accelerating the numerical code, and focus to on the importance of beginning by choosing good algorithms as a necessary step to implementing efficient numerical codes. 

​    As a preview of the remainder of the series, the timings of all cases from parts 1 through 4 follows below:

|     Part 1 : Algorithm / Implementation / Configuration      | Time (Seconds)  | Speedup (Rel. to slowest) |
| :----------------------------------------------------------: | :-------------: | :-----------------------: |
| Naive Calculation (i.e. direct two loop) / AoS, simple, no optimizations / gcc -std=c11 -Wall | 188.46 +- 0.71  |          **1 x**          |
| Cell Linked List / AoS, simple, no optimizations / gcc -std=c11 -Wall | 3.642  +- 0.043 |         **51 x**          |

| Part 2 (Compilers, Data Layout and Threads) : Algorithm / Implementation / Configuration |  Time (Seconds)  | Speedup (Rel. to slowest) |
| :----------------------------------------------------------: | :--------------: | :-----------------------: |
| Naive Calculation (i.e. direct two loop) / AoS, simple, no optimizations / gcc -std=c11 -Wall -O3 |  50.19 +- 0.61   |        **3.75 x**         |
| Cell Linked List / AoS, simple, no optimizations / gcc -std=c11 -Wall -O3 | 1.572  +- 0.002  |         **119 x**         |
| Naive Calculation (i.e. direct two loop) / SoA, single thread / gcc -std=c11 -Wall -O3 |  44.83 +- 0.20   |        **4.20 x**         |
| Cell Linked List / SoA, single thread / gcc -std=c11 -Wall -O3 | 1.572  +- 0.0072 |         **119 x**         |
| Naive Calculation (i.e. direct two loop) / SoA, OpenMP / gcc -std=c11 -Wall -O3 | 4.0413 +- 0.0087 |         **46 x**          |
| Cell Linked List / SoA, Inner OpenMP / gcc -std=c11 -Wall -O3 |  0.275 +- 0.027  |         **685 x**         |

| Part 3 (SIMD, Architecture and Strip Mining) : Algorithm / Implementation / Configuration |  Time (Seconds)  | Speedup (Rel. to slowest) |
| :----------------------------------------------------------: | :--------------: | :-----------------------: |
| Naive Calculation / AoS, OpenMP, SIMD / gcc -std=c11 -O3 -ffast-math -march=native |  4.408 +- 0.009  |         **42 x**          |
| Cell Linked List / AoS, OpenMP, SIMD / gcc -std=c11 -O3 -ffast-math -march=native |  0.274 +- 0.059  |         **687 x**         |
| Naive Calculation / SoA, OpenMP, SIMD / gcc -std=c11 -O3 -ffast-math -march=native | 0.7507 +- 0.0018 |         **251 x**         |
| Cell Linked List / SoA, OpenMP, SIMD / gcc -std=c11 -O3 -ffast-math -march=native | 0.1598 +- 0.0122 |        **1179 x**         |

| Part 4 (Loops, Load Bal. and Symmetrization) : Algorithm / Implementation / Configuration |  Time (Seconds)  | Speedup (Rel. to slowest) |
| :----------------------------------------------------------: | :--------------: | :-----------------------: |
| CLL / SoA, Outer Loop OpenMP, SIMD / gcc -std=c11 -O3 -ffast-math -march=native | 0.0412 +- 0.0003 |        **4574 x**         |
| CLL / SoA, Load Balanced OpenMP, SIMD / gcc -std=c11 -O3 -ffast-math -march=native | 0.0307 +- 0.0011 |        **6138 x**         |
| CLL / SoA, Symmetrical LB OpenMP, SIMD / gcc -std=c11 -O3 -ffast-math -march=native | 0.0236 +- 0.0006 |        **7985 x**         |
| CLL / SoA, SymmLB OpenMP, CountSort SIMD / gcc -std=c11 -O3 -ffast-math -march=native | 0.0193 +- 0.0021 |        **9764 x**         |

​	The second part explores the relevance of compiler flags to code performance, the interplay of data representation in memory in the choice between Array of Structs and Struct of Arrays, and finally a first implementation of multi-threaded programming using openMP library. The third part is concerned with the use of Instruction Level Parallelism in modern CPUs, in the form of code Vectorization with SIMD instructions, and with how to utilize them with a proper choice of compiler flags as well as judicious selection of what goes in the most critical section of the code. At last, the fourth part deals with improvements of thread-level parallelism in the form of loop re-writting, exposing hidden parallelism and load balancing. Each optimization will be discussed in the context of its implementation in real code, alongside with its impact to the final performance.

​	The present era has seen an explosion of demand for computing power, usually associated with the increasing amount of data collected and stored for Big Data, and the need to interpret this same data and use it for some sort of decision making (analytics and inference). Less known to the wider world, even in technical circles, is a commensurate increase in demand for computer power to perform ever larger computer simulations, both physically based such as [Computational Fluid Dynamics](https://en.wikipedia.org/wiki/Computational_fluid_dynamics) (CFD) and non physical such as [actor model-based simulations](https://en.wikipedia.org/wiki/Actor_model), such as for modeling traffic flow.

​	A recent example of the interplay of engineering, machine learning, simulation and the ever increasing demand for additional power to feed them all is the recent [Tesla AI Day](https://www.youtube.com/watch?v=j0z4FweCy4M), in which Tesla showed how they are using actor model-based simulation to create synthetic data, in the form of 3D rendered traffic through which the autopilot system can wander through a potentially unlimited number of times, using an unlimited number of instances of the autopilot while also having perfect labels because they can be dynamically and exactly generated by the simulation code itself. 

​	Regardless of whether it is in the form of a simulation, pre-processing or learning algorithm, a huge range of algorithms that power modern day progress and innovation are [numerical algorithms](https://people.sc.fsu.edu/~jburkardt/fun/misc/algorithms_dongarra.html), which have achieved such impact in society **because** they are efficient in terms of resources and have very fast implementations, which enable results that matter to be obtained in feasible time spans, for problems of meaningful size: Krizhevsky, Sutskever and Hinton were able to win the ILSCRC 2012 in great part because they had an [efficient implementation of an efficient algorithm](https://dl.acm.org/doi/10.1145/3065386) that, though very expensive, had a [learning capacity](https://en.wikipedia.org/wiki/Vapnik%E2%80%93Chervonenkis_dimension) that enabled it to greatly outperform the competition, even though it was not the first [fast GPU-implementation](https://en.wikipedia.org/wiki/AlexNet#Historic_context) of a convolutional neural network. Though this series is not directly based upon previous literature, many good references exist for particle code implementation in modern systems, including [Cielo et. al. 2020](https://www.sciencedirect.com/science/article/pii/S0167739X19308350) on cosmological n-body codes.

​	Notwithstanding the abundance of tools nowadays, from time to time, the need arises to [modify existing libraries](https://arxiv.org/abs/1802.05799) or to implement new algorithms from scratch, and doing so efficiently requires knowledge not only from the problem, but also how to approach the problem efficiently. Additionally, knowledge on how to translate mathematical ideas into actual fast code requires understanding how the [computer hardware sees, moves and processes data](https://ieeexplore.ieee.org/document/9189797), and how to write the code that can be smartly translated in executable that will run fast, as fast as possible. 

​		This series of articles is based in what I learned re-writing my SPH simulation code from my master's thesis, but in a much cleaner, faster and overall better way using the knowledge I have accumulated since. The material upon which these articles are based can be found in [https://github.com/jhelsas/sphalerite/tree/medium-publication](https://github.com/jhelsas/sphalerite/tree/medium-publication).

## The Example: SPH Density

​	[Smoothed Particle Hydrodynamics](https://en.wikipedia.org/wiki/Smoothed-particle_hydrodynamics) is a computational method for simulating the dynamics of continuous media, such as fluid flows. It was developed by [Gingold and Monaghan](https://doi.org/10.1093%2Fmnras%2F181.3.375) and [Lucy](https://ui.adsabs.harvard.edu/abs/1977AJ.....82.1013L/abstract) in the context of astrophysical problems, which seek to solve Partial Differential Equations that arise in modeling liquid and gases, such as Euler and Navier-Stokes equations. 

​	For the original, [compressible](https://en.wikipedia.org/wiki/Compressible_flow), version of the SPH method, the fluid is discretized using a set of nodes without any connectivity function turning them into a mesh (therefore referred as a mesh-free method), in which such nodes can move during the evolution of the simulation and, consequently, behaving somewhat like "particles" of fluid. A given particle $i$ has a mass $\nu_i$ and is at position ${\mathbf r}_i$, but having other particles around it contribute to the density around it. The density $\rho_i = \rho({\mathbf r}_i )$ is computed as a weighted sum from the masses of and around the particle of interest, the weight given by the smoothing kernel $w(r,h)$:
$$
\rho_i = \rho({\bf r}_i) = \sum_{j=0}^N \nu_j W(|{\mathbf r }_i  - {\mathbf r }_j |,h) = \sum_{j=0}^N \nu_j W_{ij}
$$
​	The weight depends on the distance between particles and it is regulated by the smoothing length h. For those interested in knowing how this calculation arises, it originates from the discretization of the convolution of a smoothed field, which is an [finite-length approximation](https://en.wikipedia.org/wiki/Dirac_delta_function#Representations_of_the_delta_function) of the convolution with the [Dirac Delta](https://en.wikipedia.org/wiki/Dirac_delta_function#As_a_measure), allowing for a finite unstructured set of nodes to approximate a complex function. 

​	The kernel function is a positive function, usually of compact support, such as the cubic [b-spline](https://en.wikipedia.org/wiki/B-spline) that is used in this work:
$$
W(|{\mathbf r }_i  - {\mathbf r }_j |,h) = A_d \left \{ \begin{matrix} \frac{2}{3} - q^2 + \frac{1}{2}q^3 , & \mbox{if }\ q \leq 1 \\ \frac{1}{6}(2-q)^3, & \mbox{if }\ 1 \leq q \leq 2 \\ 0, & q \geq 2\end{matrix} \right. \ \ \\ \ \mbox{where}\ \ \ q = \frac{|{\mathbf r }_i  - {\mathbf r }_j |}{h} \ \ \mbox{and} \ \ A_d = \left \{ \begin{matrix} 1/h , & \mbox{if }\ d = 1 \\ 15/(7\pi h^2), & \mbox{if }\ d = 2 \\ 3/(2\pi h^3), & \mbox{if }\ d = 3\end{matrix} \right.
$$


​	One of the core parts of a Smoothed Particle Hydrodynamics is to compute the density function for all particles for each time-step, this is because the positions  of the particles changes as the simulations progresses, and therefore the density must be updated. The problem I will focus on this article series is how to efficiently compute the density of a reasonable number of particles, ranging from a few 100s of thousands to a few tens of millions. 

## The Problem: Density calculation

​	The basic calculation can be translated pretty much as a double loop. Before anything else, it is important to start defining the data structures that will be operated upon. The simplest solution to begin with is to conceptualize each particle as an entity in itself, and the particle system work as an array of particles. The implementation I started with was using the C programming language, without any of the extra help coming from high-level abstractions such as STL Vectors in C++. I did so because I am much more familiarized with the C language than with the C++ language, and this was my first project in a C-like language in a very long time, and since I did my original SPH code, of which this is a re-writing of, in C, I felt much more comfortable doing it this way. 

​	The basic data structure is a single particle, that has positions `r`, velocities `u`, forces `F`, mass `nu`, density `rho` and additional integer fields named `id` and `hash`. Each of the vector fields can have up to four values stored, one for each dimension. All the data for a given particle is packed together, and the overall feel is that what should be together is being moved together. 

```c
typedef struct double4 {
    double x;
    double y;
    double z;
    double t;
} double4; // defines a "vector" of 4 components: x, y, z and t

typedef struct SPHparticle
{
    double4 r,u,F;   // r, u and F store position, velocity and force data
	int64_t id,hash; // unique-ID,, hash is the index of the cell  
	double nu,rho;  // mass and density at the particle position
} SPHparticle; 
```

​	The basic kernel can be implemented as:

```c
double w_bspline_3d(double r,double h){
  const double A_d = 3./(2.*M_PI*h*h*h); // The 3d normalization constant 
  double q=0.;                  // normalized distance, initialized to zero
  
  if(r<0||h<=0.)                // If either distance or smoothing length
    exit(10);                   // are negative, declare an emergency
  
  q = r/h;                      // Compute the normalized distance
  if(q<=1)                      // If the distance is small
    return A_d*(2./3.-q*q + q*q*q/2.0);// Compute this first polynomial
  else if((1.<=q)&&(q<2.))             // If the distance is a bit larger
    return A_d*(1./6.)*(2.-q)*(2.-q)*(2.-q);// Compute this other polynomial 
  else                                 // Otherwise, if the distance is large
    return 0.;                         // The value of the kernel is 0
}
```

​	Now, the most basic implementation becomes a direct translation of the first equation presented: 

```c
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

​	The simplest, though unrealistic, compilation is using `gcc` with flags `-std=c11 -Wall`, without any optimization flags. The example I used to benchmark all modifications and optimizations was computing the density of particles that all have equal mass equal to $1/N$ , initial positions sampled from a uniform distribution in the $[0,1]\times [0,1] \times [0,1]$ box, the smoothing length $h=0.05$, with all other physical values set to zero. The `hash` field is set to zero and the `id` field is set to the initial index. For the present moment, `hash`was not utilized, and `id` will not be important throughout this series. 

​	Running this code for $10^5$ particles takes around 180 seconds on a modern AMD Ryzen 3900x CPU. Though unrealistically slow, this could be thought as the basest of baselines against which to compare the progress of each optimization step. At 180s for simply computing the density, this code is mostly unusable but for the smallest of simulations,  clearly requiring speeding up in order to be useful in realistic cases. The speedup will come in many flavors: Smarter algorithms, exploiting the parallelism available in the computation, correct use of the compiler and adaptation of the code to make it more digestible to a modern CPU architecture. The first improvement we explore is of the first kind, in the form of being a smarter way to compute the density. 

​	You might be asking why should we start with the simplest, least optimized, "dumbest" version instead of already using a good algorithm right from the start. There are several reasons for this which are perfectly summarized by this quote from Ken Thompson, one of the creators of the C programming language:

![](https://www.azquotes.com/picture-quotes/quote-when-in-doubt-use-brute-force-ken-thompson-131-46-05.jpg)

​	Writing the simplest version of the code allow us to get the math right before actually start doing fancy code changes, thus providing **a way to check correctness** of our optimized versions. It also provides us a baseline timing against which to compare the performance of all future versions: If some optimization performs worse than the dumb version, it is not an optimization, it is a "pessimization". 

​	For the **python programmers**, a code that is similar in spirit to the one above (and therefore a code that *shouldn't* be written), would be something like:

```python
import math
import copy

rho = []
for i in range(N):
    rho_val = 0.
    for j in range(N):
        dist = 0.
        dist += (x[i]-x[j])*(x[i]-x[j])
        dist += (y[i]-y[j])*(y[i]-y[j])
        dist += (z[i]-z[j])*(z[i]-z[j])
        dist = math.sqrt(dist)
        rho_val += nu[j] * w_bspline_3d(dist,h)
    rho.append(copy.copy(rho_val))
```

​	This is a highly non-optimized code and would run terribly slowly if tried. That said, it showcases how would a very naive implementation of the same idea be attempted in python.

## Understanding different cultures: Algorithm Design and Software Architecture

​	One great thing emphasized in the [Algorithms and Data Structures](https://www.geeksforgeeks.org/data-structures/) literature is the focus on how to achieve a goal in more than one way and the attention to algorithm complexity, the main message being: **Don't do work that doesn't need to be done**. As an example, **don't** use bubble sort when quick/heap/merge sort is available, mostly because bubble sort requires $\mathcal{O}(N^2)$ comparisons for an array of size $N$, while the others only require $\mathcal{O}(N\log(N))$ which happen to be the lowest possible asymptotic complexity for the problem of sorting. 

​	Though it is usually self-evident that choosing good algorithms is a **good practice**, when implementing good algorithms involves revamping or re-structuring your entire code-base to incorporate the added logic, it usually is easier than not to left aside them in favor of having a working prototype **going as soon as possible**, an approach that can be easily found in the [agile mindset](https://en.wikipedia.org/wiki/Agile_software_development) of software development. When dealing with numerically intensive code, which is usually meant to be of high performance, this is a mistake because it under-emphasizes important early planning in the project where it would be optimal to find the best algorithm to structure your code around, in favor of a very high number of incremental steps that might lead to easier choices in the beginning but will later make it overly cumbersome to switch to the optimal solution. 	

​	This is not usually a problem in most of software development because most of the software being written **is not performance sensitive**, therefore the developers' efforts as measured in terms of development time, ease of maintenance and reliability are much more important than whether the code runs a bit faster or slower, which justifies many of the premises related to Agile and related mindsets. A common, very cited, related quote is the following one by Knuth, in which it explains his reasoning why **premature** optimization is the root of all evil:

"The conventional wisdom shared by many of today’s software engineers calls for ignoring efficiency in the small; **but I believe this is simply an overreaction to the abuses they see being practiced by pennywise- and-pound-foolish programmers, who can’t debug or maintain their “optimized” programs**. In established engineering disciplines a 12% improvement, easily obtained, is never considered marginal; and I believe the same viewpoint should prevail in software engineering. Of  course I wouldn’t bother making such optimizations on a one-shot job, but when it’s a question of preparing quality programs, I don’t want to  restrict myself to tools that deny me such efficiencies.

There is no doubt that the grail of efficiency leads to abuse. Programmers waste enormous amounts of time thinking about, or worrying  about, the speed of noncritical parts of their programs, and these attempts at efficiency actually have a strong negative impact when debugging and maintenance are considered. **We should forget about small efficiencies, say about 97% of the time: premature optimization is the root of all evil**.

**Yet we should not pass up our opportunities in that critical 3%**. A  good programmer will not be lulled into complacency by such reasoning,  he will be wise to look carefully at the critical code; but only after  that code has been identified. It is often a mistake to make a priori judgments  about what parts of a program are really critical, since the universal  experience of programmers who have been using measurement tools has been that their intuitive guesses fail. (Computing Surveys, Vol. 6, No. 4, December 1974, p. 268 [p. 8 of the [PDF](http://pplab.snu.ac.kr/courses/adv_pl05/papers/p261-knuth.pdf)])" - Emphasis on my part, extracted from [Premature optimization is the root of all evil](https://scottdorman.blog/2009/08/28/premature-optimization-is-the-root-of-all-evil/). 

​	I would like to argue that this is not our case, exactly because we benchmarked so to assert that is calculation one of these "critical 3%", and also because the application itself is performance sensitive. By performance sensitive, what is meant is: It is not uncommon to find modest fluid simulations that take **from hours to days** to run in clusters possessing from hundreds to thousands of CPU cores, and even small simplified simulations can take several hours in modern desktops and workstations. If a very simple simulation would take **a week, or even a year**, to complete every single time it would execute, nobody would do it. 

​	The equivalent case with DNN training goes as follows: The end of the second [AI winter](https://en.wikipedia.org/wiki/AI_winter) and the beginning of the current AI spring came in the back of, though not exclusively, **newer and better algorithms** (e.g. CNNs, backpropagation, stochastic gradient descent), availability of cheap computing power to the masses in the form of retrofitted semi-specialized hardware (**GPGPUs with CUDA**) and existence of **good enough implementations** of these algorithms in these commodity semi-specialized hardware that are also usable by non-specialists (e.g. **CuDNN wrapped by Tensorflow, pyTorch and the like**). 

​	Before gradient learning, it was very difficult to train large and/or deep neural networks because the available optimization methods, like simplex methods, were very poorly suited for optimization in very high dimensionality needed for these neural networks. Gradient learning with symbolic differentiation, e.g. as implemented by Tensorflow, basically bypasses many of the thorniest issues and makes it possible to write, train and perform inference on arbitrarily complex neural network architectures in a manageable time frame without completely overloading the libary user, i.e. the "Deep Learning Practitioner". This is the power of correctly implementing good algorithms, and case in point Tensorflow was [designed to be like that](http://download.tensorflow.org/paper/whitepaper2015.pdf) from scratch even if it since improved incrementally, many of the major design considerations had to be put in from the start, and a significant part of these considerations were performance related. 

​	For the subject of this article, the important thing to notice is: If Tensorflow/Keras/pyTorch existed but it were $1000 \times$ slower, very few people would use them. Easily getting the answer is not sufficient, easily getting the answer in manageable time is necessary. In case you still can't relate to my examples, please read the XKCD cartoon below and exchange "compiling" for "running". 

![](https://www.androidpolice.com/wp-content/uploads/2015/11/nexus2cee_compiling.png)

## Choosing good algorithms: Cell Linked List 

​	On the problem of computing the SPH density the **same principle** discussed before,  "don't do work that doesn't need to be done", can be applied. The basic observation to be exploited is: the kernel $W(r,h) = 0$ if $r/h > 2$ , therefore adding contributions of particles that are farther than $2h$ away from the particle of interest does not contribute meaningfully to the density calculation. The way we can lower the computational cost is to skip the additions that we know for sure are going to be zero: even if we do some needless additions, if we can skip the majority of the zero contributions there is considerable speedup to be gained. 

### The Cell and the List

​	In our case, the work that needs skipping is computing the contribution to density from particles which we know beforehand that is going to be zero. Just using the present data structures there is no way to clearly split which particles are far away from a given particle, therefore not meaningfully contributing to its density, and which are near and need to be accounted for. 

​	The way this is executed is by overlaying an additional support structure that allows us to short-cut the inner loop of the density calculation, which is normally called the "Box", in the method called [Cell Linked List](https://en.wikipedia.org/wiki/Cell_lists). The Box is a grid of smaller boxes (or cells), either explicit or implicit, in which each small box "stores" the particles that falls in its bounds. The idea is that if it is quick to find which particles are in a given box, knowing the width of the smaller box it is possible to only fetch the particles that are in the boxes closer than $2h$ to the one you are interested in. The simplest case is using the smaller boxes width equals to $2h$ , which allows only the adjacent boxes to be selected, which yields 9 boxes in 2D and 27 boxes in 3D. 

  ![CellLists](https://upload.wikimedia.org/wikipedia/en/6/62/CellLists.png)  

​	A simple example can be seen in the images above: We want to compute the contributions to the particle the arrays radiate from, but only a limited number of particles can contribute a non-zero value in the first place. By overlaying a grid with width equals to the diameter of the [support](https://en.wikipedia.org/wiki/Support_(mathematics)) of the kernel, $2h$, it is possible to simply go through the particles that fall within the green and red boxes, skipping the rest. 

​	The way most people would approach implementing this overlay structure is through a *explicit* data structure, such as a [real linked list](https://en.wikipedia.org/wiki/Linked_list), and using pointers to reference the particles that are in each box. For **Python only or Python first users**, think of pointers as a reference to another object, only a bit *more raw* as it is managed entirely by the developer. A potential implementation using pointers would be as follows: 

``` c
typedef struct SPHparticle_ll{
    SPHparticle p;
    SPHparticle_ll *next;
} SPHparticle_ll;

typedef struct linkedListBox{
    int Nx,Ny,Nz;
    double Xmin,Ymin,Zmin;
    double Xmax,Ymax,Zmax;
    SPHparticle **ll_cell;
} linkedListBox;
```

​	Though it might feel good, and I did implement in a very similar fashion in my original code, I would strongly argue against doing it this way nowadays. There are several reasons why this is now a very good way to implement this search:

- The Box structure consumes considerable memory or complexity: Either you will need to store a potentially `NULL` pointer value for each possible cell, regardless of whether there are particles there or not, or you will need this structure and an additional structure like a hash table to book-keep which cells are filled and, therefore, that can be used for referencing the actual particles. 
- Using linked lists requires wrapping your Struct: though not too important, this adds a bit more complexity and consumes more memory, and this can be avoided if the cell Box is implemented implicitly.
- Going through the particle this way introduces pointer chasing: This is the main reason **not to implement it this way**, because it can cause a very significant performance hit in the actual execution, specially when compared to proper cache friendly implementations. Details in the Appendix. 

​	Instead of using explicit pointers an alternative route is to implement implicitly, which was done here through a combination of hashes, sorting and hash tables. 

### The Hash and the Table

​	As mentioned, pointers can cause problems in numerical code, and the main way to avoid them is to use indexes wherever possible. The first step to do so is to associate a number, the cell **hash**, to each particle that is in a given cell. With this number any sorting algorithm will be able to collapse all particles that are in a given cell together **sequentially** in the array. This effectively creates the ability to go through all particles in a given cell as long as you know the starting and ending indexes of a given cell, essentially creating a **virtual** linked list structure **without** explicit pointers. 

​	As far as collapsing particles inside a given cell together, any hash that is unique for each cell will do the job, but some hashes are better than others. Suppose that you have a cell with $x$, $y$ and $z$  integer indexes `kx`, `ky` and `kz` of a Box with `Nx`, `Ny` and `Nz` in each of the respective directions, one possible solution is to compute the hash as:

```c
hash = kx + Nx*( ky + Ny*kz )
```

​	This is a linear index, which is what linear translation from 3D to 1D would happen using a [multi-dimensional array](https://stackoverflow.com/questions/22259306/indexing-multidimensional-arrays-in-c), credible solution for collapsing the data into cells, but it does have a somewhat inconvenient feature: even if you have two boxes with the same `kx` and `ky`, `kz` and `kz+1` will produce values that are `Nx*Ny` apart, so most of the boxes that are spatially close will be mapped into hashes that are quite far apart. In the `y` direction is a bit less dramatic, but no better in practice. This can have consequences for performance and will be discussed later. 

​	 Though it is not possible to create a hash that will **always** map nearby cells in 3D to nearby hashes, it is possible to create a hashes that will perform like that **most of the time**. Examples of hash are the [Hilbert Curve hash](https://en.wikipedia.org/wiki/Hilbert_curve#Applications_and_mapping_algorithms) and the [Morton Z-order Curve hash](https://en.wikipedia.org/wiki/Hilbert_curve#Applications_and_mapping_algorithms) (MZC), which was used in this work. The MZC hash uses bit interleaving of the 2D or 3D indices to create a composite number that encodes the 3D position while still preserving a fair amount of spatial locality. From the **point of view of the developer** all you need to know is: How you choose the $3D$ to $1D$ mapping matters, find a library which implements the hash you like, don't fuss too much about afterwards. Also, MZC is a good default for this kind of spatial mapping problems.

​	![600px-Z-curve.svg](https://upload.wikimedia.org/wikipedia/commons/3/30/Z-curve.svg)

​	In possession of a good hash, the next step is to quickly find which are the boxes surrounding a given box of interest. For any given particle, or cell, it is essentially **free** to compute its spatial indexes `kx`, `ky` and `kz`, and the associated hash. The same is true for computing the spatial indexes and hashes of the adjacent boxes and, subsequently, all that is needed is a fast way to translate hashes into indexes in the ordered array of particles. To perform such task I chose an old-fashioned [hash table](https://en.wikipedia.org/wiki/Hash_table) with integer indexes for the hashes, and with values as pair of integers for the index ranges for each virtual cell. The hash table implementation chosen was the [khash](https://stackoverflow.com/questions/3300525/super-high-performance-c-c-hash-map-table-dictionary) from [klib](https://github.com/attractivechaos/klib), mostly because it is a very fast C hash table, but for C++ users Google's [Dense Hash Map](http://goog-sparsehash.sourceforge.net/doc/dense_hash_map.html) would be a good alternative.

​	Ultimately, some form of more complex container or data-structure is necessary to implement this more complex logic, but the key insight is to use it to store the information on the **indexes to the data** , and **not the data itself**, which is kept separated in a **flat memory layout**. This allows us to use smart and flexible data structures, which can be  used to implement smart and cost-saving logic, while also being overall simpler in logic and more friendly to the CPU underneath. 

### 	The Table and the Code

​	 Now, for the implementation, the actual box structure looks like this:

```c
KHASH_MAP_INIT_INT64(0, int64_t)
KHASH_MAP_INIT_INT64(1, int64_t)

typedef struct linkedListBox{
    int Nx,Ny,Nz,N,width;
    double Xmin,Ymin,Zmin;
    double Xmax,Ymax,Zmax;
    khash_t(0) *hbegin;
    khash_t(1) *hend ;
} linkedListBox;
```

​	I found it easier to run two hash tables, one for the beginning and the other for the end of the range, instead of trying to make the library work with pairs of integers. That said, this simply wouldn't be a problem with Dense Hash Map in the case of C++ users. 

​	The hash computation is a pretty straightforward loop:

```c
for(int64_t i=0;i<N;i+=1){
	uint32_t kx,ky,kz;
	kx = (uint32_t)((lsph[i].r.x - box->Xmin)*box->Nx/(box->Xmax - box->Xmin));
	ky = (uint32_t)((lsph[i].r.y - box->Ymin)*box->Ny/(box->Ymax - box->Ymin));
	kz = (uint32_t)((lsph[i].r.z - box->Zmin)*box->Nz/(box->Zmax - box->Zmin));

	if((kx<0)||(ky<0)||(kz<0))
		return 1;
	else if((kx>=box->Nx)||(ky>=box->Nx)||(kz>=box->Nx))
		return 1;
	else
		lsph[i].hash = ullMC3Dencode(kx,ky,kz);
}
```

​	For python users, in C we use `var.member` for data we own and `var->ref` for data we access elsewhere, which is a small but important syntactic difference. And setting up the hash tables is also pretty straightforward:

```c
int64_t hash0 = lsph[0].hash;
kbegin = kh_put(0, box->hbegin, lsph[0].hash, &ret); kh_value(box->hbegin, kbegin) = (int64_t)0;
for(int i=0;i<N;i+=1){
	if(lsph[i].hash == hash0)
		continue;
	hash0 = lsph[i].hash;
	kbegin = kh_put(0, box->hbegin, lsph[i  ].hash, &ret); 
    kend   = kh_put(1, box->hend  , lsph[i-1].hash, &ret); 
    kh_value(box->hbegin, kbegin) = i;
    kh_value(box->hend  , kend)   = i;
}
kend   = kh_put(1, box->hend  , lsph[N-1].hash, &ret); kh_value(box->hend  , kend)   = N;
```

​	Now, the density computation function becomes more interesting, because it now loops over the neighboring cells instead of all particles, which is what effectively lowers the computational cost. The code is more complex than the original (naïve) one, but it runs much faster because it skips the majority of meaningless computations that, otherwise, would be performed. The actual code has two parts: The inner density calculation for a given pair of cells and an (outer) iteration over all cell pairs. The code for the inner loops is as follows:

```C
int compute_density_3d_chunk(int64_t node_begin, int64_t node_end,
                             int64_t nb_begin, int64_t nb_end,double h,
                             SPHparticle *lsph)
{
  // Iterate over the ii index of the chunk
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

And the code for the outer loops is:

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

## Analyzing the Gains:

​	So, for all this trouble, what performance gain do we get? For the same calculation that took around **180 seconds**  in the naive implementation, this version takes only around **3.5 seconds** to arrive at the same result, representing an over $ 50\times$ speedup! 

​	To understand where this speedup comes from, a back of envelope calculation goes like this: Supposing each box has an equal number of particles, there are:
$$
\left[\frac{X_\mbox{max}-X_\mbox{min}}{2\times h}\right] \times \left[\frac{Y_\mbox{max}-Y_\mbox{min}}{2\times h}\right] \times \left[\frac{Z_\mbox{max}-Z_\mbox{min}}{2\times h}\right]= \left[\frac{1-0}{2\times 0.05}\right]^3 = 1000
$$
​	boxes, each with 27 adjacent boxes, totaling $1000 \times 27 = 27000$ box pairs. The actual number of box pairs used for the calculation is lower because we are counting box pairs that would have boxes outside the domain, such as beyond the upper and lower faces. To calculate the right number we need to subtract each box that is a continuation of the faces, edges and corners. Considering a cube that is 1 box wider in each direction, the outer-most boxes must be subtracted. 

​	Each box in one of the 6 faces is over-counted 9 times, corresponding to the number of boxes in the original cube that is adjacent to it, and there is $10\times 10 = 100$ boxes in each face. There are 12 edges, each with 10 boxes, which are over-counted 3 times over. Finally, there are 8 corners that are over-counted once. This results in a total number of box pairs of  $27000−6\times 100\times 9 − 12\times 10\times 3−8 = 21232$ , whereas it would have been $10^3 \times 10^3 = 10^6$ box pairs for the naïve calculation. As a consequence, the fraction of box pairs that are actually pooled to compute is $21232/ 10^6 = 0.021232$ or $\approx 2.12 \%$ which translate to a potential speedup of $1/2.12\% \approx  47\times$ . There is some reuse of data in the cache, so the final speedup comes up a bit higher than this. 

​	As with anything, the actual speedup depends on the actual distribution of particles in space, the smoothing length, and might well vary during the simulation. Nevertheless, this simplified case showcases how much impact this approach can contribute to enable the use of this method in realistic scenarios. 

​	This result represents over **50 x ** performance improvement which is, arguably, more than enough to justify the extra mile to implement the additional structures and modifying the code. More importantly and sometimes overlooked, this is an improvement gained from re-thinking the way the calculation is performed, and as such it is **not something that can be gained by tweaking** the original code. This fact will become more apparent as we progress through the next parts of this series, since most of the techniques that can be used to improve the original code can also be applied to this cell linked list search version of it, therefore this performance gain comes **on top** of everything else that can be done to improve the performance of this calculation. 

​	The main lesson from this section is: **Choosing good algorithms matter**.

​	In a sense, this is the most *difficult* of the performance improvements that can be added to this kind of computation, since it requires adding additional structures and, if the code already existed, rewriting extensive parts of the code to accommodate these new structures. In this sense is is a very "Algorithmic" form of optimization, which is different from what is usually thought of as implementation optimization, but it serves the same purpose by the day's end: make your code run **faster**. 

​	In the next parts of this series, we will cover several techniques to further increase the performance by exploiting several resources available in modern CPU architectures: Compiler flags, thread-level parallelism, memory layout and data locality, vectorization and load balancing. 

