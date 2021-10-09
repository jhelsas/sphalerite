/*
 * SPDX-License-Identifier:  BSD-3-Clause
 * 
 * example_04-StructOfArrays-Naive-Omp-SIMD-Tiled.c : 
 *      Example of SPH Density Calculation using a 
 *      naive implementation of the main density loop, 
 *      no neighbours earch, and Struct of Arrays (SoA) 
 *      data layout, OpenMP parallelization and SIMD 
 *      directives on the kernel and density calculation.
 *      This incorporates strip mining and exchange to 
 *      implement cache blocking and support performance 
 *      for large number of particles that would otherwise
 *      be lost. 
 *
 * (C) Copyright 2021 José Hugo Elsas
 * Author: José Hugo Elsas <jhelsas@gmail.com>
 *
 * Command Line Options: 
 *   -runs  <int>   : Set the number of repetitions (runs) for
 *                      calculating the density. The value of
 *                      the density is based on the last 
 *                      iteration.
 *                    Default value: 1
 *   -run_seed <int>: Flag to set an alternative seed use for
 *                      for the PRNG. Instead of feeding seed
 *                      to the PRNG directly, it feeds 
 *                      seed + iteration, as to generate different
 *                      configurations for each iteration. 
 *                    Default value: 0 - (possible 0/1)
 *   -seed     <int>: Set the seed to use for the SPH particles 
 *                      uniform position generation in the box
 *                    Default value: 123123123
 *
 *   -N        <int>: Set the number of SPH particles to be used
 *                    Default value: 1e5 = 100,000
 *   -h      <float>: Set the value of the smoothing kernel 
 *                      parameter h, which corresponds to half
 *                      of the support of the kernel. 
 *                    Default value: 0.05
 *
 *   -Nx       <int>: Set the number of Cells in the X direction
 *                    Default value: 10
 *   -Ny       <int>: Set the number of Cells in the Y direction
 *                    Default value: 10
 *   -Nz       <int>: Set the number of Cells in the Z direction
 *                    Default value: 10
 * 
 *   -Xmin   <float>: Set the lower bound in the X direction for 
 *                      the Cell Linked List box 
 *                    Default value: 0.0
 *   -Ymin   <float>: Set the lower bound in the Y direction for 
 *                    the Cell Linked List box 
 *                      Default value: 0.0
 *   -Zmin   <float>: Set the lower bound in the Z direction for 
 *                      the Cell Linked List box 
 *                    Default value: 0.0
 * 
 *   -Xmax   <float>: Set the upper bound in the X direction for 
 *                      the Cell Linked List box 
 *                    Default value: 1.0
 *   -Ymax   <float>: Set the upper bound in the Y direction for 
 *                      the Cell Linked List box 
 *                    Default value: 1.0
 *   -Zmax   <float>: Set the upper bound in the Z direction for 
 *                      the Cell Linked List box 
 *                    Default value: 1.0
 */

#include <math.h>
#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include <unistd.h>
#include <stdbool.h>
#include <sys/time.h>
#include <inttypes.h>

#include <omp.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_heapsort.h>

#include "sph_data_types.h"
#include "sph_linked_list.h"
#include "sph_utils.h"

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

#define COMPUTE_BLOCKS 1

int main_loop(int run, bool run_seed, int64_t N, double h, long int seed, 
              void *swap_arr, linkedListBox *box, SPHparticle *lsph, double *times);

int compute_density_3d_naive_omp_simd_tiled(int N,double h,
                                            double* restrict x, double* restrict y,
                                            double* restrict z, double* restrict nu,
                                            double* restrict rho);

double w_bspline_3d_constant(double h);

#pragma omp declare simd
double w_bspline_3d_simd(double q);

int main(int argc, char **argv){
  bool run_seed = false;       // By default the behavior is to use the same seed
  int runs = 1;                // By default the main loop only runs once
  long int seed = 123123123;   // The default seed is 123123123
  int64_t N = 100000;          // The default number of particles is N = 1e5 = 100,000
  double h = 0.05;             // The default kernel smoothing length is h = 0.05
  linkedListBox *box;          // Uninitialized Box containing the cells for the cell linked list method
  SPHparticle *lsph;           // Uninitialized array of SPH particles

  box = (linkedListBox*)malloc(1*sizeof(linkedListBox)); // Create a box representing the entire 3d domain

  // allow for command line customization of the run
  arg_parse(argc,argv,&N,&h,&seed,&runs,&run_seed,box);  // Parse the command line options
                                                         // line arguments and override default values
  int err = SPHparticle_SoA_malloc(N,&lsph);             // Create an arrays for the N particles
  if(err)
    fprintf(stderr,"error in SPHparticle_SoA_malloc\n");  

  void *swap_arr = malloc(N*sizeof(double));
  double times[runs*COMPUTE_BLOCKS];

  for(int run=0;run<runs;run+=1)
    main_loop(run,run_seed,N,h,seed,swap_arr,box,lsph,times);

  bool is_cll = false;
  const char *prefix = "ex04,naive,SoA,omp,simd,tiled";
  print_time_stats(prefix,is_cll,N,h,seed,runs,lsph,box,times);
  print_sph_particles_density(prefix,is_cll,N,h,seed,runs,lsph,box);

  SPHparticleSOA_safe_free(N,&lsph);
  safe_free_box(box);
  free(swap_arr);
  
  return 0;
}

/*
 *  Function main_loop:
 *    Runs the main loop of the program, including the particle array generation, 
 *    density calculation and the timings annotations.
 * 
 *    Arguments:
 *       run <int>            : index (or value) or the present iteration
 *       run_seed <bool>      : boolean defining whether to use run index for seed or not
 *       N <int>              : Number of SPH particles to be used in the run
 *       h <double>           : Smoothing Length for the Smoothing Kernel w_bspline
 *       seed <long int>      : seed for GSL PRNG to generate particle positions
 *       box  <linkedListBox> : Box of linked list cells, encapsulating the 3d domain
 *       lsph <SPHparticle>   : Array (pointer) of SPH particles to be updated
 *       times <double>       : Array to store the computation timings to be updated
 *    Returns:
 *       0                    : error code returned
 *       lsph <SPHparticle>   : SPH particle array is updated in the rho field by reference
 *       times <double>       : Times is updated by reference
 */
int main_loop(int run, bool run_seed, int64_t N, double h, long int seed, 
              void *swap_arr, linkedListBox *box, SPHparticle *lsph, double *times)
{
  int err;
    
  if(run_seed)
    err = gen_unif_rdn_pos_box(N,seed+run,box,lsph);
  else
    err = gen_unif_rdn_pos_box(N,seed,box,lsph);

  if(err)
    fprintf(stderr,"error in gen_unif_rdn_pos\n");
  
  // ------------------------------------------------------ //

  double t0,t1;

  t0 = omp_get_wtime();
  
  compute_density_3d_naive_omp_simd_tiled(N,h,lsph->x,lsph->y,      // Compute the density for all particles
                                    lsph->z,lsph->nu,lsph->rho);

  t1 = omp_get_wtime();

  // ------------------------------------------------------ //
  
  times[COMPUTE_BLOCKS*run+0] = t1-t0;                      // Only one component to measure time
  
  return 0;
}
/*
 *  Function compute_density_3d_naive_omp_simd_tiled:
 *    Computes the SPH density from the particles implementing a strip mine and exchange
 *    strategy to re-use data in cache over the direct loop. It executes calculations  
 *    in parallel for the outer-most loop using openMP and SIMD in inner-most loop, 
 *    though SIMD only for limited success. 
 * 
 *    Reference: https://en.wikipedia.org/wiki/Loop_nest_optimization
 *    
 *    Arguments:
 *       N  <int>             : Number of SPH particles to be used in the run
 *       h  <double>          : Smoothing Length for the Smoothing Kernel w_bspline
 *       x  <double*>         : X position array of the particles
 *       y  <double*>         : Y position array of the particles
 *       z  <double*>         : Z position array of the particles
 *       nu <double*>         : nu position array of the particles
 *    Returns:
 *       0                    : error code returned
 *       rho <double>         : rho position array 
 */
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

/*
 *  Function w_bspline_3d_constant:
 *    Returns the 3d normalization constant for the cubic b-spline SPH smoothing kernel
 *    
 *    Arguments:
 *       h <double>           : Smoothing Length for the Smoothing Kernel w_bspline
 *    Returns:
 *       3d bspline normalization density <double>
 */
double w_bspline_3d_constant(double h){                            
  return 3./(2.*M_PI*h*h*h);  // 3d normalization value for the b-spline kernel
}

/*
 *  Function w_bspline_3d_simd:
 *    Returns the un-normalized value of the cubic b-spline SPH smoothing kernel
 *    
 *    Arguments:
 *       q <double>           : Distance between particles normalized by the smoothing length h
 *    Returns:
 *       wq <double>          : Unnormalized value of the kernel
 * 
 *    Observation: 
 *       Why not else if(q<2.)? 
 *       Because if you use "else if", the compiler refuses to vectorize, 
 *       This results in a large slowdown, as of 2.5x slower for example_04
 */
#pragma omp declare simd
double w_bspline_3d_simd(double q){                                // Use as input the normalized distance
  double wq=0;
  double wq1 = (0.6666666666666666 - q*q + 0.5*q*q*q);             // The first polynomial of the spline
  double wq2 = 0.16666666666666666*(2.-q)*(2.-q)*(2.-q);           // The second polynomial of the spline
  
  if(q<2.)                                                         // If the distance is below 2
    wq = wq2;                                                      // Use the 2nd polynomial for the spline
  
  if(q<1.)                                                         // If the distance is below 1
    wq = wq1;                                                      // Use the 1st polynomial for the spline

  return wq;                                                       // return which ever value corresponds to the distance
}