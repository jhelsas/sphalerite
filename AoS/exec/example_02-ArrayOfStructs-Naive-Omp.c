/*
 * SPDX-License-Identifier:  BSD-3-Clause
 * 
 * example_02-ArrayOfStructs-Naive-Omp.c : 
 *      Example of SPH Density Calculation using a 
 *      naive implementation of the main density loop, 
 *      no neighbours earch, and Array of Structs (AoS) 
 *      data layout and OpenMP parallelization.
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
 *   -Ymin   <float>: Set the lower bound in the Z direction for 
 *                      the Cell Linked List box 
 *                    Default value: 0.0
 * 
 *   -Xmax   <float>: Set the lower bound in the X direction for 
 *                      the Cell Linked List box 
 *                    Default value: 1.0
 *   -Ymax   <float>: Set the lower bound in the Y direction for 
 *                      the Cell Linked List box 
 *                    Default value: 1.0
 *   -Zmax   <float>: Set the lower bound in the Z direction for 
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

int compute_density_3d_naive_omp(int N,double h,SPHparticle *lsph);

double w_bspline_3d(double r,double h);

int main(int argc, char **argv){
  bool run_seed = false;       // By default the behavior is is to use the same seed
  int runs = 1,err;            // it only runs once
  long int seed = 123123123;   // The default seed is 123123123
  int64_t N = 100000;          // The default number of particles is N = 1e5 = 100,000
  double h=0.05;               // The default kernel smoothing length is h = 0.05
  linkedListBox *box;          // Uninitialized Box containing the cells for the cell linked list method
  SPHparticle *lsph;           // Uninitialized array of SPH particles

  box = (linkedListBox*)malloc(1*sizeof(linkedListBox)); // Create a box

  // allow for command line customization of the run
  arg_parse(argc,argv,&N,&h,&seed,&runs,&run_seed,box);  // Parse the command 
                                                         // line arguments and override default values

  lsph = (SPHparticle*)malloc(N*sizeof(SPHparticle));    // Create an array of N particles
  
  void *swap_arr = malloc(N*sizeof(double));
  double times[runs*COMPUTE_BLOCKS];

  for(int run=0;run<runs;run+=1)
    main_loop(run,run_seed,N,h,seed,swap_arr,box,lsph,times);

  bool is_cll = false;
  const char *prefix = "ex02,naive,AoS,omp";
  print_time_stats(prefix,is_cll,N,h,seed,runs,lsph,box,times);
  print_sph_particles_density(prefix,is_cll,N,h,seed,runs,lsph,box);

  free(lsph);
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
 *       seed <long int>      : seed for GSL PRNG generator to generate particle positions
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

  // Initialize the particles' positions and other values
  if(run_seed)
    err = gen_unif_rdn_pos_box(N,seed+run,box,lsph);
  else
    err = gen_unif_rdn_pos_box(N,seed,box,lsph);

  if(err)
    fprintf(stderr,"error in gen_unif_rdn_pos\n");

  // ------------------------------------------------------ //

  double t0,t1;

  t0 = omp_get_wtime();
  
  compute_density_3d_naive_omp(N,h,lsph);                   // Compute the density for all particles

  t1 = omp_get_wtime();

  // ------------------------------------------------------ //

  times[COMPUTE_BLOCKS*run+0] = t1-t0;                      // Only one component to measure time

  return 0;
}

/*
 *  Function compute_density_3d_naive_omp:
 *    Computes the SPH density from the particles naively (i.e. direct loop).
 *    The outer-most loop is parallelized using openMP. 
 * 
 *    Arguments:
 *       N <int>              : Number of SPH particles to be used in the run
 *       h <double>           : Smoothing Length for the Smoothing Kernel w_bspline
 *       lsph <SPHparticle>   : Array (pointer) of SPH particles to be updated
 *    Returns:
 *       0                    : error code returned
 *       lsph <SPHparticle>   : SPH particle array is updated in the rho field by reference
 */
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

/*
 *  Function w_bspline_3d:
 *    Returns the normalized value of the cubic b-spline SPH smoothing kernel
 *    
 *    Arguments:
 *       q <double>           : Distance between particles
 *       h <double>           : Smoothing Length for the Smoothing Kernel w_bspline
 *    Returns:
 *       wq <double>          : Normalized value of the kernel
 */
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