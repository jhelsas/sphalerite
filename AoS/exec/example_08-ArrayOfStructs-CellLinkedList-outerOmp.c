/*
 * SPDX-License-Identifier:  BSD-3-Clause
 * 
 * example_08-ArrayOfStructs-CellLinkedList-outerOmp.c : 
 *      Example of SPH Density Calculation using 
 *      fast neighbor search the main density loop via
 *      Cell Linked List method, Array of Structs (AoS) 
 *      data layout, OpenMP parallelization at the 
 *      cell level, SIMD directives in the kernel
 *      and in the inner-most loop. 
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
 *                      for the PNRG. Instead of feeding seed
 *                      to the PNRG directly, it feeds 
 *                      seed + iteration, as to generate different
 *                      configurations for each iteration. 
 *                    Default value: 0 - (possible 0/1)
 *   -seed     <int>: Set the seed to use for the SPH particles 
 *                      uniform position generation in the box
 *                    Default value: 123123123
 *
 *   -N        <int>: Set the number of SPH particles to be used
 *                    Default value: 10000
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

#define dbg false

int main_loop(int run, bool run_seed, int64_t N, double h, long int seed, 
              void *swap_arr, linkedListBox *box, SPHparticle *lsph, double *times);

int compute_density_3d_noomp(int64_t node_begin, int64_t node_end,
                             int64_t nb_begin, int64_t nb_end,double h,
                             SPHparticle *lsph);

int compute_density_3d_fns_outerOmp(int N, double h, SPHparticle *lsph, linkedListBox *box);

double w_bspline_3d_constant(double h);

#pragma omp declare simd
double w_bspline_3d_simd(double q);

int main(int argc, char **argv){
  bool run_seed = false;       // By default the behavior is is to use the same seed
  int runs = 1,err;            // it only runs once
  long int seed = 123123123;   // The default seed is 123123123
  int64_t N = 100000;          // The default number of particles is N = 100000 = 10^5
  double h=0.05;               // The default kernel smoothing length is h = 0.05
  linkedListBox *box;          // Uninitialized Box containing the cells for the cell linked list method
  SPHparticle *lsph;           // Uninitialized array of SPH particles

  box = (linkedListBox*)malloc(1*sizeof(linkedListBox)); // Create a box representing the entire 3d domain

  // allow for command line customization of the run
  arg_parse(argc,argv,&N,&h,&seed,&runs,&run_seed,box);  // Parse the command line options
                                                         // line arguments and override default values

  if(dbg)
    printf("hello - 0\n");
  lsph = (SPHparticle*)malloc(N*sizeof(SPHparticle));    // Create an array of N particles
  
  void *swap_arr = malloc(N*sizeof(double));
  double times[runs][5];

  for(int run=0;run<runs;run+=1)
    main_loop(run,run_seed,N,h,seed,swap_arr,box,lsph,times);

  bool is_cll = true;
  const char *prefix = "outerOmp,SIMD";
  print_time_stats(prefix,is_cll,N,h,seed,runs,lsph,box,times);
  print_sph_particles_density(prefix,is_cll,N,h,seed,runs,lsph,box);

  if(dbg)
    printf("hello - 10\n");
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
 *       seed <long int>      : seed for GSL PNRG generator to generate particle positions
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
  if(dbg)
    printf("hello - 1\n");
    
  if(run_seed)
    err = gen_unif_rdn_pos_box(N,seed+run,box,lsph);
  else
    err = gen_unif_rdn_pos_box(N,seed,box,lsph);

  if(err)
    printf("error in gen_unif_rdn_pos\n");

  if(dbg)
    printf("hello - 2\n");

  // ------------------------------------------------------ //

  double t0,t1,t2,t3,t4;

  t0 = omp_get_wtime();

  err = compute_hash_MC3D(N,lsph,box);
  if(err)
    printf("error in compute_hash_MC3D\n");

  t1 = omp_get_wtime();
  
  qsort(lsph,N,sizeof(SPHparticle),compare_SPHparticle);
  
  t2 = omp_get_wtime();

  err = setup_interval_hashtables(N,lsph,box);
  if(err)
    printf("error in setup_interval_hashtables\n");

  t3 = omp_get_wtime();

  err = compute_density_3d_fns_outerOmp(N,h,lsph,box);
  if(err)
    printf("error in compute_density_3d_innerOmp\n");

  t4 = omp_get_wtime();

  // ------------------------------------------------------ //

  times[5*run+0] = t1-t0;
  times[5*run+1] = t2-t1;
  times[5*run+2] = t3-t2;
  times[5*run+3] = t4-t3;
  times[5*run+4] =    0.;

  if(dbg){
    printf("compute_hash_MC3D          : %.5lf s : %.2lf%%\n",t1-t0,100*(t1-t0)/(t4-t0));
    printf("qsort calculation time     : %.5lf s : %.2lf%%\n",t2-t1,100*(t2-t1)/(t4-t0));
    printf("setup_interval_hashtables  : %.5lf s : %.2lf%%\n",t3-t2,100*(t3-t2)/(t4-t0));
    printf("compute_density_3d         : %.5lf s : %.2lf%%\n",t4-t3,100*(t4-t3)/(t4-t0));
    printf("compute_density_3d total   : %.5lf s : %.2lf%%\n",t4-t0,100*(t4-t0)/(t4-t0));
  }

  return 0;
}

int compute_density_3d_noomp(int64_t node_begin, int64_t node_end,
                             int64_t nb_begin, int64_t nb_end,double h,
                             SPHparticle *lsph)
{
  const double inv_h = 1./h;
  const double kernel_constant = w_bspline_3d_constant(h);

  for(int64_t ii=node_begin;ii<node_end;ii+=1){           // Iterate over the ii index of the chunk
    double xii = lsph[ii].r.x;                            // Load the X component of the ii particle position
    double yii = lsph[ii].r.y;                            // Load the Y component of the ii particle position
    double zii = lsph[ii].r.z;                            // Load the Z component of the ii particle position
    double rhoii = 0.0;                                   // Initialize the chunk contribution to density 
   
    #pragma omp simd reduction(+:rhoii)                   // Hint at the compiler to vectorize the inner most loop
    for(int64_t jj=nb_begin;jj<nb_end;jj+=1){             // Iterate over the each other particle in jj loop
      double q = 0.;                                      // Initialize the distance

      double xij = xii-lsph[jj].r.x;                      // Load and subtract jj particle's X position component
      double yij = yii-lsph[jj].r.y;                      // Load and subtract jj particle's Y position component
      double zij = zii-lsph[jj].r.z;                      // Load and subtract jj particle's Z position component

      q += xij*xij;                                       // Add the jj contribution to the ii distance in X
      q += yij*yij;                                       // Add the jj contribution to the ii distance in Y
      q += zij*zij;                                       // Add the jj contribution to the ii distance in Z

      q = sqrt(q)*inv_h;                                  // Sqrt to compute the normalized distance, measured in h

      rhoii += lsph[jj].nu*w_bspline_3d_simd(q);          // Add up the contribution from the jj particle
    }                                                     // to the intermediary density and then
    lsph[ii].rho += kernel_constant*rhoii;                // add the intermediary density to the full density
  }

  return 0;
}

/*
 *  Function compute_density_3d_fns_outerOmp:
 *    Computes the SPH density from the particles using cell linked list with
 *    parallelization in the outer-most loop, iterating over the cells. 
 * 
 *    Arguments:
 *       N <int>              : Number of SPH particles to be used in the run
 *       h <double>           : Smoothing Length for the Smoothing Kernel w_bspline
 *       lsph <SPHparticle>   : Array (pointer) of SPH particles to be updated
 *    Returns:
 *       0                    : error code returned
 *       lsph <SPHparticle>   : SPH particle array is updated in the rho field by reference
 */
int compute_density_3d_fns_outerOmp(int N, double h, SPHparticle *lsph, linkedListBox *box){
  #pragma omp parallel for //private(node_hash,node_begin,node_end,nb_begin,nb_end,nblist)
  for (khiter_t kbegin = kh_begin(box->hbegin); kbegin != kh_end(box->hbegin); kbegin++){
    int64_t node_hash=-1,node_begin=0, node_end=0;
    int64_t nb_begin= 0, nb_end = 0;
    int64_t nblist[(2*box->width+1)*(2*box->width+1)*(2*box->width+1)];
  
    if (kh_exist(box->hbegin, kbegin)){
      khiter_t kend = kh_get(1, box->hend, kh_key(box->hbegin, kbegin));
      node_hash = kh_key(box->hbegin, kbegin);
      node_begin = kh_value(box->hbegin, kbegin);
      node_end   = kh_value(box->hend, kend);

      for(int64_t ii=node_begin;ii<node_end;ii+=1)  // this loop inside was the problem
        lsph[ii].rho = 0.0; 

      neighbour_hash_3d(node_hash,nblist,box->width,box);
      for(unsigned int j=0;j<(2*box->width+1)*(2*box->width+1)*(2*box->width+1);j+=1){
        if(nblist[j]>=0){
          nb_begin = kh_value(box->hbegin, kh_get(0, box->hbegin, nblist[j]) );
          nb_end   = kh_value(box->hend  , kh_get(1, box->hend  , nblist[j]) );

          compute_density_3d_noomp(node_begin,node_end,nb_begin,nb_end,h,lsph);
        }
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
double w_bspline_3d_simd(double q){
  double wq=0;
  double wq1 = (0.6666666666666666 - q*q + 0.5*q*q*q);             // The first polynomial of the spline
  double wq2 = 0.16666666666666666*(2.-q)*(2.-q)*(2.-q);           // The second polynomial of the spline
  
  if(q<2.)                                                         // If the distance is below 2
    wq = wq2;                                                      // Use the 2nd polynomial for the spline
  
  if(q<1.)                                                         // If the distance is below 1
    wq = wq1;                                                      // Use the 1st polynomial for the spline

  return wq;                                                       // return which ever value corresponds to the distance
}