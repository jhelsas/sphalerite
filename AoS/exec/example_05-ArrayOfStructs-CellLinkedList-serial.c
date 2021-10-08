/*
 * SPDX-License-Identifier:  BSD-3-Clause
 * 
 * example_05-ArrayOfStructs-CellLinkedList-serial.c : 
 *      Example of SPH Density Calculation using 
 *      fast neighbor search the main density loop via
 *      Cell Linked List method, Array of Structs (AoS) 
 *      data layout, no parallelization or SIMD.
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
              linkedListBox *box, SPHparticle *lsph, double *times);

int compute_density_3d_chunk(int64_t node_begin, int64_t node_end,
                             int64_t nb_begin, int64_t nb_end,double h,
                             SPHparticle *lsph);

int compute_density_3d_cll(int N, double h, SPHparticle *lsph, linkedListBox *box);

double w_bspline_3d(double r,double h);

int main(int argc, char **argv){
  bool run_seed = false;       // By default the behavior is is to use the same seed
  int runs = 1;                // it only runs once
  long int seed = 123123123;   // The default seed is 123123123
  int64_t N = 100000;          // The default number of particles is N = 100000 = 10^5
  double h=0.05;               // The default kernel smoothing length is h = 0.05
  linkedListBox *box;          // Uninitialized Box containing the cells for the cell linked list method
  SPHparticle *lsph;           // Uninitialized array of SPH particles

  box = (linkedListBox*)malloc(1*sizeof(linkedListBox)); // Create a box

  // allow for command line customization of the run
  arg_parse(argc,argv,&N,&h,&seed,&runs,&run_seed,box);  // Parse the command 
                                                         // line arguments and override default values

  if(dbg)
    printf("hello - 0\n");
  lsph = (SPHparticle*)malloc(N*sizeof(SPHparticle));    // Create an array of N particles
  double times[runs][5];

  for(int run=0;run<runs;run+=1)
    main_loop(run,run_seed,N,h,seed,box,lsph,times);

  bool is_cll = true;
  const char *prefix = "simple";
  print_time_stats(prefix,is_cll,N,h,seed,runs,lsph,box,times);
  print_sph_particles_density(prefix,is_cll,N,h,seed,runs,lsph,box);

  if(dbg)
    printf("hello - 10\n");
  free(lsph);
  safe_free_box(box);

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
              linkedListBox *box, SPHparticle *lsph, double *times)
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

  err = compute_hash_MC3D(N,lsph,box);                    // Compute Morton Z 3D hash based on the 
  if(err)                                                 // cell index for each of the X, Y and Z 
    printf("error in compute_hash_MC3D\n");               // directions, in which a given particle reside

  t1 = omp_get_wtime();
  
  qsort(lsph,N,sizeof(SPHparticle),compare_SPHparticle);  // Sort Particle Array according to hash, therefore
                                                          // implicitly creating a cell of particles of same hash
  t2 = omp_get_wtime();

  err = setup_interval_hashtables(N,lsph,box);            // Annotate the begining and end of each cell
  if(err)                                                 // As to have a quick way to retrieve a cell 
    printf("error in setup_interval_hashtables\n");       // given its hash . 

  t3 = omp_get_wtime();

  err = compute_density_3d_cll(N,h,lsph,box);             // Compute the density of the particles based
  if(err)                                                 // on the cell linked list method for fast
    printf("error in compute_density_3d_innerOmp\n");     // neighbor search. 

  t4 = omp_get_wtime();

  // ------------------------------------------------------ //

  times[5*run+0] = t1-t0;                                 // Time for compute morton Z 3d hash
  times[5*run+1] = t2-t1;                                 // Time for sorting the particles
  times[5*run+2] = t3-t2;                                 // Time for setting up the interval hash tables
  times[5*run+3] = t4-t3;                                 // Time for computing the SPH particle densities
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

/*
 *  Function compute_density_3d_cll:
 *    Computes the SPH density from the particles using cell linked list 
 * 
 *    Arguments:
 *       N <int>              : Number of SPH particles to be used in the run
 *       h <double>           : Smoothing Length for the Smoothing Kernel w_bspline
 *       lsph <SPHparticle>   : Array (pointer) of SPH particles to be updated
 *    Returns:
 *       0                    : error code returned
 *       lsph <SPHparticle>   : SPH particle array is updated in the rho field by reference
 */
int compute_density_3d_cll(int N, double h, SPHparticle *lsph, linkedListBox *box){
  khiter_t kbegin,kend;
  int64_t node_hash=-1,node_begin=0, node_end=0;                                 // Start initializing the node indexes on the array 
  int64_t nb_begin= 0, nb_end = 0;                                               // initialize the neighbor indexes 
  int64_t nblist[(2*box->width+1)*(2*box->width+1)*(2*box->width+1)];            // prepare a list of potential neighbor hashes

  for (kbegin = kh_begin(box->hbegin); kbegin != kh_end(box->hbegin); kbegin++){ // Iterate over each receiver cell begin index 
    if (kh_exist(box->hbegin, kbegin)){                                          // verify if that given iterator actually exists
      kend = kh_get(1, box->hend, kh_key(box->hbegin, kbegin));                  // Then get the end of the receiver cell iterator
      node_hash = kh_key(box->hbegin, kbegin);                                   // Then get the hash corresponding to it
      node_begin = kh_value(box->hbegin, kbegin);                                // Get the receiver cell begin index in the array
      node_end   = kh_value(box->hend, kend);                                    // Get the receiver cell end index in the array

      for(int64_t ii=node_begin;ii<node_end;ii+=1)                               // iterate over the receiver cell particles
        lsph[ii].rho = 0.0;                                                      // and initialize its densities to zero

      neighbour_hash_3d(node_hash,nblist,box->width,box);                        // then find the hashes of its neighbors 
      for(int j=0;j<(2*box->width+1)*(2*box->width+1)*(2*box->width+1);j+=1){    // and the iterate over them
        if(nblist[j]>=0){                                                        // if a given neighbor actually has particles
          
          nb_begin = kh_value(box->hbegin, kh_get(0, box->hbegin, nblist[j]) );  // then get the contributing cell begin index
          nb_end   = kh_value(box->hend  , kh_get(1, box->hend  , nblist[j]) );  // and get the contributing cell end index 

          compute_density_3d_chunk(node_begin,node_end,nb_begin,nb_end,h,lsph);  // and compute the density contribution from 
        }                                                                        // the contributing cell to the receiver cell
      }
    }
  }

  return 0;
}

/*
 *  Function compute_density_3d_chunk:
 *    Computes the SPH density contribution for a pair of cells, from nb_ indexes
 *    to the node_ indexes. The computation is performed serially and without 
 *    vectorization.
 * 
 *    Arguments:
 *       node_begin <int>     : Begin index of the receiver cell
 *       node_end   <int>     : End   index of the receiver cell
 *       nb_begin <int>       : Begin index of the sender (neighbor) cell
 *       nb_end   <int>       : End   index of the sender (neighbor) cell
 *       h       <double>     : Smoothing Length for the Smoothing Kernel w_bspline
 *       lsph <SPHparticle*>  : Array (pointer) of SPH particles to be updated
 *    Returns:
 *       0                    : error code returned
 *       lsph <SPHparticle*>  : SPH particle array is updated in the rho field by reference
 */
int compute_density_3d_chunk(int64_t node_begin, int64_t node_end,
                             int64_t nb_begin, int64_t nb_end,double h,
                             SPHparticle *lsph)
{
  for(int64_t ii=node_begin;ii<node_end;ii+=1){           // Iterate over the ii index of the chunk
    double xii = lsph[ii].r.x;                            // Load the X component of the ii particle position
    double yii = lsph[ii].r.y;                            // Load the Y component of the ii particle position
    double zii = lsph[ii].r.z;                            // Load the Z component of the ii particle position
    double rhoii = 0.0;                                   // Initialize the chunk contribution to density 
   
    for(int64_t jj=nb_begin;jj<nb_end;jj+=1){             // Iterate over the each other particle in jj loop
      double q = 0.;                                      // Initialize the distance

      double xij = xii-lsph[jj].r.x;                      // Load and subtract jj particle's X position component
      double yij = yii-lsph[jj].r.y;                      // Load and subtract jj particle's Y position component
      double zij = zii-lsph[jj].r.z;                      // Load and subtract jj particle's Z position component

      q += xij*xij;                                       // Add the jj contribution to the ii distance in X
      q += yij*yij;                                       // Add the jj contribution to the ii distance in Y
      q += zij*zij;                                       // Add the jj contribution to the ii distance in Z

      q = sqrt(q);                                        // Sqrt to compute the distance

      rhoii += lsph[jj].nu*w_bspline_3d(q,h);             // Add up the contribution from the jj particle
    }                                                     // to the intermediary density and then
    lsph[ii].rho += rhoii;                                // add the intermediary density to the full density
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