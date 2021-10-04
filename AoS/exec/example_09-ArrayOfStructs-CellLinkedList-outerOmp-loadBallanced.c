/*
 * SPDX-License-Identifier:  BSD-3-Clause
 * 
 * example_09-ArrayOfStructs-CellLinkedList-OuterOmp-loadBallanced.c : 
 *      Example of SPH Density Calculation using 
 *      fast neighbor search the main density loop via
 *      Cell Linked List method, Array of Structs (AoS) 
 *      data layout, OpenMP parallelization at the 
 *      cell-pair level, SIMD directives in the kernel
 *      and in the inner-most loop. It also implements 
 *      load balancing by moving the parallelism from 
 *      iterating over cells to iterate over cell pairs. 
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

int compute_density_3d_chunk_noomp(int64_t node_begin, int64_t node_end,
                                   int64_t nb_begin, int64_t nb_end,double h,
                                   SPHparticle *lsph);

int compute_density_3d_fns_load_ballanced(int N, double h, SPHparticle *lsph, linkedListBox *box);

double w_bspline_3d_constant(double h);

#pragma omp declare simd
double w_bspline_3d_simd(double q);

int count_box_pairs(linkedListBox *box);

int setup_box_pairs(linkedListBox *box,
                    int64_t *node_begin,int64_t *node_end,
                    int64_t *nb_begin,int64_t *nb_end);

int main(int argc, char **argv){
  bool run_seed = false;
  int runs = 1;
  long int seed = 123123123;
  int64_t N = 100000;
  double h=0.05;
  linkedListBox *box;
  SPHparticle *lsph;

  box = (linkedListBox*)malloc(1*sizeof(linkedListBox));
  arg_parse(argc,argv,&N,&h,&seed,&runs,&run_seed,box);

  if(dbg)
    printf("hello - 0\n");
  lsph = (SPHparticle*)malloc(N*sizeof(SPHparticle));
  
  void *swap_arr = malloc(N*sizeof(double));
  double times[runs][5];

  for(int run=0;run<runs;run+=1)
    main_loop(run,run_seed,N,h,seed,swap_arr,box,lsph,times);

  bool is_cll = true;
  const char *prefix = "outerOmp,SIMD,loadBallance";
  print_time_stats(prefix,is_cll,N,h,seed,runs,lsph,box,times);
  print_sph_particles_density(prefix,is_cll,N,h,seed,runs,lsph,box);

  if(dbg)
    printf("hello - 10\n");
  free(lsph);
  safe_free_box(box);
  free(swap_arr);

  return 0;
}

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

  err = compute_density_3d_fns_load_ballanced(N,h,lsph,box);
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

int compute_density_3d_chunk_noomp(int64_t node_begin, int64_t node_end,
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
   
    #pragma omp simd reduction(+:rhoii)
    for(int64_t jj=nb_begin;jj<nb_end;jj+=1){
      double q = 0.;

      double xij = xii-lsph[jj].r.x;
      double yij = yii-lsph[jj].r.y;
      double zij = zii-lsph[jj].r.z;  

      q += xij*xij;
      q += yij*yij;
      q += zij*zij;

      q = sqrt(q)*inv_h;

      rhoii += lsph[jj].nu*w_bspline_3d_simd(q);
    }
    lsph[ii].rho += kernel_constant*rhoii;
  }

  return 0;
}

int compute_density_3d_fns_load_ballanced(int N, double h, SPHparticle *lsph, linkedListBox *box){
  int64_t *node_begin,*node_end,*nb_begin,*nb_end;
  int64_t max_box_pair_count = 0;

  max_box_pair_count = count_box_pairs(box);
  
  node_begin = (int64_t*)malloc(max_box_pair_count*sizeof(int64_t));
  node_end   = (int64_t*)malloc(max_box_pair_count*sizeof(int64_t));
  nb_begin   = (int64_t*)malloc(max_box_pair_count*sizeof(int64_t));
  nb_end     = (int64_t*)malloc(max_box_pair_count*sizeof(int64_t));

  setup_box_pairs(box,node_begin,node_end,nb_begin,nb_end);

  for(int64_t ii=0;ii<N;ii+=1)
    lsph[ii].rho = 0.0; 

  #pragma omp parallel for 
  for(size_t i=0;i<max_box_pair_count;i+=1)
    compute_density_3d_chunk_noomp(node_begin[i],node_end[i],nb_begin[i],nb_end[i],h,lsph);
  
  free(node_begin); 
  free(node_end);
  free(nb_begin);
  free(nb_end);

  return 0;
}

double w_bspline_3d_constant(double h){
  return 3./(2.*M_PI*h*h*h);
}

#pragma omp declare simd
double w_bspline_3d_simd(double q){
  double wq = 0.0;
  double wq1 = (0.6666666666666666 - q*q + 0.5*q*q*q);
  double wq2 = 0.16666666666666666*(2.-q)*(2.-q)*(2.-q); 
  
  if(q<2.)
    wq = wq2;

  if(q<1.)
    wq = wq1;
  
  return wq;
}

int count_box_pairs(linkedListBox *box){
  int64_t box_pair_count = 0;

  for (khint32_t kbegin = kh_begin(box->hbegin); kbegin != kh_end(box->hbegin); kbegin++){
    int64_t node_hash=-1,node_begin=0, node_end=0;
    int64_t nb_begin= 0, nb_end = 0;
    int64_t nblist[(2*box->width+1)*(2*box->width+1)*(2*box->width+1)];

    if (kh_exist(box->hbegin, kbegin)){ // I have to call this!
      khint32_t kend = kh_get(1, box->hend, kh_key(box->hbegin, kbegin));

      node_hash  = kh_key(box->hbegin, kbegin);
      node_begin = kh_value(box->hbegin, kbegin);
      node_end   = kh_value(box->hend, kend);

      neighbour_hash_3d(node_hash,nblist,box->width,box);
      for(unsigned int j=0;j<(2*box->width+1)*(2*box->width+1)*(2*box->width+1);j+=1){
        if(nblist[j]>=0){
          nb_begin = kh_value(box->hbegin, kh_get(0, box->hbegin, nblist[j]) );
          nb_end   = kh_value(box->hend  , kh_get(1, box->hend  , nblist[j]) );

          box_pair_count += 1;
        }
      }
    }
  }
  
  return box_pair_count;
}

int setup_box_pairs(linkedListBox *box,
                    int64_t *node_begin,int64_t *node_end,
                    int64_t *nb_begin,int64_t *nb_end)
{
  int64_t box_pair_count = 0;

  for (khint32_t kbegin = kh_begin(box->hbegin); kbegin != kh_end(box->hbegin); kbegin++){
    int64_t node_hash=-1;
    int64_t nblist[(2*box->width+1)*(2*box->width+1)*(2*box->width+1)];

    if (kh_exist(box->hbegin, kbegin)){ 
      khint32_t kend = kh_get(1, box->hend, kh_key(box->hbegin, kbegin));

      node_hash = kh_key(box->hbegin, kbegin);

      neighbour_hash_3d(node_hash,nblist,box->width,box);
      for(unsigned int j=0;j<(2*box->width+1)*(2*box->width+1)*(2*box->width+1);j+=1){
        if(nblist[j]>=0){
          node_begin[box_pair_count] = kh_value(box->hbegin, kbegin);
          node_end[box_pair_count]   = kh_value(box->hend, kend);
          nb_begin[box_pair_count]   = kh_value(box->hbegin, kh_get(0, box->hbegin, nblist[j]) );
          nb_end[box_pair_count]     = kh_value(box->hend  , kh_get(1, box->hend  , nblist[j]) );

          box_pair_count += 1;
        }
      }
    }
  }

  return box_pair_count;
}