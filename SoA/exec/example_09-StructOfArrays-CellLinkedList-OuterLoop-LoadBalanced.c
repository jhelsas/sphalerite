/*
 * SPDX-License-Identifier:  BSD-3-Clause
 * 
 * example_09-StructOfArrays-CellLinkedList-OuterLoop-LoadBalanced.c : 
 *      Example of SPH Density Calculation using 
 *      fast neighbor search the main density loop via
 *      Cell Linked List method, Struct of Arrays (SoA) 
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

int compute_density_3d_load_ballanced(int N, double h, SPHparticle *lsph, linkedListBox *box);

int compute_density_3d_chunk_noomp(int64_t node_begin, int64_t node_end,
                                   int64_t nb_begin, int64_t nb_end,double h,
                                   double* restrict x, double* restrict y,
                                   double* restrict z, double* restrict nu,
                                   double* restrict rho);

int count_box_pairs(linkedListBox *box);

int setup_box_pairs(linkedListBox *box,
                    int64_t *node_begin,int64_t *node_end,
                    int64_t *nb_begin,int64_t *nb_end);

double w_bspline_3d_constant(double h);

#pragma omp declare simd
double w_bspline_3d_simd(double q);

int main(int argc, char **argv){
  bool run_seed = false;
  int err, runs = 1;
  long int seed = 123123123;
  int64_t N = 100000;
  double h=0.05;
  linkedListBox *box;
  SPHparticle *lsph;

  box = (linkedListBox*)malloc(1*sizeof(linkedListBox));
  arg_parse(argc,argv,&N,&h,&seed,&runs,&run_seed,box);

  if(dbg)
    printf("hello - 0\n");
  err = SPHparticle_SoA_malloc(N,&lsph);
  if(err)
    printf("error in SPHparticle_SoA_malloc\n");

  void *swap_arr = malloc(N*sizeof(double));
  double times[runs][5];

  for(int run=0;run<runs;run+=1)
    main_loop(run,run_seed,N,h,seed,swap_arr,box,lsph,times);

  bool is_cell = true;
  print_time_stats("SoA,simd,outer,loadBallance",is_cell,N,h,seed,runs,lsph,box,times);
  print_sph_particles_density("SoA,simd,outer,loadBallance",is_cell,N,h,seed,runs,lsph,box);

  if(dbg)
    printf("hello - 10\n");
  SPHparticleSOA_safe_free(N,&lsph);
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

  double t0,t1,t2,t3,t4,t5;

  t0 = omp_get_wtime();

  err = compute_hash_MC3D(N,lsph,box);

  // ------------------------------------------------------ //

  t1 = omp_get_wtime();
  
  qsort(lsph->hash,N,2*sizeof(int64_t),compare_int64_t);

  // ------------------------------------------------------ //
  
  t2 = omp_get_wtime();

  err = reorder_lsph_SoA(N,lsph,swap_arr);
  if(err)
    printf("error in reorder_lsph_SoA\n");

  // ------------------------------------------------------ //

  t3 = omp_get_wtime();

  if(dbg)
    printf("hello - 6\n");
  err = setup_interval_hashtables(N,lsph,box);
  if(err)
    printf("error in setup_interval_hashtables\n");

  // ------------------------------------------------------ //

  t4 = omp_get_wtime();

  if(dbg)
    printf("hello - 7\n");

  err = compute_density_3d_load_ballanced(N,h,lsph,box);
  if(err)
    printf("error in compute_density\n");

  // ------------------------------------------------------ //

  t5 = omp_get_wtime();

  times[5*run+0] = t1-t0;
  times[5*run+1] = t2-t1;
  times[5*run+2] = t3-t2;
  times[5*run+3] = t4-t3;
  times[5*run+4] = t5-t4;

  if(dbg){
    printf("fast neighbour search / SoA / outer-openMP / symmetric load balanced\n");
    printf("compute_hash_MC3D calc time                 : %lg s : %.2lg%%\n",t1-t0,100*(t1-t0)/(t5-t0));
    printf("qsort calc time                             : %lg s : %.2lg%%\n",t2-t1,100*(t2-t1)/(t5-t0));
    printf("reorder_lsph_SoA calc time                  : %lg s : %.2lg%%\n",t3-t2,100*(t3-t2)/(t5-t0));
    printf("setup_interval_hashtables calc time         : %lg s : %.2lg%%\n",t4-t3,100*(t4-t3)/(t5-t0));
    printf("compute_density_3d load balanced calc time  : %lg s : %.2lg%%\n",t5-t4,100*(t5-t4)/(t5-t0));
    printf("compute_density_3d load balanced total time : %lg s : %.2lg%%\n",t5-t0,100*(t5-t0)/(t5-t0));
  }

  return 0;
}

int compute_density_3d_load_ballanced(int N, double h, SPHparticle *lsph, linkedListBox *box){
  int64_t *node_begin,*node_end,*nb_begin,*nb_end;
  int64_t max_box_pair_count = 0;

  max_box_pair_count = count_box_pairs(box);
  
  node_begin = (int64_t*)malloc(max_box_pair_count*sizeof(int64_t));
  node_end   = (int64_t*)malloc(max_box_pair_count*sizeof(int64_t));
  nb_begin   = (int64_t*)malloc(max_box_pair_count*sizeof(int64_t));
  nb_end     = (int64_t*)malloc(max_box_pair_count*sizeof(int64_t));

  setup_box_pairs(box,node_begin,node_end,nb_begin,nb_end);

  for(int64_t ii=0;ii<N;ii+=1)
    lsph->rho[ii] = 0.0; 

  #pragma omp parallel for 
  for(size_t i=0;i<max_box_pair_count;i+=1){
    compute_density_3d_chunk_noomp(node_begin[i],node_end[i],nb_begin[i],nb_end[i],
                                   h,lsph->x,lsph->y,lsph->z,lsph->nu,lsph->rho);
  }
  
  free(node_begin); 
  free(node_end);
  free(nb_begin);
  free(nb_end);

  return 0;
}

int compute_density_3d_chunk_noomp(int64_t node_begin, int64_t node_end,
                                   int64_t nb_begin, int64_t nb_end,double h,
                                   double* restrict x, double* restrict y,
                                   double* restrict z, double* restrict nu,
                                   double* restrict rho){
  const double inv_h = 1./h;
  const double kernel_constant = w_bspline_3d_constant(h);

  for(int64_t ii=node_begin;ii<node_end;ii+=1){
    double xii = x[ii];
    double yii = y[ii];
    double zii = z[ii];
    double rhoii = 0.0;
   
    #pragma omp simd reduction(+:rhoii) 
    for(int64_t jj=nb_begin;jj<nb_end;jj+=1){
      double q = 0.;

      double xij = xii-x[jj];
      double yij = yii-y[jj];
      double zij = zii-z[jj];

      q += xij*xij;
      q += yij*yij;
      q += zij*zij;

      q = sqrt(q)*inv_h;

      rhoii += nu[jj]*w_bspline_3d_simd(q);
    }
    rho[ii] += rhoii*kernel_constant;
  }

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
    int64_t node_hash=-1;
    int64_t nblist[(2*box->width+1)*(2*box->width+1)*(2*box->width+1)];

    if (kh_exist(box->hbegin, kbegin)){ // I have to call this!
      khint32_t kend = kh_get(1, box->hend, kh_key(box->hbegin, kbegin));

      node_hash  = kh_key(box->hbegin, kbegin);
      
      neighbour_hash_3d(node_hash,nblist,box->width,box);
      for(unsigned int j=0;j<(2*box->width+1)*(2*box->width+1)*(2*box->width+1);j+=1){
        if(nblist[j]>=0){
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
