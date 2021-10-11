/*
 * SPDX-License-Identifier:  BSD-3-Clause
 * 
 * example_10-StructOfArrays-CellLinkedList-OuterLoop-SymmetricalLoadBalancing.c : 
 *      Example of SPH Density Calculation using 
 *      fast neighbor search the main density loop via
 *      Cell Linked List method, Struct of Arrays (SoA) 
 *      data layout, OpenMP parallelization at the 
 *      cell-pair level, SIMD directives in the kernel
 *      and in the inner-most loop. It also implements 
 *      symmetrical load balancing by moving the parallelism 
 *      from iterating over cells to iterate over 
 *      unique pairs of cell (i.e. i<j) and recyling the 
 *      intermediary calculations. 
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
 *                    Default value: 0.0
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

#define COMPUTE_BLOCKS 5

#define dbg false

int main_loop(int run, bool run_seed, int64_t N, double h, long int seed, 
              void *swap_arr,int64_t *temp_hash, linkedListBox *box, SPHparticle *lsph, double *times);

int compute_density_3d_symmetrical_load_ballance(int N, double h, SPHparticle *lsph, linkedListBox *box);

int compute_density_3d_chunk_symmetrical(int64_t node_begin, int64_t node_end,
                                         int64_t nb_begin, int64_t nb_end,double h,
                                         double* restrict x, double* restrict y,
                                         double* restrict z, double* restrict nu,
                                         double* restrict rhoi, double* restrict rhoj);

//void quickSort_int64_t2(int64_t *arr, int64_t low, int64_t high);

void quickSort_int64_t_2(int64_t *arr, int64_t low, int64_t high);

void mergeSort(int64_t arr[], int64_t l, int64_t r);

double w_bspline_3d_constant(double h);

#pragma omp declare simd
double w_bspline_3d_simd(double q);

#define ip_swap(a,b) (b)^=((a)^=((b)^=(a)))

KHASH_MAP_INIT_INT64(2, int64_t)

void counting_sort(int64_t N, linkedListBox *box, SPHparticle *lsph,void *swap_arr,int64_t *temp_hash);

int main(int argc, char **argv){
  bool run_seed = false;       // By default the behavior is is to use the same seed
  int runs = 1,err;            // it only runs once
  long int seed = 123123123;   // The default seed is 123123123
  int64_t N = 100000;          // The default number of particles is N = 1e5 = 100,000
  double h=0.05;               // The default kernel smoothing length is h = 0.05
  linkedListBox *box;          // Uninitialized Box containing the cells for the cell linked list method
  SPHparticle *lsph;           // Uninitialized array of SPH particles

  box = (linkedListBox*)malloc(1*sizeof(linkedListBox)); // Create a box representing the entire 3d domain

  // allow for command line customization of the run
  arg_parse(argc,argv,&N,&h,&seed,&runs,&run_seed,box);  // Parse the command line options
                                                         // line arguments and override default values

  err = SPHparticle_SoA_malloc(N,&lsph);
  if(err)
    fprintf(stderr,"error in SPHparticle_SoA_malloc\n");

  void *swap_arr = malloc(N*sizeof(double));
  int64_t *temp_hash = (int64_t*)malloc(2*N*sizeof(int64_t));
  double times[runs*COMPUTE_BLOCKS];

  for(int run=0;run<runs;run+=1)
    main_loop(run,run_seed,N,h,seed,swap_arr,temp_hash,box,lsph,times);

  bool is_cll = true;
  const char *prefix = "ex11,cll,SoA,outer,simd,symmLB,quicker";
  print_time_stats(prefix,is_cll,N,h,seed,runs,lsph,box,times);
  print_sph_particles_density(prefix,is_cll,N,h,seed,runs,lsph,box);

  SPHparticleSOA_safe_free(N,&lsph);
  safe_free_box(box);
  free(swap_arr);
  free(temp_hash);

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
              void *swap_arr, int64_t *temp_hash, linkedListBox *box, SPHparticle *lsph, double *times)
{
  int err;
    
  if(run_seed)
    err = gen_unif_rdn_pos_box(N,seed+run,box,lsph);
  else
    err = gen_unif_rdn_pos_box(N,seed,box,lsph);

  if(err)
    fprintf(stderr,"error in gen_unif_rdn_pos\n");

  // ------------------------------------------------------ //

  double t0,t1,t2,t3,t4,t5;

  t0 = omp_get_wtime();

  err = compute_hash_MC3D(N,lsph,box);                               // Compute Morton Z 3D hash based on the 
  if(err)                                                            // cell index for each of the X, Y and Z 
    fprintf(stderr,"error in compute_hash_MC3D\n");                  // directions, in which a given particle reside

  t1 = omp_get_wtime();
  
  counting_sort(N,box,lsph,swap_arr,temp_hash);

  t2 = omp_get_wtime();

  t3 = omp_get_wtime();

  t4 = omp_get_wtime();

  err = compute_density_3d_symmetrical_load_ballance(N,h,lsph,box);  // Compute the density of the particles based
  if(err)                                                            // on the cell linked list method for fast  
    fprintf(stderr,"error in compute_density_3d_load_ballanced\n");  // neighbor search

  // --------------------------------------------------------------- //

  t5 = omp_get_wtime();

  kh_clear(0, box->hbegin);
  kh_clear(1, box->hend);

  times[COMPUTE_BLOCKS*run+0] = t1-t0;                               // Time for compute morton Z 3d hash
  times[COMPUTE_BLOCKS*run+1] = t2-t1;                               // Time for sorting the particles' hashes
  times[COMPUTE_BLOCKS*run+2] = t3-t2;                               // Time for reordering all other arrays accordingly
  times[COMPUTE_BLOCKS*run+3] = t4-t3;                               // Time for setting up the interval hash tables
  times[COMPUTE_BLOCKS*run+4] = t5-t4;                               // Time for computing the SPH particle densities

  return 0;
}

/*
 *  Function compute_density_3d_symmetrical_load_ballance:
 *    Computes the SPH density from the particles using cell linked list with
 *    vectorization at the compute_density_3d_chunk level, but the parallelization
 *    done at the level of the outer-most loop of the compute_density_3d_cll_outerOmp
 *    function, not at the chunk level. 
 *
 *    The parallelization is done at the level of unique cell pair instead of cells, 
 *    with the indexes for the cell pairs pre-computed before parallelization. 
 *    
 *    Arguments:
 *       N <int>              : Number of SPH particles to be used in the run
 *       h <double>           : Smoothing Length for the Smoothing Kernel w_bspline
 *       lsph <SPHparticle>   : Array (pointer) of SPH particles to be updated
 *    Returns:
 *       0                    : error code returned
 *       lsph <SPHparticle>   : SPH particle array is updated in the rho field by reference
 */
int compute_density_3d_symmetrical_load_ballance(int N, double h, SPHparticle *lsph, linkedListBox *box){
  int64_t *node_begin,*node_end,*nb_begin,*nb_end;                        // Define the arrays for cell boundaries 
  int64_t max_cell_pair_count = 0;                                        // and the number of cell pairs
  const double kernel_constant = w_bspline_3d_constant(h);               

  max_cell_pair_count = count_box_pairs(box);                             // compute the maximum number of cell pairs
  
  node_begin = (int64_t*)malloc(max_cell_pair_count*sizeof(int64_t));     // allocate space for node_begin
  node_end   = (int64_t*)malloc(max_cell_pair_count*sizeof(int64_t));     // allocate space for node_end
  nb_begin   = (int64_t*)malloc(max_cell_pair_count*sizeof(int64_t));     // allocate space for nb_begin
  nb_end     = (int64_t*)malloc(max_cell_pair_count*sizeof(int64_t));     // allocate space for nb_end

  max_cell_pair_count = setup_unique_box_pairs(box,                       // set the values for cell pairs
                                               node_begin,node_end,
                                               nb_begin,nb_end); 
  
  memset(lsph->rho,(int)0,N*sizeof(double));                              // Pre-initialize the density to zero

                                                                          // Parallelism was moved 
                                                                          // to the level of unique pairs
  #pragma omp parallel for schedule(dynamic,5) proc_bind(master)          // Execute in parallel 
  for(size_t i=0;i<max_cell_pair_count;i+=1){                             // over the unique pairs' array
    double local_rhoi[node_end[i] - node_begin[i]];                       // partial density array for node indexs
    double local_rhoj[  nb_end[i] -   nb_begin[i]];                       // partial density array for nb   indexs

    memset(local_rhoi,(int)0,(node_end[i]-node_begin[i])*sizeof(double)); // initialize node partial density to zero
    memset(local_rhoj,(int)0,    (nb_end[i]-nb_begin[i])*sizeof(double)); // initialize nb partial density to zero

    compute_density_3d_chunk_symmetrical(node_begin[i],node_end[i],       // Compute the density contribution
                                         nb_begin[i],nb_end[i],h,         // from this particular cell pair
                                         lsph->x,lsph->y,lsph->z,         // for both node and nb partial density
                                         lsph->nu,local_rhoi,          
                                         local_rhoj);

    // merging the results can result in race conditions, therefore needs to be serialized
    #pragma omp critical                                                  // this serializes this code section
    {

      for(size_t ii=node_begin[i];ii<node_end[i];ii+=1){                  // iterate over the node_ cell
        lsph->rho[ii] += kernel_constant*local_rhoi[ii-node_begin[i]];    // add the partial density contribution
      }
      
      if(node_begin[i] != nb_begin[i])                                    // if sender and receiver are different
        for(size_t ii=nb_begin[i];ii<nb_end[i];ii+=1){                    // iterate over the nb_ cell
          lsph->rho[ii] += kernel_constant*local_rhoj[ii-nb_begin[i]];    // add the partial density contribution
        }
    }
  }

  free(node_begin); 
  free(node_end);
  free(nb_begin);
  free(nb_end);
  
  return 0;
}

/*
 *  Function compute_density_3d_chunk_symmetrical:
 *    Computes the SPH density contribution to both the node_ cell and the nb_ cell. 
 *    Vectorization in the inner-most loop, but no parallelization. 
 *    The density contribution is symmetrical.     
 * 
 *    Arguments:
 *       N <int>              : Number of SPH particles to be used in the run
 *       h <double>           : Smoothing Length for the Smoothing Kernel w_bspline
 *       x       <double*>    : Array of particles' X positions
 *       y       <double*>    : Array of particles' Y positions
 *       z       <double*>    : Array of particles' Z positions
 *       nu      <double*>    : Array of particles' density weights (i.e. masses)
 *    Returns:
 *       0                    : error code returned
 *       rho       <double*>  : Array of particles' densities
 */
int compute_density_3d_chunk_symmetrical(int64_t node_begin, int64_t node_end,
                                         int64_t nb_begin, int64_t nb_end,double h,
                                         double* restrict x, double* restrict y,
                                         double* restrict z, double* restrict nu,
                                         double* restrict rhoi, double* restrict rhoj){
  const double inv_h = 1./h;

  for(int64_t ii=node_begin;ii<node_end;ii+=1){ // Iterate over the ii index of the chunk
    double xii = x[ii];                         // Load the X component of the ii particle position
    double yii = y[ii];                         // Load the Y component of the ii particle position
    double zii = z[ii];                         // Load the Z component of the ii particle position
   
    #pragma omp simd                            // Hint at the compiler to vectorize the inner most loop
    for(int64_t jj=nb_begin;jj<nb_end;jj+=1){   // Iterate over the each other particle in jj loop
      double q = 0.;                            // Initialize the distance

      double xij = xii-x[jj];                   // Load and subtract jj particle's X position component
      double yij = yii-y[jj];                   // Load and subtract jj particle's Y position component
      double zij = zii-z[jj];                   // Load and subtract jj particle's Z position component

      q += xij*xij;                             // Add the jj contribution to the ii distance in X
      q += yij*yij;                             // Add the jj contribution to the ii distance in Y
      q += zij*zij;                             // Add the jj contribution to the ii distance in Z

      q = sqrt(q)*inv_h;                        // Sqrt to compute the normalized distance, measured in h

      double wij = w_bspline_3d_simd(q);        // compute the smoothing kernel separately for re-use

      rhoi[ii-node_begin] += nu[jj]*wij;        // add the jj contribution to ii density
      rhoj[jj-nb_begin]   += nu[ii]*wij;        // add the ii contribution to jj density
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

int cmp_int64_t(const void *p,const void *q){
  int64_t *data1,*data2;
  data1 = (int64_t*)p;
  data2 = (int64_t*)q;

  if(*data1 < *data2)         // data[0] is the hash value, 
    return -1;                    
  else if(*data1 == *data2)   // in the unsorted array
    return 0;
  else
    return 1;
}

#define swap_loop(N,lsph,temp_swap,member,type) for(int64_t i=0;i<(N);i+=1)                            \
                                                  (temp_swap)[i] = (lsph)->member[(lsph)->hash[2*i+1]];\
                                                memcpy((lsph)->member,temp_swap,(N)*sizeof(type))

void counting_sort(int64_t N, linkedListBox *box, SPHparticle *lsph,void *swap_arr,int64_t *temp_hash){

  double t0,t1,t2,t3,t4,t5,t6;
  double t31,t32,t33,t34;

  t0 = omp_get_wtime();
 
  for(int64_t i=0;i<N;i+=1){
    int ret; 
    khiter_t k = kh_put(0, box->hbegin, lsph->hash[2*i+0], &ret);
    if(ret==1)
      kh_value(box->hbegin, k) = 1;
    else if(ret==0){
      int64_t val = kh_value(box->hbegin, k);
      kh_value(box->hbegin, k) = val + 1;
    }
    else{
      printf("error radixSort init");
      return;
    }
  }

  t1 = omp_get_wtime();

  unsigned int dict_size = kh_size(box->hbegin);
  int64_t hash[dict_size];
  int64_t prefix[dict_size];
  int idx = 0;

  for (khiter_t k = kh_begin(box->hbegin); k != kh_end(box->hbegin); ++k){
    if (kh_exist(box->hbegin, k)){
      hash[idx] = kh_key(box->hbegin,k);
      idx+=1;
    }
  }

  t2 = omp_get_wtime();

  qsort(hash,dict_size,sizeof(int64_t),cmp_int64_t);

  t3 = omp_get_wtime();

  prefix[0] = 0;
  for(int i=1;i<dict_size;i+=1){
    prefix[i] = prefix[i-1];

    khiter_t k = kh_get(0, box->hbegin, hash[i-1]);
    prefix[i] += kh_value(box->hbegin, k);
  }

  t31 = omp_get_wtime();

  for(int i = 0; i< dict_size;i+=1){
    khiter_t kp = kh_get(0, box->hbegin, hash[i]);
    kh_value(box->hbegin, kp) = prefix[i];
  }

  t32 = omp_get_wtime();

  for (khiter_t k = kh_begin(box->hbegin); k != kh_end(box->hbegin); ++k){
    if (kh_exist(box->hbegin, k)){
      int ret;
      int64_t hash = kh_key(box->hbegin,k);
      khiter_t ke = kh_put(1, box->hend, hash, &ret);
      kh_value(box->hend, ke) = kh_value(box->hbegin, k);
    }
  }

  t33 = omp_get_wtime();

  for(int64_t i=0;i<N;i+=1){
    khiter_t kp = kh_get(1, box->hend, lsph->hash[2*i]);
    if (kh_exist(box->hend, kp)){
      lsph->hash[2*i+1] = kh_value(box->hend, kp);
      kh_value(box->hend, kp) += 1;
    }
    else{
      printf("there is an issue\n");
    }
  }

  t34 = omp_get_wtime();

  t4 = omp_get_wtime();

  for(int64_t i=0;i<N;i+=1){
    int64_t idx = lsph->hash[2*i+1];
    temp_hash[2*idx+0] = lsph->hash[2*i+0];
    temp_hash[2*idx+1] = i;
  }

  for(int64_t i=0;i<N;i+=1){
    lsph->hash[2*i+0] = temp_hash[2*i+0];
    lsph->hash[2*i+1] = temp_hash[2*i+1];
  }

  t4 = omp_get_wtime();

  int64_t *int64_temp_swap = (int64_t *)swap_arr;
  swap_loop(N,lsph,int64_temp_swap,id ,int64_t);
  double *double_temp_swap = (double *)swap_arr;
  swap_loop(N,lsph,double_temp_swap,nu ,double);
  swap_loop(N,lsph,double_temp_swap,rho,double);
  swap_loop(N,lsph,double_temp_swap,x  ,double);
  swap_loop(N,lsph,double_temp_swap,y  ,double);
  swap_loop(N,lsph,double_temp_swap,z  ,double);
  swap_loop(N,lsph,double_temp_swap,ux ,double);
  swap_loop(N,lsph,double_temp_swap,uy ,double);
  swap_loop(N,lsph,double_temp_swap,uz ,double);
  swap_loop(N,lsph,double_temp_swap,Fx ,double);
  swap_loop(N,lsph,double_temp_swap,Fy ,double);
  swap_loop(N,lsph,double_temp_swap,Fz ,double);

  t5 = omp_get_wtime();
  
  for(int64_t i=0;i<N;i+=1)
    lsph->hash[i] = lsph->hash[2*i];

  t6 = omp_get_wtime();

  if(dbg){
    printf("1: %lf \n",t1-t0);
    printf("2: %lf \n",t2-t1);
    printf("3: %lf \n",t3-t2);
    printf("4: %lf \n",t4-t3);
    printf("  31: %lf\n",t31-t3);
    printf("  32: %lf\n",t32-t31);
    printf("  33: %lf\n",t33-t32);
    printf("  34: %lf\n",t34-t33);
    printf("  35: %lf\n",t4-t34);
    printf("5: %lf \n",t5-t4);
    printf("6: %lf \n",t6-t5);
  }

  return;
}