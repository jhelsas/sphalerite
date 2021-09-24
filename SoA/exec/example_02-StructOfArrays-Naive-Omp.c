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

int compute_density_3d_naive_omp(int N,double h,
                                 double* restrict x, double* restrict y,
                                 double* restrict z, double* restrict nu,
                                 double* restrict rho);

double w_bspline_3d(double r,double h);

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

  bool is_cll = false;
  print_time_stats("omp",is_cll,N,h,seed,runs,lsph,box,times);
  print_sph_particles_density("omp",is_cll,N,h,seed,runs,lsph,box);

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

  double t0,t1;

  t0 = omp_get_wtime();
  
  compute_density_3d_naive_omp(N,h,lsph->x,lsph->y,lsph->z,lsph->nu,lsph->rho);

  t1 = omp_get_wtime();

  printf("compute_density_3d SoA naive omp simd tiled calc time : %lf s \n",t1-t0);

  // ------------------------------------------------------ //

  times[5*run+0] = t1-t0;
  times[5*run+1] =    0.;
  times[5*run+2] =    0.;
  times[5*run+3] =    0.;
  times[5*run+4] =    0.;

  if(dbg){
    printf("compute_density_3d SoA naive omp simd tiled calc time : %lf s \n",t1-t0);
  }

  return 0;
}

int compute_density_3d_naive_omp(int N,double h,
                                 double* restrict x, double* restrict y,
                                 double* restrict z,double* restrict nu,
                                 double* restrict rho){
  #pragma omp parallel for
  for(int64_t ii=0;ii<N;ii+=1){
    rho[ii] = 0;
    for(int64_t jj=0;jj<N;jj+=1){
      double dist = 0.;

      dist += (x[ii]-x[jj])*(x[ii]-x[jj]);
      dist += (y[ii]-y[jj])*(y[ii]-y[jj]);
      dist += (z[ii]-z[jj])*(z[ii]-z[jj]);

      dist = sqrt(dist);

      rho[ii] += nu[jj]*w_bspline_3d(dist,h);
    }
  }

  return 0;
}

double w_bspline_3d(double r,double h){
  const double A_d = 3./(2.*M_PI*h*h*h);
  double q=0.;
  
  if(r<0||h<=0.)
    exit(10);
  
  q = r/h;
  if(q<=1)
    return A_d*(2./3.-q*q + q*q*q/2.0);
  else if((1.<=q)&&(q<2.))
    return A_d*(1./6.)*(2.-q)*(2.-q)*(2.-q);
  else 
    return 0.;
}