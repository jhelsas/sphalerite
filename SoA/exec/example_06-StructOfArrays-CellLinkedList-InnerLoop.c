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

int compute_density_3d_innerOmp(int N, double h, SPHparticle *lsph, linkedListBox *box);

int compute_density_3d_chunk(int64_t node_begin, int64_t node_end,
                             int64_t nb_begin, int64_t nb_end,double h,
                             double* restrict x, double* restrict y,
                             double* restrict z, double* restrict nu,
                             double* restrict rho);

int count_box_pairs(linkedListBox *box);

int setup_box_pairs(linkedListBox *box,
                    int64_t *node_begin,int64_t *node_end,
                    int64_t *nb_begin,int64_t *nb_end);

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

  bool is_cll = true;
  print_time_stats("SoA,inner",is_cll,N,h,seed,runs,lsph,box,times);
  print_sph_particles_density("SoA,inner",is_cll,N,h,seed,runs,lsph,box);

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

  err = compute_density_3d_innerOmp(N,h,lsph,box);
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

int compute_density_3d_innerOmp(int N, double h, SPHparticle *lsph, linkedListBox *box){
  khiter_t kbegin,kend;
  int64_t node_hash=-1,node_begin=0, node_end=0;
  int64_t nb_begin= 0, nb_end = 0;
  int64_t nblist[(2*box->width+1)*(2*box->width+1)*(2*box->width+1)];

  for (kbegin = kh_begin(box->hbegin); kbegin != kh_end(box->hbegin); kbegin++){
    
    if (kh_exist(box->hbegin, kbegin)){
      kend = kh_get(1, box->hend, kh_key(box->hbegin, kbegin));
      node_hash = kh_key(box->hbegin, kbegin);
      node_begin = kh_value(box->hbegin, kbegin);
      node_end   = kh_value(box->hend, kend);

      for(int64_t ii=node_begin;ii<node_end;ii+=1)// this loop inside was the problem
        lsph->rho[ii] = 0.0; 

      neighbour_hash_3d(node_hash,nblist,box->width,box);
      for(unsigned int j=0;j<(2*box->width+1)*(2*box->width+1)*(2*box->width+1);j+=1){
        if(nblist[j]>=0){
          //nb_hash  = nblist[j];
          nb_begin = kh_value(box->hbegin, kh_get(0, box->hbegin, nblist[j]) );
          nb_end   = kh_value(box->hend  , kh_get(1, box->hend  , nblist[j]) );

          compute_density_3d_chunk(node_begin,node_end,nb_begin,nb_end,h,
                                   lsph->x,lsph->y,lsph->z,lsph->nu,lsph->rho);
        }
      }
    }
  }

  return 0;
}

int compute_density_3d_chunk(int64_t node_begin, int64_t node_end,
                             int64_t nb_begin, int64_t nb_end,double h,
                             double* restrict x, double* restrict y,
                             double* restrict z, double* restrict nu,
                             double* restrict rho){

  #pragma omp parallel for
  for(int64_t ii=node_begin;ii<node_end;ii+=1){
    double xii = x[ii];
    double yii = y[ii];
    double zii = z[ii];
    double rhoii = 0.0;
   
    #pragma omp simd reduction(+:rhoii) aligned(x,y,z,nu) 
    for(int64_t jj=nb_begin;jj<nb_end;jj+=1){
      double q = 0.;

      double xij = xii-x[jj];
      double yij = yii-y[jj];
      double zij = zii-z[jj];

      q += xij*xij;
      q += yij*yij;
      q += zij*zij;

      q = sqrt(q);

      rhoii += nu[jj]*w_bspline_3d(q,h);
    }
    rho[ii] += rhoii;
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
