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

#define print_pair_count 0

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

int compute_density_3d_chunk_symmetrical(int64_t node_begin, int64_t node_end,
                                         int64_t nb_begin, int64_t nb_end,double h,
                                         double* restrict x, double* restrict y,
                                         double* restrict z, double* restrict nu,
                                         double* restrict rhoi, double* restrict rhoj){
  const double inv_h = 1./h;

  for(int64_t ii=node_begin;ii<node_end;ii+=1){
    double xii = x[ii];
    double yii = y[ii];
    double zii = z[ii];
   
    #pragma omp simd 
    for(int64_t jj=nb_begin;jj<nb_end;jj+=1){
      double q = 0.;

      double xij = xii-x[jj];
      double yij = yii-y[jj];
      double zij = zii-z[jj];

      q += xij*xij;
      q += yij*yij;
      q += zij*zij;

      q = sqrt(q)*inv_h;

      double wij = w_bspline_3d_simd(q);

      rhoi[ii-node_begin] += nu[jj]*wij;
      rhoj[jj-nb_begin]   += nu[ii]*wij;
    }
  }

  return 0;
}

int count_box_pairs(linkedListBox *box){
  int64_t pair_count = 0, particle_pair_count = 0;

  for (khint32_t kbegin = kh_begin(box->hbegin); kbegin != kh_end(box->hbegin); kbegin++){
    int64_t node_hash=-1,node_begin=0, node_end=0;
    int64_t nb_begin= 0, nb_end = 0;
    int64_t nblist[(2*box->width+1)*(2*box->width+1)*(2*box->width+1)];

    if (kh_exist(box->hbegin, kbegin)){ // I have to call this!
      khint32_t kend = kh_get(1, box->hend, kh_key(box->hbegin, kbegin));

      node_hash = kh_key(box->hbegin, kbegin);
      node_begin = kh_value(box->hbegin, kbegin);
      node_end   = kh_value(box->hend, kend);

      int res = neighbour_hash_3d(node_hash,nblist,box->width,box);
      for(unsigned int j=0;j<(2*box->width+1)*(2*box->width+1)*(2*box->width+1);j+=1){
        if(nblist[j]>=0){
          nb_begin = kh_value(box->hbegin, kh_get(0, box->hbegin, nblist[j]) );
          nb_end   = kh_value(box->hend  , kh_get(1, box->hend  , nblist[j]) );

          pair_count += 1;
          particle_pair_count += (node_end-node_begin)*(nb_end-nb_begin);
        }
      }
    }
  }

  if(dbg)
    printf("unique ordered particle_pair_count = %ld\n",particle_pair_count);

  return pair_count;
}

int setup_unique_box_pairs(linkedListBox *box,
                           int64_t *node_begin,int64_t *node_end,
                           int64_t *nb_begin,int64_t *nb_end)
{
  int64_t pair_count = 0;

  for (khint32_t kbegin = kh_begin(box->hbegin); kbegin != kh_end(box->hbegin); kbegin++){
    int res;
    int64_t node_hash=-1;
    int64_t nblist[(2*box->width+1)*(2*box->width+1)*(2*box->width+1)];

    if (kh_exist(box->hbegin, kbegin)){ // I have to call this!
      khint32_t kend = kh_get(1, box->hend, kh_key(box->hbegin, kbegin));

      node_hash = kh_key(box->hbegin, kbegin);

      res = neighbour_hash_3d(node_hash,nblist,box->width,box);
      for(unsigned int j=0;j<(2*box->width+1)*(2*box->width+1)*(2*box->width+1);j+=1){
        if(nblist[j]>=0){
          //nb_hash  = nblist[j];

          if(kh_value(box->hbegin, kbegin) <= kh_value(box->hbegin, kh_get(0, box->hbegin, nblist[j]) ))
          {
            node_begin[pair_count] = kh_value(box->hbegin, kbegin);
            node_end[pair_count]   = kh_value(box->hend, kend);
            nb_begin[pair_count]   = kh_value(box->hbegin, kh_get(0, box->hbegin, nblist[j]) );
            nb_end[pair_count]     = kh_value(box->hend  , kh_get(1, box->hend  , nblist[j]) );

            pair_count += 1;
          }
        }
      }
    }
  }

  return pair_count;
}

int compute_density_3d_symmetrical_load_ballance(int N, double h, SPHparticle *lsph, linkedListBox *box){
  int64_t *node_begin,*node_end,*nb_begin,*nb_end;
  int64_t max_box_pair_count = 0;
  const double kernel_constant = w_bspline_3d_constant(h);

  max_box_pair_count = count_box_pairs(box);
  
  node_begin = (int64_t*)malloc(max_box_pair_count*sizeof(int64_t));
  node_end   = (int64_t*)malloc(max_box_pair_count*sizeof(int64_t));
  nb_begin   = (int64_t*)malloc(max_box_pair_count*sizeof(int64_t));
  nb_end     = (int64_t*)malloc(max_box_pair_count*sizeof(int64_t));

  max_box_pair_count = setup_unique_box_pairs(box,node_begin,node_end,nb_begin,nb_end);//setup_box_pairs(box,node_begin,node_end,nb_begin,nb_end);
  
  for(int64_t ii=0;ii<N;ii+=1)
    lsph->rho[ii] = 0.0; 

  #pragma omp parallel for schedule(dynamic,5) proc_bind(master)
  for(size_t i=0;i<max_box_pair_count;i+=1){

    double local_rhoi[node_end[i] - node_begin[i]];
    double local_rhoj[  nb_end[i] -   nb_begin[i]];

    for(size_t ii=0;ii<node_end[i]-node_begin[i];ii+=1)
      local_rhoi[ii] = 0.;

    for(size_t ii=0;ii<nb_end[i]-nb_begin[i];ii+=1)
      local_rhoj[ii] = 0.;

    compute_density_3d_chunk_symmetrical(node_begin[i],node_end[i],nb_begin[i],nb_end[i],h,
                                         lsph->x,lsph->y,lsph->z,lsph->nu,local_rhoi,local_rhoj);

    #pragma omp critical
    {

      for(size_t ii=node_begin[i];ii<node_end[i];ii+=1){
        lsph->rho[ii] += kernel_constant*local_rhoi[ii - node_begin[i]];
      }
      
      if(node_begin[i] != nb_begin[i])
        for(size_t ii=nb_begin[i];ii<nb_end[i];ii+=1){
          lsph->rho[ii] += kernel_constant*local_rhoj[ii - nb_begin[i]];
        }
    }
  }

  free(node_begin); 
  free(node_end);
  free(nb_begin);
  free(nb_end);
  
  return 0;
}

int main(int argc, char **argv){

  bool run_seed = false;
  int err,runs = 1;
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

  for(int run=0;run<runs;run+=1){
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

    err = compute_density_3d_symmetrical_load_ballance(N,h,lsph,box);
    if(err)
      printf("error in compute_density_3d_load_ballanced\n");

    // ------------------------------------------------------ //

    t5 = omp_get_wtime();

    times[run][0] = t1-t0;
    times[run][1] = t2-t1;
    times[run][2] = t3-t2;
    times[run][3] = t4-t3;
    times[run][4] = t5-t4;

    if(dbg){
      printf("fast neighbour search / SoA / outer-openMP / symmetric load balanced\n");
      printf("compute_hash_MC3D calc time                 : %.5lf s : %.2lf%%\n",t1-t0,100*(t1-t0)/(t5-t0));
      printf("qsort calc time                             : %.5lf s : %.2lf%%\n",t2-t1,100*(t2-t1)/(t5-t0));
      printf("reorder_lsph_SoA calc time                  : %.5lf s : %.2lf%%\n",t3-t2,100*(t3-t2)/(t5-t0));
      printf("setup_interval_hashtables calc time         : %.5lf s : %.2lf%%\n",t4-t3,100*(t4-t3)/(t5-t0));
      printf("compute_density_3d load balanced calc time  : %.5lf s : %.2lf%%\n",t5-t4,100*(t5-t4)/(t5-t0));
      printf("compute_density_3d load balanced total time : %.5lf s : %.2lf%%\n",t5-t0,100*(t5-t0)/(t5-t0));
    }
  }

  print_time_stats(runs,times);
  print_sph_particles_density(N,h,seed,runs,lsph,box);

  if(dbg)
    printf("hello - 10\n");
  SPHparticleSOA_safe_free(N,&lsph);
  safe_free_box(box);
  free(swap_arr);

  return 0;
}

