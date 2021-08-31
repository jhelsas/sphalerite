#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <sys/time.h>
#include <inttypes.h>
#include <omp.h>

#include "sph_data_types.h"
#include "sph_linked_list.h"

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

#define print_pair_count 1

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

int count_box_pairs(linkedListBox *box){
  int64_t box_pair_count = 0, particle_pair_count = 0;

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
          particle_pair_count += (node_end-node_begin)*(nb_end-nb_begin);
        }
      }
    }
  }

  if(print_pair_count)
    printf("number of particle pairs: %ld\n",particle_pair_count);
  
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

  #pragma omp parallel for num_threads(24) 
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

int main(){

  int err,dbg=0;
  int64_t N = 10000000;
  double h=0.05;
  linkedListBox *box;
  SPHparticle *lsph;

  omp_set_dynamic(0);              /** Explicitly disable dynamic teams **/

  if(dbg)
    printf("hello - 0\n");
  err = SPHparticle_SoA_malloc(N,&lsph);
  if(err)
    printf("error in SPHparticle_SoA_malloc\n");

  if(dbg)
    printf("hello - 1\n");
  err = gen_unif_rdn_pos(N,123123123,lsph);
  if(err)
    printf("error in gen_unif_rdn_pos\n");

  if(dbg)
    printf("hello - 2\n");
  box = (linkedListBox*)malloc(1*sizeof(linkedListBox));

  box->Xmin = -1.0; box->Ymin = -1.0; box->Zmin = -1.0;
  box->Xmax =  2.0; box->Ymax =  2.0; box->Zmax =  2.0;
  box->Nx = (int)( (box->Xmax-box->Xmin)/(2*h) );
  box->Ny = (int)( (box->Ymax-box->Ymin)/(2*h) );
  box->Nz = (int)( (box->Zmax-box->Zmin)/(2*h) );
  box->N  = (box->Nx)*(box->Ny)*(box->Nz);
  double min_val = fmin((box->Xmax-box->Xmin)/box->Nx,fmin((box->Ymax-box->Ymin)/box->Ny,(box->Zmax-box->Zmin)/box->Nz));
  box->width  = (int)( 0.5 + 2*h/min_val );
  box->hbegin = kh_init(0);
  box->hend   = kh_init(1);

  double t0,t1,t2,t3,t4,t5;
  t0 = omp_get_wtime();

  err = compute_hash_MC3D(N,lsph,box);

  t1 = omp_get_wtime();
  
  qsort(lsph->hash,N,2*sizeof(int64_t),compare_int64_t);
  
  t2 = omp_get_wtime();

  void *swap_arr = malloc(N*sizeof(double));
  err = reorder_lsph_SoA(N,lsph,swap_arr);
  if(err)
    printf("error in reorder_lsph_SoA\n");

  t3 = omp_get_wtime();

  if(dbg)
    printf("hello - 6\n");
  err = setup_interval_hashtables(N,lsph,box);
  if(err)
    printf("error in setup_interval_hashtables\n");

  t4 = omp_get_wtime();

  if(dbg)
    printf("hello - 7\n");

  err = compute_density_3d_load_ballanced(N,h,lsph,box);
  if(err)
    printf("error in compute_density_3d_load_ballanced\n");

  t5 = omp_get_wtime();

  printf("fast neighbour search / SoA / outer-openMP / load balanced\n");
  printf("compute_hash_MC3D calc time                 : %.5lf s : %.2lf%%\n",t1-t0,100*(t1-t0)/(t5-t0));
  printf("qsort calc time                             : %.5lf s : %.2lf%%\n",t2-t1,100*(t2-t1)/(t5-t0));
  printf("reorder_lsph_SoA calc time                  : %.5lf s : %.2lf%%\n",t3-t2,100*(t3-t2)/(t5-t0));
  printf("setup_interval_hashtables calc time         : %.5lf s : %.2lf%%\n",t4-t3,100*(t4-t3)/(t5-t0));
  printf("compute_density_3d load balanced calc time  : %.5lf s : %.2lf%%\n",t5-t4,100*(t5-t4)/(t5-t0));
  printf("compute_density_3d load balanced total time : %.5lf s : %.2lf%%\n",t5-t0,100*(t5-t0)/(t5-t0));

  if(dbg)
    printf("hello - 10\n");
  SPHparticleSOA_safe_free(N,&lsph);
  safe_free_box(box);
  free(swap_arr);

  return 0;
}